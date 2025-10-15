pub mod run;
pub mod level;
pub mod in_memory_mbtree;
use in_memory_mbtree::InMemoryMBTree;
use merkle_btree_storage::{traits::BPlusTreeNodeIO, RangeProof as MBTreeRangeProof};
use run::{LevelRun, RunProof, reconstruct_run_proof, RunFilterSize};
use serde::{Serialize, Deserialize};
use utils::{cacher::CacheManagerOld, config::Configs, pager::old_state_pager::{InMemStateIteratorOld, StateIteratorOld}, types::{compute_concatenate_hash, AddrKey, CompoundKey, StateValue}, MemCost, OpenOptions, Read, Write};
use level::Level;
use primitive_types::H256;
use std::fmt::{Debug, Formatter, Error};

/* COLE consists of:
    (i) a reference of configs that include params
    (ii) an in-memory MB-Tree as the authenticated index
    (iii) a vector of levels that stores each level's LevelRuns
 */
pub struct Cole<'a> {
    pub configs: &'a Configs,
    pub mem_mht: InMemoryMBTree,
    pub levels: Vec<Level>,
    pub run_id_cnt: u32, // this helps generate a new run_id
    pub cache_manager: CacheManagerOld,
    pub latest_block_id: u32,
}

impl<'a> Cole<'a> {
    // create a new index using configs,
    pub fn new(configs: &'a Configs) -> Self {
        Self {
            configs,
            mem_mht: InMemoryMBTree::new(configs.fanout * 2), // an empty in-memory MB-Tree, max_fanout = exp_fnaout * 2
            levels: Vec::new(), // empty levels' vector
            run_id_cnt: 0, // initiate the counter to be 0
            cache_manager: CacheManagerOld::new(),
            latest_block_id: 0, // initiate the latest block id to 0
        }
    }

    fn get_meta(&mut self) -> usize{
        let path = self.get_tree_meta_path();
        let mut file = OpenOptions::new().create(true).read(true).write(true).open(&path).unwrap();
        // read level len
        let mut level_len_bytes = [0u8; 4];
        let mut level_len: u32 = 0;
        match file.read_exact(&mut level_len_bytes) {
            Ok(_) => {
                level_len = u32::from_be_bytes(level_len_bytes);
            },
            Err(_) => {}
        }
        // read run_id_cnt
        let mut run_id_cnt_bytes = [0u8; 4];
        let mut run_id_cnt: u32 = 0;
        match file.read_exact(&mut run_id_cnt_bytes) {
            Ok(_) => {
                run_id_cnt = u32::from_be_bytes(run_id_cnt_bytes);
            },
            Err(_) => {}
        }
        self.run_id_cnt = run_id_cnt;
        // read latest_block_id
        let mut latest_block_id_bytes = [0u8; 4];
        let mut latest_block_id: u32 = 0;
        match file.read_exact(&mut latest_block_id_bytes) {
            Ok(_) => {
                latest_block_id = u32::from_be_bytes(latest_block_id_bytes);
            },
            Err(_) => {}
        }
        self.latest_block_id = latest_block_id;
        // read mem mht
        let mut mem_tree_bytes = Vec::<u8>::new();
        match file.read_to_end(&mut mem_tree_bytes) {
            Ok(_) => {
                self.mem_mht = bincode::deserialize(&mem_tree_bytes).unwrap();
            },
            Err(_) => {}
        }
        return level_len as usize;
    } 

    // load a new index using configs,
    pub fn load(configs: &'a Configs) -> Self {
        let mut ret = Self::new(configs);
        let level_len = ret.get_meta();
        // load levels
        for i in 0..level_len {
            let level = Level::load(i as u32, configs);
            ret.levels.push(level);
        }
        return ret;
    }

    fn get_tree_meta_path(&self) -> String {
        format!("{}/mht", &self.configs.dir_name)
    }

    fn new_run_id(&mut self) -> u32 {
        // increment the run_id and return it
        self.run_id_cnt += 1;
        return self.run_id_cnt;
    }

    pub fn insert(&mut self, state: (CompoundKey, StateValue)) {
        let (compound_key, value) = state;
        // update the latest block id
        self.latest_block_id = compound_key.version;
        // directly insert the state into the mem_mht
        merkle_btree_storage::insert(&mut self.mem_mht, compound_key, value);
        // compute the in-memory threshold
        let in_mem_thres = self.configs.base_state_num;
        if self.mem_mht.key_num as usize == in_mem_thres {
            let start = std::time::Instant::now();
            // the in-memory mb-tree is full, the data should be merged to the run in the disk-level
            let key_values = self.mem_mht.load_all_key_values();
            // clear the in-mem
            self.mem_mht.clear();
            let iter = InMemStateIteratorOld::create(key_values);
            let inputs = vec![iter];
            let run_id = self.new_run_id();
            let level_id = 0; // the first on-disk level's id is 0
            let level_num_of_run = match self.levels.get(level_id as usize) {
                Some(level) => {
                    level.run_vec.len()
                },
                None => {
                    0
                }
            };
            let run = LevelRun::construct_run_by_in_memory_merge(inputs, run_id, level_id, &self.configs.dir_name, self.configs.epsilon, self.configs.fanout, self.configs.max_num_of_states_in_a_run(level_id), level_num_of_run, self.configs.size_ratio);
            let elapse = start.elapsed().as_nanos();
            println!("flush time: {:?}", elapse);
            match self.levels.get_mut(level_id as usize) {
                Some(level_ref) => {
                    level_ref.run_vec.push(run); // push the new run to the end for efficiency, but query runs in the revert sort
                    // level_ref.run_vec.insert(0, run); // always insert the new run to the front, so that the latest states are at the front of the level
                },
                None => {
                    let mut level = Level::new(level_id); // the level with level_id does not exist, so create a new one
                    level.run_vec.push(run); // .insert(0, run); add the run to the end
                    self.levels.push(level); // push the new level to the level vector
                }
            }
            // iteratively merge the levels if the level reaches the capacity
            self.check_and_merge();
        }
    }

    // from the first disk level to the last disk level, check whether a level reaches the capacity, if so, merge all the runs in the level to the next level
    pub fn check_and_merge(&mut self) {
        let mut level_id = 0; // start from 0 disk level
        while self.levels[level_id].level_reach_capacity(&self.configs) {
            let level_ref = self.levels.get_mut(level_id).unwrap();
            let mut iters = Vec::<StateIteratorOld>::new();
            let n = level_ref.run_vec.len();
            // transform each run in the level to the state iterator
            let mut run_id_vec = Vec::<u32>::new();
            for _ in 0..n {
                let run = level_ref.run_vec.remove(0);
                run_id_vec.push(run.run_id);
                let iter = run.state_reader.to_state_iter_old();
                iters.push(iter);
            }
            // create a new run_id
            let run_id = self.new_run_id();
            // next disk level's id
            let next_level_id = level_id + 1;
            let level_num_of_run = match self.levels.get(level_id as usize) {
                Some(level) => {
                    level.run_vec.len()
                },
                None => {
                    0
                }
            };
            let new_run = LevelRun::construct_run_by_merge(iters, run_id, next_level_id as u32, &self.configs.dir_name, self.configs.epsilon, self.configs.fanout, self.configs.max_num_of_states_in_a_run(next_level_id as u32), level_num_of_run, self.configs.size_ratio);
            match self.levels.get_mut(next_level_id) {
                // the next level exists, insert the new run to the front of the run_vec
                Some(level_ref) => {
                    level_ref.run_vec.push(new_run);
                    // level_ref.run_vec.insert(0, new_run);
                },
                None => {
                    // the level with next_level_id does not exist, should create a new level first
                    let mut level = Level::new(next_level_id as u32);
                    level.run_vec.push(new_run);
                    // level.run_vec.insert(0, new_run);
                    self.levels.push(level);
                }
            }
            // remove the merged files in level_id by using multi-threads; not that we do not need to wait for the ending of the thread.
            Level::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
            level_id += 1;
        }
    }

    pub fn search_latest_state_value(&mut self, addr_key: AddrKey) -> Option<StateValue> {
        // compute the boundary compound key
        let upper_key = CompoundKey {
            addr: addr_key,
            version: u32::MAX,
        };
        // search in the in-memory MB-Tree
        match merkle_btree_storage::search_without_proof(&mut self.mem_mht, upper_key) {
            Some((read_key, read_v)) => {
                if read_key.addr == addr_key {
                    // matches the addresses and should be the latest value since latest value should be in the upper levels
                    return Some(read_v);
                }
            },
            None => {},
        }
        // search other levels on the disk
        for level in &mut self.levels {
            for run in level.run_vec.iter_mut().rev() {
                let res = run.search_run(addr_key, &self.configs, &mut self.cache_manager);
                if res.is_some() {
                    let res = res.unwrap();
                    return Some(res.1);
                }
            }
        }
        return None;
    }

    pub fn search_latest_state_value_on_disk(&mut self, addr_key: AddrKey) -> Option<(u32, StateValue)> {
        // search other levels on the disk
        let mut cnt = 0;
        for level in &mut self.levels {
            for run in level.run_vec.iter_mut().rev() {
                cnt += 1;
                let res = run.search_run(addr_key, &self.configs, &mut self.cache_manager);
                if res.is_some() {
                    let res = res.unwrap();
                    return Some((cnt, res.1));
                }
            }
        }
        return None;
    }

    // generate the range proof given the addr_key and two version ranges
    pub fn search_with_proof(&mut self, addr_key: AddrKey, low_version: u32, upper_version: u32) -> ColeProof {
        let mut proof = ColeProof::new();
        // generate the two compound keys
        let low_key = CompoundKey::new(addr_key, low_version);
        let upper_key = CompoundKey::new(addr_key, upper_version);
        // search the in-memory mbtree
        let (r, p) = merkle_btree_storage::get_range_proof(&mut self.mem_mht, low_key, upper_key);
        let mut rest_is_hash = false;
        if r.is_some() {
            // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
            // there is no need to prove_range the run
            let left_most_result = r.as_ref().unwrap()[0].0;
            let result_version = left_most_result.version;
            if result_version < low_version {
                rest_is_hash = true;
            }
        }
        // include the result and proof
        proof.in_mem_level = (r, p);
        // search the runs in all disk levels
        
        for level in &mut self.levels {
            let mut level_proof = vec![];
            for run in level.run_vec.iter_mut().rev() {
                // decide to add the run's proof or the run's digest
                if rest_is_hash == false {
                    let (r, p) = run.prove_range(addr_key, low_version, upper_version, &self.configs, &mut self.cache_manager);
                    if r.is_some() {
                        // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
                        // there is no need to prove_range the run
                        let left_most_result = r.as_ref().unwrap()[1]; // index 1 because of the [0] is the boundary element whose addr is not addr_key
                        let result_version = left_most_result.0.version;
                        if result_version < low_version {
                            rest_is_hash = true;
                        }
                    }
                    level_proof.push((r, RunProofOrHash::Proof(p)));
                } else {
                    level_proof.push((None, RunProofOrHash::Hash(run.digest)));
                }
            }
            proof.disk_level.push(level_proof);
        }
        return proof;
    }

    fn update_manifest(&self) {
        // first persist all levels
        for level in &self.levels {
            level.persist_level(&self.configs);
        }
        // persist level len
        let level_len = self.levels.len() as u32;
        let mut bytes = level_len.to_be_bytes().to_vec();
        // persist run_id_cnt
        let run_id_cnt = self.run_id_cnt;
        bytes.extend(run_id_cnt.to_be_bytes());
        // persist latest_block_id
        let latest_block_id = self.latest_block_id;
        bytes.extend(latest_block_id.to_be_bytes());
        // serialize mem mht and persist it
        let mht_bytes = bincode::serialize(&self.mem_mht).unwrap();
        bytes.extend(&mht_bytes);
        // persist the bytes to the manifest file
        let path = self.get_tree_meta_path();
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&path).unwrap();
        file.write_all(&mut bytes).unwrap();
    }

    // compute the digest of COLE
    pub fn compute_digest(&self) -> H256 {
        // collect each disk-level's hash
        let mut level_hash_vec: Vec<H256> = self.levels.iter().map(|level| level.compute_digest()).collect();
        // in mem mbtree's root hash
        let in_mem_root = self.mem_mht.get_root_hash();
        // put the mbtree's hash to the front
        level_hash_vec.insert(0, in_mem_root);
        compute_concatenate_hash(&level_hash_vec)
    }

    // compute filter cost
    pub fn filter_cost(&self) -> RunFilterSize {
        let mut filter_size = RunFilterSize::new(0);
        for level in &self.levels {
            filter_size.add(&level.filter_cost());
        }
        return filter_size;
    }

    pub fn memory_cost(&self) -> MemCost {
        let filter_size = self.filter_cost().filter_size;
        // let cache_size = self.cache_manager.compute_cacher_size();
        let mht_size = bincode::serialize(&self.mem_mht).unwrap().len();
        MemCost::new(0, filter_size, mht_size)
    }

    pub fn print_structure_info(&self) {
        // println!("in mem num: {:?}", self.mem_mht.get_key_num());
        println!("num of disk levels: {}", self.levels.len());
        println!("each level info:");
        for level in &self.levels {
            println!("level num of runs: {}", level.run_vec.len());
            for run in &level.run_vec {
                println!("run id: {}, state len: {}", run.run_id, run.state_reader.num_states);
            }
        }
    }
}

pub fn verify_and_collect_result(addr_key: AddrKey, low_version: u32, upper_version: u32, root_hash: H256, proof: &ColeProof, fanout: usize) -> (bool, Option<Vec<(CompoundKey, StateValue)>>) {
    let mut level_roots = Vec::<H256>::new();
    // first reconstruct the in_memory_proof
    let low_key = CompoundKey::new(addr_key, low_version);
    let upper_key = CompoundKey::new(addr_key, upper_version);
    let in_memory_result = &proof.in_mem_level.0;
    let h = merkle_btree_storage::reconstruct_range_proof(low_key, upper_key, in_memory_result, &proof.in_mem_level.1);
    level_roots.push(h);
    let mut merge_result: Vec<(CompoundKey, StateValue)> = vec![];
    let mut rest_is_hash = false;
    if in_memory_result.is_some() {
        let left_most_result = in_memory_result.as_ref().unwrap()[0].0;
        let result_version = left_most_result.version;
        if result_version < low_version {
            rest_is_hash = true;
        }
        let r: Vec<(CompoundKey, StateValue)> = in_memory_result.as_ref().unwrap().iter().map(|(k, v)| {
            (*k, *v)
        }).collect();
        merge_result.extend(r);
    }
    
    for level in &proof.disk_level {
        let mut level_h_vec: Vec<H256> = Vec::new();
        for run in level {
            let r = &run.0;
            let p = &run.1;
            match p {
                RunProofOrHash::Hash(h) => {
                    if rest_is_hash == false {
                        // in-complete result, return false
                        return (false, None);
                    }
                    level_h_vec.push(*h);
                },
                RunProofOrHash::Proof(proof) => {
                    if rest_is_hash == true {
                        // in-complete result, return false
                        return (false, None);
                    }
                    let (_, h) = reconstruct_run_proof(addr_key, low_version, upper_version, r, proof, fanout);
                    level_h_vec.push(h);
                }
            }
            if r.is_some() {
                let left_most_result = r.as_ref().unwrap()[1]; // index 1 because of the [0] is the boundary element whose addr is not addr_key
                let result_version = left_most_result.0.version;
                if result_version < low_version {
                    rest_is_hash = true;
                }
                merge_result.extend(r.as_ref().unwrap());
            }
        }
        let level_h = compute_concatenate_hash(&level_h_vec);
        level_roots.push(level_h);
    }
    let reconstruct_root = compute_concatenate_hash(&level_roots);
    if reconstruct_root != root_hash {
        return (false, None);
    }
    merge_result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    merge_result = merge_result.into_iter().filter(|r| r.0 >= low_key && r.0 <= upper_key).collect();
    return (true, Some(merge_result));
}

impl<'a> Drop for Cole<'a> {
    fn drop(&mut self) {
        self.update_manifest();
    }
}

impl<'a> Debug for Cole<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "in mem num: {:?}\n", self.mem_mht.get_key_num()).unwrap();
        write!(f, "run_id_cnt: {}\n", self.run_id_cnt).unwrap();
        for level in &self.levels {
            write!(f, "level: {:?}\n", level).unwrap();
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RunProofOrHash {
    Proof(RunProof),
    Hash(H256),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColeProof {
    pub in_mem_level: (Option<Vec<(CompoundKey, StateValue)>>, MBTreeRangeProof<CompoundKey, StateValue>),
    pub disk_level: Vec<Vec<(Option<Vec<(CompoundKey, StateValue)>>, RunProofOrHash)>>,
}

impl ColeProof {
    // initiate the cole's proof
    pub fn new() -> Self {
        let in_mem_level = (None, MBTreeRangeProof::default());
        let disk_level = Vec::<Vec::<(Option<Vec<(CompoundKey, StateValue)>>, RunProofOrHash)>>::new();
        Self {
            in_mem_level,
            disk_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use primitive_types::{H160, H256};
    use utils::compute_cole_size_breakdown;

    #[test]
    fn test_query_cole_old_design() {
        let num_of_contract = 100;
        let num_of_addr = 1000;
        let num_of_version = 10;
        let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 23;
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 45000;
        let size_ratio = 10;
        let configs = Configs::new(fanout, epsilon, dir_name.to_string(), base_state_num, size_ratio, false);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
                addr_key_vec.push(addr_key);
            }
        }
        // let multi_factor = 2;
        for k in 1..=num_of_version {
            for addr_key in &addr_key_vec {
                let compound_key = CompoundKey::new(*addr_key, k * 2);
                state_vec.push((compound_key, H256::from_low_u64_be((k * 2) as u64).into()));
            }
        }
        /* for addr_key in &addr_key_vec {
            for k in 1..=num_of_version {
                let compound_key = CompoundKey::new_with_addr_key(*addr_key, k * 2);
                state_vec.push(CompoundKeyValue::new_with_compound_key(compound_key, H256::from_low_u64_be((k * 2) as u64).into()));
            }
        } */
        
        let mut cole = Cole::new(&configs);
        let start = std::time::Instant::now();
        for state in &state_vec {
            cole.insert(*state);
        }
        let elapse = start.elapsed().as_nanos();
        println!("average insert: {:?}", elapse / n as u128);
        println!("cole memory cost: {:?}", cole.memory_cost());
        println!("{:?}", cole);
        drop(cole);

        let mut load_cole =  Cole::load(&configs);
        let root = load_cole.compute_digest();
        let storage_size = compute_cole_size_breakdown(dir_name);
        println!("storage size: {:?}", storage_size);
/*         let in_mem_states = load_cole.mem_mht.load_all_key_values();
        let load_values: Vec<CompoundKeyValue> = in_mem_states.into_iter().map(|(k, v)| { CompoundKeyValue::new_with_compound_key(k, v.value)}).collect();
        println!("in mem: {:?}", load_values);
        for level in &mut load_cole.levels {
            for run in &mut level.run_vec {
                println!("{:?}", run.load_states());
            }
        } */
        // load_cole.levels[0].run_vec[0].print_models();
        // addr_key_vec.sort();
        // let addr = addr_key_vec[0];
        // let proof = load_cole.search_with_proof(addr, 0, num_of_version * 2);
        // let root = load_cole.compute_digest();

        
        let start = std::time::Instant::now();
        for (i, addr) in addr_key_vec.iter().enumerate() {
            let r = load_cole.search_latest_state_value(*addr);
            let true_value = StateValue(H256::from_low_u64_be((num_of_version* 2) as u64));
            
            if r.unwrap() != true_value {
                println!("false addr: {:?}", addr);
                println!("r: {:?}, true value: {:?}", r, true_value);
                println!("i: {}", i);
            }
        }
        let elapse = start.elapsed().as_nanos();
        println!("average point query: {:?}", elapse / n as u128);

        let mut search_prove = 0;
        for addr in &addr_key_vec {
            let start = std::time::Instant::now();
            let proof = load_cole.search_with_proof(*addr, 0, num_of_version * 2);
            let (b, r) = verify_and_collect_result(*addr, 0, num_of_version * 2, root, &proof, configs.fanout);
            let elapse = start.elapsed().as_nanos();
            search_prove += elapse;
            let true_states: Vec<(CompoundKey, StateValue)> = (1..=num_of_version).map(|k| {
                let compound_key = CompoundKey::new(*addr, k*2);
                (compound_key, H256::from_low_u64_be((k * 2) as u64).into())
            }).collect();
            if b == false {
                println!("false");
            }
            let r = r.unwrap();
            if true_states != r {
                println!("true states: {:?}", true_states);
                println!("r: {:?}", r);
            }
        }
        println!("avg search prove: {}", search_prove / (num_of_contract * num_of_addr) as u128);

        let mut search_prove = 0;
        for addr in &addr_key_vec {
            for i in 1..=num_of_version {
                let start = std::time::Instant::now();
                let proof = load_cole.search_with_proof(*addr, (2 * i) as u32, (2 * i) as u32);
                let (b, r) = verify_and_collect_result(*addr, (2 * i) as u32, (2 * i) as u32, root, &proof, configs.fanout);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                if b == false {
                    println!("false");
                }
                let r = r.unwrap();
                let compound_key = CompoundKey::new(*addr, i * 2);
                let value = (compound_key, StateValue(H256::from_low_u64_be((i * 2) as u64)));
                let true_states = vec![value];
                if true_states != r {
                    println!("true states: {:?}", true_states);
                    println!("r: {:?}", r);
                }
            }
        }
        println!("avg search prove: {}", search_prove / (num_of_contract * num_of_addr * num_of_version) as u128);
    }
}