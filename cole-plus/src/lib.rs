pub mod in_memory_mbtree;
pub mod in_memory_postree;
pub mod run;
pub mod level;
use in_memory_postree::InMemoryPOSTree;
use level::Level;
use pattern_oridented_split_tree::{traits::POSTreeNodeIO, POSTreeRangeProof};
use primitive_types::H256;
use std::fmt::{Debug, Formatter, Error};
use run::{reconstruct_run_proof, LevelRun, RunFilterSize, RunProof};
use utils::{cacher::CacheManager, config::Configs, pager::{cdc_mht::{CDCTreeReader, VerObject}, state_pager::StateIterator, upper_mht::UpperMHTReader}, types::{compute_concatenate_hash, AddrKey, CompoundKey, StateValue}, MemCost, OpenOptions, Read, Write};
use serde::{Serialize, Deserialize};
use utils::DEFAULT_MAX_NODE_CAPACITY;
use std::fs;

pub struct InMemGroup {
    pub mem_mht: InMemoryPOSTree, // in-mem POS-tree
    pub max_block_id: u32,
}

impl Debug for InMemGroup {
    fn fmt(&self, _: &mut Formatter<'_>) -> Result<(), Error> {
        println!("mem mht: ");
        self.mem_mht.print_tree();
        Ok(())
    }
}

impl InMemGroup {
    pub fn new(exp_fanout: usize, max_fanout: usize) -> Self {
        Self {
            mem_mht: InMemoryPOSTree::new(exp_fanout, max_fanout),
            max_block_id: 0,
        }
    }
    pub fn clear(&mut self) {
        self.mem_mht.clear();
        self.max_block_id = 0;
    }

    pub fn get_mem_mht_ref(&self) -> &InMemoryPOSTree {
        &self.mem_mht
    }
}
/* COLE consists of:
    (i) a reference of configs that include params
    (ii) an in-memory POS-Tree as the authenticated index
    (iii) a vector of levels that stores each level's LevelRuns
 */
pub struct ColePlus<'a> {
    pub configs: &'a Configs,
    pub in_mem_group: [InMemGroup; 2],
    pub in_mem_write_group_flag: bool,
    pub levels: Vec<Level>,
    pub run_id_cnt: u32, // this helps generate a new run_id
    pub cache_manager: CacheManager,
    pub latest_block_id: u32, // this helps keep the latest x blocks
}

impl<'a> ColePlus<'a> {
    // create a new index using configs,
    pub fn new(configs: &'a Configs) -> Self {
        Self {
            configs,
            in_mem_group: [InMemGroup::new(configs.fanout, DEFAULT_MAX_NODE_CAPACITY), InMemGroup::new(configs.fanout, DEFAULT_MAX_NODE_CAPACITY)],
            in_mem_write_group_flag: true,
            levels: Vec::new(), // empty levels' vector
            run_id_cnt: 0, // initiate the counter to be 0
            cache_manager: CacheManager::new(),
            latest_block_id: 0, // initiate the latest block id to 0
        }
    }

    fn get_meta(&mut self) -> usize {
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
        // read mem mht in the write group
        let mut write_mht_len_bytes = [0u8; 4];
        let mut write_mht_len = 0;
        match file.read_exact(&mut write_mht_len_bytes) {
            Ok(_) => {
                write_mht_len = u32::from_be_bytes(write_mht_len_bytes);
            },
            Err(_) => {}
        }
        let mut write_mht_bytes = vec![0u8; write_mht_len as usize];
        let write_index = self.get_write_in_mem_group_index();
        match file.read_exact(&mut write_mht_bytes) {
            Ok(_) => {
                self.in_mem_group[write_index].mem_mht = bincode::deserialize(&write_mht_bytes).unwrap();
            },
            Err(_) => {},
        }
        // read mem mht in the merge group
        let mut merge_mht_len_bytes = [0u8; 4];
        let mut merge_mht_len = 0;
        match file.read_exact(&mut merge_mht_len_bytes) {
            Ok(_) => {
                merge_mht_len = u32::from_be_bytes(merge_mht_len_bytes);
            },
            Err(_) => {}
        }
        let mut merge_mht_bytes = vec![0u8; merge_mht_len as usize];
        let merge_index = self.get_merge_in_mem_group_index();
        match file.read_exact(&mut merge_mht_bytes) {
            Ok(_) => {
                self.in_mem_group[merge_index].mem_mht = bincode::deserialize(&merge_mht_bytes).unwrap();
            },
            Err(_) => {},
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

    pub fn new_run_id(&mut self) -> u32 {
        // increment the run_id and return it
        self.run_id_cnt += 1;
        return self.run_id_cnt;
    }

    pub fn get_write_in_mem_group_index(&self) -> usize {
        if self.in_mem_write_group_flag == true {
            // the first in_mem group is the write group
            0
        } else {
            // the second in_mem group is the write group
            1
        }
    }

    pub fn get_merge_in_mem_group_index(&self) -> usize {
        if self.in_mem_write_group_flag == true {
            // the second in_mem group is the merge group
            1
        } else {
            // the first in_mem group is the merge group
            0
        }
    }

    fn switch_in_mem_group(&mut self) {
        // reverse the flag of write group
        if self.in_mem_write_group_flag == true {
            self.in_mem_write_group_flag = false;
        } else {
            self.in_mem_write_group_flag = true;
        }
    }

    pub fn insert(&mut self, state: (AddrKey, u32, StateValue)) {
        let (addr_key, ver, value) = state;
        // update the latest block id
        self.latest_block_id = ver;
        // compute the in-memory threshold
        let in_mem_thres = (self.configs.base_state_num as f64 * 0.5) as usize;
        // get the write in_mem group index
        let write_index = self.get_write_in_mem_group_index();
        // insert the state to the tree of write group
        let tree_ref = &mut self.in_mem_group[write_index].mem_mht;
        pattern_oridented_split_tree::insert(tree_ref, CompoundKey::new(addr_key, ver), value);
        if self.configs.test_in_mem_roll == true {
            // reorg-related code
            self.in_mem_group[write_index].max_block_id = ver;
        }
        // check wheither the write group tree is full
        if tree_ref.key_num as usize == in_mem_thres {
            // get the merge group index
            let merge_index = self.get_merge_in_mem_group_index();
            if self.in_mem_group[merge_index].mem_mht.key_num as usize == in_mem_thres {
                // merge group is full, the data should be merged to the run in the disk-level
                let state_vec = self.in_mem_group[merge_index].mem_mht.load_all_key_values();
                self.in_mem_group[merge_index].clear();
                let run_id = self.new_run_id();
                let level_id = 0; // the first on-disk level's id is 0
                // get level's num of run to estimate filter size
                let level_num_of_run = match self.levels.get(level_id as usize) {
                    Some(level) => {
                        level.run_vec.len()
                    },
                    None => {
                        0
                    }
                };
                
                let run = LevelRun::construct_run_by_in_memory_collection(state_vec, run_id, level_id, &self.configs.dir_name, self.configs.fanout, self.configs.max_num_of_states_in_a_run(level_id), level_num_of_run, self.configs.size_ratio, self.configs.is_pruned);
                // println!("flush time: {:?}", elapse);
                match self.levels.get_mut(level_id as usize) {
                    Some(level_ref) => {
                        level_ref.run_vec.push(run); // push the new run to the end for efficiency, but query runs in the revert sort
                    },
                    None => {
                        let mut level = Level::new(level_id); // the level with level_id does not exist, so create a new one
                        level.run_vec.push(run); 
                        self.levels.push(level); // push the new level to the level vector
                    }
                }
            }
            self.switch_in_mem_group();
            // iteratively merge the levels if the level reaches the capacity
            self.check_and_merge();
            // println!("merge time: {:?}", elapse);
        }
    }

    // from the first disk level to the last disk level, check whether a level reaches the capacity, if so, merge all the runs in the level to the next level
    pub fn check_and_merge(&mut self,) {
        if self.configs.test_disk_roll == true {
            // large rewind related code
            if self.latest_block_id % 100 == 0 {
                // for each 500 blocks, log the checkpoint
                let mut levels_logs = vec![];
                for level in &self.levels {
                    let mut runs = vec![];
                    for run in level.run_vec.iter() {
                        runs.push((run.block_range_low, run.block_range_high));
                    }
                    levels_logs.push(runs);
                }
                let json_string = serde_json::to_string(&levels_logs).unwrap();
                let log_file_name = self.get_block_range_checkpoint_log(self.latest_block_id);
                fs::write(log_file_name, json_string).unwrap();
            }
        }
        let mut level_id = 0; // start from 0 disk level
        while level_id < self.levels.len() {
            if self.levels[level_id].level_reach_capacity(&self.configs) {
                let level_ref = self.levels.get_mut(level_id).unwrap();
                let mut state_iters = Vec::<StateIterator>::new();
                let mut lower_cdc_tree_readers = Vec::<CDCTreeReader>::new();
                let mut upper_mht_readers: Vec::<(u32, UpperMHTReader)> = Vec::<(u32, UpperMHTReader)>::new();
                let n = level_ref.run_vec.len();
                // transform each run in the level to the state iterator
                let mut run_id_vec = Vec::<u32>::new();
                // note that the runs in level_ref are ordered from older to newer
                for _ in 0..n {
                    let run = level_ref.run_vec.remove(0);
                    run_id_vec.push(run.run_id);
                    let iter = run.state_reader.to_state_iter();
                    state_iters.push(iter);
                    lower_cdc_tree_readers.push(run.lower_cdc_tree_reader);
                    upper_mht_readers.push((run.run_id, run.upper_mht_reader));
                }
                // create a new run_id
                let run_id = self.new_run_id();
                // next disk level's id
                let next_level_id = level_id + 1;
                let next_level_num_of_run = match self.levels.get(next_level_id as usize) {
                    Some(level) => {
                        level.run_vec.len()
                    },
                    None => {
                        0
                    }
                };
                let new_run = LevelRun::construct_run_by_merge(state_iters, lower_cdc_tree_readers, upper_mht_readers, run_id, next_level_id as u32, &self.configs.dir_name, self.configs.fanout, self.configs.max_num_of_states_in_a_run(next_level_id as u32), next_level_num_of_run, self.configs.size_ratio, self.configs.is_pruned);
                match self.levels.get_mut(next_level_id) {
                    // the next level exists, insert the new run to run_vec
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
                // remove the merged files in level_id by using multi-threads; note that we do not need to wait for the ending of the thread.
                Level::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
                level_id += 1;
            } else {
                break;
            }
        }
    }

    pub fn search_latest_state_value(&mut self, addr_key: AddrKey) -> Option<(AddrKey, u32, StateValue)> {
        // compute the boundary compound key
        let upper_key = CompoundKey::new(addr_key, u32::MAX);
        // search the write-group in-mem tree
        let write_index = self.get_write_in_mem_group_index();
        let write_tree = &mut self.in_mem_group[write_index].mem_mht;
        match pattern_oridented_split_tree::search_with_upper_key(write_tree, upper_key) {
            Some((read_key, read_v)) => {
                if read_key.addr == addr_key {
                    // matches the addresses and should be the latest value since latest value should be in the upper levels
                    return Some((read_key.addr, read_key.version, read_v));
                }
            },
            None => {},
        }
        // search the merge-group in-mem tree 
        let merge_index = self.get_merge_in_mem_group_index();
        let merge_tree = &mut self.in_mem_group[merge_index].mem_mht;
        match pattern_oridented_split_tree::search_with_upper_key(merge_tree, upper_key) {
            Some((read_key, read_v)) => {
                if read_key.addr == addr_key {
                    // matches the addresses and should be the latest value since latest value should be in the upper levels
                    return Some((read_key.addr, read_key.version, read_v));
                }
            },
            None => {},
        }
        // search other levels on the disk
        for level in &mut self.levels {
            // search each run in the reverse order (new -> old)
            for run in level.run_vec.iter_mut().rev() {
                let res = run.search_run(&upper_key, &mut self.cache_manager);
                if let Some(inner_res) = res {
                    if inner_res.0 == addr_key {
                        return Some(inner_res);
                    }
                }
            }
        }
        return None;
    }

    pub fn search_latest_state_value_on_disk(&mut self, addr_key: AddrKey) -> Option<(u32, (AddrKey, u32, StateValue))> {
        // compute the boundary compound key
        let upper_key = CompoundKey::new(addr_key, u32::MAX);
        // search other levels on the disk
        let mut cnt = 0u32;
        for level in &mut self.levels {
            // search each run in the reverse order (new -> old)
            for run in level.run_vec.iter_mut().rev() {
                cnt += 1;
                let res = run.search_run(&upper_key, &mut self.cache_manager);
                if let Some(inner_res) = res {
                    if inner_res.0 == addr_key {
                        return Some((cnt, inner_res));
                    }
                }
            }
        }
        return None;
    }

    pub fn search_with_proof(&mut self, addr_key: AddrKey, lb: u32, ub: u32) -> ColePlusProof {
        let mut proof = ColePlusProof::new();
        // generate the two compound keys
        let low_key = CompoundKey::new(addr_key, lb);
        let upper_key = CompoundKey::new(addr_key, ub);
        let mut rest_is_hash = false;
        // search the write group of the in-memory tree
        let write_index = self.get_write_in_mem_group_index();
        let (r, p) = pattern_oridented_split_tree::get_range_proof(&mut self.in_mem_group[write_index].mem_mht, low_key, upper_key);
        if r.is_some() {
            // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
            // there is no need to prove_range the run
            let left_most_result = r.as_ref().unwrap()[0].0;
            let result_version = left_most_result.version;
            if result_version < lb {
                rest_is_hash = true;
            }
        }
        // include the result and proof of write group
        proof.in_mem_level.set_write_group(r, p);
        // search the merge group of the in-memory tree
        let merge_index = self.get_merge_in_mem_group_index();
        let (r, p) = pattern_oridented_split_tree::get_range_proof(&mut self.in_mem_group[merge_index].mem_mht, low_key, upper_key);
        if r.is_some() {
            // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
            // there is no need to prove_range the run
            let left_most_result = r.as_ref().unwrap()[0].0;
            let result_version = left_most_result.version;
            if result_version < lb {
                rest_is_hash = true;
            }
        }
        // include the result and proof of merge group
        proof.in_mem_level.set_merge_group(r, p);
        
        // search the runs in all disk levels
        for level in &mut self.levels {
            let mut level_proof = Vec::new();
            for run in level.run_vec.iter_mut().rev() {
                // decide to add the run's proof or the run's digest
                if rest_is_hash == false {
                    let (r, p) = run.prove_range(addr_key, lb, ub, &self.configs, &mut self.cache_manager);
                    if r.is_some() {
                        // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
                        // there is no need to prove_range the run
                        let left_most_result = &r.as_ref().unwrap()[0];
                        let result_version = left_most_result.ver;
                        if result_version < lb {
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
        // serialize write in_mem mht
        let write_index = self.get_write_in_mem_group_index();
        let write_mht_bytes = bincode::serialize(&self.in_mem_group[write_index].mem_mht).unwrap();
        let write_mht_len = write_mht_bytes.len() as u32;
        bytes.extend(write_mht_len.to_be_bytes());
        bytes.extend(&write_mht_bytes);
        // serialize merge in_mem mht
        let merge_index = self.get_merge_in_mem_group_index();
        let merge_mht_bytes = bincode::serialize(&self.in_mem_group[merge_index].mem_mht).unwrap();
        let merge_mht_len = merge_mht_bytes.len() as u32;
        bytes.extend(merge_mht_len.to_be_bytes());
        bytes.extend(&merge_mht_bytes);
        // persist the bytes to the manifest file
        let path = self.get_tree_meta_path();
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&path).unwrap();
        file.write_all(&mut bytes).unwrap();
    }

    // compute the digest
    pub fn compute_digest(&self) -> H256 {
        let mut hash_vec: Vec<H256> = Vec::new();
        // collect the write and merge group of in_mem_mht
        let write_index = self.get_write_in_mem_group_index();
        hash_vec.push(self.in_mem_group[write_index].mem_mht.get_root_hash());
        let merge_index = self.get_merge_in_mem_group_index();
        hash_vec.push(self.in_mem_group[merge_index].mem_mht.get_root_hash());
        let level_hash_vec: Vec<H256> = self.levels.iter().map(|level| level.compute_digest()).collect();
        hash_vec.extend(level_hash_vec);
        compute_concatenate_hash(&hash_vec)
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
        let write_index = self.get_write_in_mem_group_index();
        let write_mht_bytes = bincode::serialize(&self.in_mem_group[write_index].mem_mht).unwrap();
        let write_mht_len = write_mht_bytes.len();
        let merge_index = self.get_merge_in_mem_group_index();
        let merge_mht_bytes = bincode::serialize(&self.in_mem_group[merge_index].mem_mht).unwrap();
        let merge_mht_len = merge_mht_bytes.len();
        let mht_size = write_mht_len + merge_mht_len;
        MemCost::new(0, filter_size, mht_size)
    }

    pub fn print_structure_info(&mut self) {
        // println!("in mem num: {:?}", self.in_mem_group);
        // println!("num of disk levels: {}", self.levels.len());
        println!("each level info:");
        for level in &mut self.levels {
            println!("level num of runs: {}", level.run_vec.len());
            for run in &mut level.run_vec {
                let v = run.state_reader.read_all_state(run.run_id);
                println!("run id: {}, state len: {}", run.run_id, v.len());
            }
        }
    }

    pub fn get_block_range_checkpoint_log(&self, block_id: u32) -> String {
        format!("{}/ckp_{}.log", &self.configs.dir_name, block_id)
    }
}

pub fn verify_and_collect_result(addr_key: AddrKey, lb: u32, ub: u32, root_hash: H256, proof: &ColePlusProof, fanout: usize) -> (bool, Option<Vec<VerObject>>) {
    let mut level_roots = Vec::<H256>::new();
    // first reconstruct the in_memory_proof
    let low_key = CompoundKey::new(addr_key, lb);
    let upper_key = CompoundKey::new(addr_key, ub);
    // retrieve write group result and proof
    let write_group_result = &proof.in_mem_level.write_group.0;
    let write_group_proof = &proof.in_mem_level.write_group.1;
    let h = pattern_oridented_split_tree::reconstruct_range_proof(low_key, upper_key, write_group_result, write_group_proof);
    level_roots.push(h);
    let mut merge_result: Vec<VerObject> = vec![];
    let mut rest_is_hash = false;
    if write_group_result.is_some() {
        let left_most_result = write_group_result.as_ref().unwrap()[0].0;
        let result_version = left_most_result.version;
        if result_version < lb {
            rest_is_hash = true;
        }
        let mut r: Vec<VerObject> = Vec::new();
        for (k, v) in write_group_result.as_ref().unwrap() {
            if k.addr == addr_key {
                let ver = k.version;
                let value = *v;
                r.push(VerObject::new(ver, value));
            }
        }
        merge_result.extend(r);
    }
    // then reconstruct merge group of the in-mem tree
    // retrieve merge group result and proof
    let merge_group_result = &proof.in_mem_level.merge_group.0;
    let merge_group_proof = &proof.in_mem_level.merge_group.1;
    let h = pattern_oridented_split_tree::reconstruct_range_proof(low_key, upper_key, merge_group_result, merge_group_proof);
    level_roots.push(h);
    if merge_group_result.is_some() {
        let left_most_result = merge_group_result.as_ref().unwrap()[0].0;
        let result_version = left_most_result.version;
        if result_version < lb {
            rest_is_hash = true;
        }
        let mut r: Vec<VerObject> = Vec::new();
        for (k, v) in merge_group_result.as_ref().unwrap() {
            if k.addr == addr_key {
                let ver = k.version;
                let value = *v;
                r.push(VerObject::new(ver, value));
            }
        }
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
                    let h = reconstruct_run_proof(&addr_key, lb, ub, r, proof, fanout);
                    level_h_vec.push(h);
                }
            }
            if r.is_some() {
                let left_most_result = &r.as_ref().unwrap()[0];
                let result_version = left_most_result.ver;
                if result_version < lb {
                    rest_is_hash = true;
                }
                merge_result.extend_from_slice(r.as_ref().unwrap());
            }
        }
        let level_h = compute_concatenate_hash(&level_h_vec);
        level_roots.push(level_h);
    }
    let reconstruct_root = compute_concatenate_hash(&level_roots);
    if reconstruct_root != root_hash {
        println!("reconstruct fail");
        println!("proof: {:?}", proof);
        return (false, None);
    }
    merge_result.sort_by(|a, b| a.ver.partial_cmp(&b.ver).unwrap());
    merge_result = merge_result.into_iter().filter(|r| r.ver >= lb && r.ver <= ub).collect();
    return (true, Some(merge_result));
}

impl<'a> Drop for ColePlus<'a> {
    fn drop(&mut self) {
        self.update_manifest();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RunProofOrHash {
    Proof(RunProof),
    Hash(H256),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColePlusInMemProof {
    pub write_group: (Option<Vec<(CompoundKey, StateValue)>>, POSTreeRangeProof<CompoundKey, StateValue>),
    pub merge_group: (Option<Vec<(CompoundKey, StateValue)>>, POSTreeRangeProof<CompoundKey, StateValue>),
}

impl ColePlusInMemProof {
    pub fn new() -> Self {
        Self {
            write_group: (None, POSTreeRangeProof::default()),
            merge_group: (None, POSTreeRangeProof::default()),
        }
    }

    pub fn set_write_group(&mut self, r: Option<Vec<(CompoundKey, StateValue)>>, p: POSTreeRangeProof<CompoundKey, StateValue>) {
        self.write_group = (r, p);
    }

    pub fn set_merge_group(&mut self, r: Option<Vec<(CompoundKey, StateValue)>>, p: POSTreeRangeProof<CompoundKey, StateValue>) {
        self.merge_group = (r, p);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColePlusProof {
    pub in_mem_level: ColePlusInMemProof,
    pub disk_level: Vec<Vec<(Option<Vec<VerObject>>, RunProofOrHash)>>,
}

impl ColePlusProof {
    pub fn new() -> Self {
        let in_mem_level = ColePlusInMemProof::new();
        let disk_level = Vec::new();
        Self {
            in_mem_level,
            disk_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use utils::{compute_cole_size_breakdown, types::{Address, StateKey}};
    #[test]
    fn test_insert_cole_plus() {
        let num_of_contract = 1;
        let num_of_addr = 10;
        let num_of_version = 10000;
        let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let fanout = 8;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 500;
        let size_ratio = 2;
        let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, true, false, false);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
                addr_key_vec.push(addr_key);
            }
        }
        addr_key_vec.sort();
        for k in 1..=num_of_version {
            for (i, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k * 2), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
            }
        }

        let mut cole_plus = ColePlus::new(&configs);
        let start = std::time::Instant::now();
        for state in &state_vec {
            cole_plus.insert((state.0.addr, state.0.version, state.1));
        }
        let elapse = start.elapsed().as_nanos();
        println!("average insert: {:?}", elapse / n as u128);
    }

    #[test]
    fn test_block_range_cole_plus() {
        let num_of_contract = 1;
        let num_of_addr = 100;
        let num_of_version = 100;
        let mut rng = StdRng::seed_from_u64(1);
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 1000;
        let size_ratio = 2;
        let is_rolling = true;
        let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, false, is_rolling, false);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
                addr_key_vec.push(addr_key);
            }
        }
        for k in 1..=num_of_version {
            for (i, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
            }
        }

        let mut cole_plus = ColePlus::new(&configs);
        let mut cnt = 1;
        for state in &state_vec {
            cole_plus.insert((state.0.addr, state.0.version, state.1));
            if cnt % 100 == 0 {
                println!("cnt: {}",cnt);
                println!("write group: {:?}", cole_plus.in_mem_group[cole_plus.get_write_in_mem_group_index()].max_block_id);
                println!("merge group: {:?}", cole_plus.in_mem_group[cole_plus.get_merge_in_mem_group_index()].max_block_id);
            }
            
            cnt += 1;
        }
    }

    #[test]
    fn test_query_cole_plus() {
        let num_of_contract = 10;
        let num_of_addr = 100;
        let num_of_version = 10;
        let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 1000;
        let size_ratio = 2;
        let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, false, false, false);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
                addr_key_vec.push(addr_key);
            }
        }
        for k in 1..=num_of_version {
            for (i, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k * 2), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
            }
        }

        let mut cole_plus = ColePlus::new(&configs);
        let start = std::time::Instant::now();
        for state in &state_vec {
            cole_plus.insert((state.0.addr, state.0.version, state.1));
        }
        let elapse = start.elapsed().as_nanos();
        println!("average insert: {:?}", elapse / n as u128);

        let mut break_flag = false;
        let digest = cole_plus.compute_digest();
        let mut search_prove = 0;
        for (i, addr) in addr_key_vec.iter().enumerate() {
            for k in 1..=num_of_version {
                let lb = k * 2 as u32;
                let ub = k * 2 as u32;
                let start = std::time::Instant::now();
                let proof = cole_plus.search_with_proof(*addr, lb, ub);
                let (b, r) = verify_and_collect_result(*addr, lb, ub, digest, &proof, fanout);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                if b == false {
                    println!("addr: {:?}, lb: {}, ub: {}", addr, lb, ub);
                    println!("verification fails");

                    break_flag = true;
                    break;
                }
                let true_r = (k * 2 as u32, StateValue(H256::from_low_u64_be((k as u64 + i as u64) * 2)));
                let ver_obj = &r.unwrap()[0];

                let retrieved_r = (ver_obj.ver, ver_obj.value);
                if retrieved_r != true_r {
                    println!("retrieved_r: {:?}, true_r: {:?}", retrieved_r, true_r);
                }
            }
            if break_flag {
                break;
            }
        }
        
        println!("avg search prove: {}", search_prove / (num_of_contract * num_of_addr * num_of_version) as u128);

        let start = std::time::Instant::now();
        for (i, addr) in addr_key_vec.iter().enumerate() {
            let r = cole_plus.search_latest_state_value(*addr).unwrap();
            let true_value = StateValue(H256::from_low_u64_be((num_of_version as u64 + i as u64) * 2));
            if r.2 != true_value {
                println!("false addr: {:?}", addr);
                println!("true value: {:?}, return r: {:?}", true_value, r);
            }
        }
        let elapse = start.elapsed().as_nanos();
        println!("average query latest: {:?}", elapse / addr_key_vec.len() as u128);
        // cole_plus.print_structure_info();
        drop(cole_plus);
        let storage_size = compute_cole_size_breakdown(dir_name);
        println!("storage size: {:?}", storage_size);
        println!("start query latest");
    }
}
