use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use cole_plus::{level::Level, run::LevelRun};
use utils::{config::Configs, disk_usage_check_directory_storage, types::{AddrKey, Address, CompoundKey, StateKey, StateValue}};
use super::tx_executor::Backend;
use std::{cell::UnsafeCell, collections::BTreeMap, fs, path::Path};
use anyhow::Result;
use cole_plus::ColePlus;
use utils::MemCost;
use pattern_oridented_split_tree::remove as postremove;
use caches::{RawLRU, Cache};

pub struct ColePlusBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: ColePlus<'a>,
    pub path: &'a str,
    pub stashed_blocks: BTreeMap<u32, Vec<AddrKey>>,
    pub latest_block_id: u32,
    pub kept_num_blocks: u32,
}

impl<'a> ColePlusBackend<'a> {
    pub fn new(configs: &'a Configs, path: &'a str, kept_num_blocks: u32) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: ColePlus::new(configs),
            path,
            stashed_blocks: BTreeMap::new(),
            latest_block_id: 0,
            kept_num_blocks,
        }
    }

    pub fn get_mut_total_tree(&self) -> &'a mut ColePlus<'a> {
        unsafe {
            let const_ptr = &self.states as *const ColePlus;
            let mut_ptr = UnsafeCell::new(const_ptr as *mut ColePlus);
            &mut **mut_ptr.get()
        }
    }

    pub fn rewind_in_mememory_states(&mut self, num_of_rewound_blocks: u32) {
        println!("num rewound: {}", num_of_rewound_blocks);
        println!("latest block: {}", self.latest_block_id);
        println!("write group max: {}", self.states.in_mem_group[self.states.get_write_in_mem_group_index()].max_block_id);
        println!("merge group max: {}", self.states.in_mem_group[self.states.get_merge_in_mem_group_index()].max_block_id);
        let rewound_min_block_id = self.latest_block_id - num_of_rewound_blocks + 1;
        let merge_group_max_block_id = self.states.in_mem_group[self.states.get_merge_in_mem_group_index()].max_block_id;
        if rewound_min_block_id > merge_group_max_block_id {
            // all the rewound blocks exist in the write group
            for block_id in rewound_min_block_id..=self.latest_block_id {
                // println!("remove block id: {}", block_id);
                for addr_key in self.stashed_blocks.get(&block_id).unwrap() {
                    postremove(&mut self.states.in_mem_group[self.states.get_write_in_mem_group_index()].mem_mht, CompoundKey::new(*addr_key, block_id));
                }
            }
        } else {
            // remove the write group tree
            println!("directly remove the write group.");
            self.states.in_mem_group[self.states.get_write_in_mem_group_index()].mem_mht.clear();
            // remove partial of the merge group
            // println!("stashed block num: {}", self.stashed_blocks.len());
            for block_id in rewound_min_block_id..=merge_group_max_block_id {
                // println!("remove block id: {}", block_id);
                for addr_key in self.stashed_blocks.get(&block_id).unwrap() {
                    postremove(&mut self.states.in_mem_group[self.states.get_merge_in_mem_group_index()].mem_mht, CompoundKey::new(*addr_key, block_id));
                }
            }
        }
    }

    pub fn rewind_disk_states(&mut self, num_of_rewound_blocks: u32) {
        println!("num rewound: {}", num_of_rewound_blocks);
        let rewound_destination = self.latest_block_id - num_of_rewound_blocks;
        let rewound_checkpoint_file = self.states.get_block_range_checkpoint_log(rewound_destination);
        if !Path::new(&rewound_checkpoint_file).exists() {
            println!("checkpoint not found.");
        } else {
            let rewound_checkpoint_json = fs::read_to_string(rewound_checkpoint_file).unwrap();
            let mut rewound_checkpoint_list: Vec<Vec<(u32, u32)>> = serde_json::from_str(&rewound_checkpoint_json).unwrap();
            println!("rewound checkpoint list: {:?}", rewound_checkpoint_list);
            let mut cur_checkpoint_list = self.cur_checkpoint_list();
            println!("cur checkpoint list: {:?}", cur_checkpoint_list);


            let max_level_id = rewound_checkpoint_list.len() - 1;
            let mut new_levels = Vec::<Level>::new(); // create new levels for the rewound inex

            let mut iter_cur_level = max_level_id as i64;

            if cur_checkpoint_list.len() == rewound_checkpoint_list.len() {
                // some runs can be reused!
                while iter_cur_level >= 0 {
                    let mut level = Level::new(iter_cur_level as u32);
                    let mut flag = false;
                    while rewound_checkpoint_list[iter_cur_level as usize].len() > 0 && rewound_checkpoint_list[iter_cur_level as usize][0].0 == cur_checkpoint_list[iter_cur_level as usize][0].0
                        && rewound_checkpoint_list[iter_cur_level as usize][0].1 == cur_checkpoint_list[iter_cur_level as usize][0].1 
                    {
                        flag = true; // has matched run
                        println!("matched run: level {}, run index {}", iter_cur_level, 0);
                        let reused_run = self.states.levels[iter_cur_level as usize].run_vec.remove(0);
                        rewound_checkpoint_list[iter_cur_level as usize].remove(0);
                        cur_checkpoint_list[iter_cur_level as usize].remove(0);
                        level.run_vec.push(reused_run);
                    }

                    if flag {
                        new_levels.insert(0, level);
                        iter_cur_level -= 1;
                    } else {
                        break;
                    }
                }
            }
            
            // the rest of the runs cannot be reused!
            let mut state_cache_manager = RunStateCacheManager::new();
            while iter_cur_level >= 0 {
                let mut level = Level::new(iter_cur_level as u32);
                for rewound_run_block_range in &rewound_checkpoint_list[iter_cur_level as usize] {
                    let rewound_run_block_range_low = rewound_run_block_range.0;
                    let rewound_run_block_range_high = rewound_run_block_range.1;
                    println!("rebuild {}, {}", rewound_run_block_range_low, rewound_run_block_range_high);
                    let (found_level_id_in_cur_index, found_run_index_in_cur_index) = self.find_matching_run(rewound_run_block_range_low, rewound_run_block_range_high).unwrap();
                    let collect_states = self.collect_states_from_level_id_and_run_index(found_level_id_in_cur_index, found_run_index_in_cur_index, &mut state_cache_manager);
                    let mut filtered_states: Vec<(CompoundKey, StateValue)> = collect_states.into_iter().filter(|(c_key, _)| c_key.version >= rewound_run_block_range_low && c_key.version <= rewound_run_block_range_high).collect();
                    // sort the filtered_states
                    filtered_states.sort_by(|a, b| a.0.cmp(&b.0));
                    // construct new run
                    let run_id = self.states.new_run_id();
                    let level_num_of_run = level.run_vec.len();
                    let run = LevelRun::construct_run_by_in_memory_collection(filtered_states, run_id, iter_cur_level as u32, &self.states.configs.dir_name, self.states.configs.fanout, self.states.configs.max_num_of_states_in_a_run(iter_cur_level as u32), level_num_of_run, self.states.configs.size_ratio, self.states.configs.is_pruned);
                    level.run_vec.push(run);
                }
                new_levels.insert(0, level);
                iter_cur_level -= 1;
            }

            self.clean_up_old_levels();
            self.states.levels = new_levels;
        }
    }

    fn clean_up_old_levels(&mut self) {
        for i in 0..self.states.levels.len() {
            let removed_level = self.states.levels.remove(0);
            let run_id_vec: Vec<u32> = removed_level.run_vec.into_iter().map(|run| run.run_id).collect();
            Level::remove_run_files(run_id_vec, i as u32, &self.states.configs.dir_name);
        }
    }

    fn collect_states_from_level_id_and_run_index(&mut self, level_id: usize, run_index: usize, state_cache_manager: &mut RunStateCacheManager) -> Vec<(CompoundKey, StateValue)> {
        let r = state_cache_manager.read_state_cache(level_id, run_index);
        if r.is_some() {
            let run_states = r.unwrap();
            return run_states;
        } else {
            let run_states = self.states.levels[level_id].run_vec[run_index].load_run_states();
            state_cache_manager.set_state_cache(level_id, run_index, run_states.clone());
            return run_states;
        }
    }

    // return level_id, run's index
    fn find_matching_run(&self, search_block_range_low: u32, search_block_range_high: u32) -> Option<(usize, usize)> {
        let cur_checkpoint_list = self.cur_checkpoint_list();
        let mut level_id = cur_checkpoint_list.len() as i64 - 1;
        while level_id >= 0 {
            for (run_index, cur_run_block_range) in cur_checkpoint_list[level_id as usize].iter().enumerate() {
                let cur_block_range_low = cur_run_block_range.0;
                let cur_block_range_high = cur_run_block_range.1;
                if check_overlap(search_block_range_low, search_block_range_high, cur_block_range_low, cur_block_range_high) == true {
                    return Some((level_id as usize, run_index));
                }
            }
            level_id -= 1;
        }
        return None;
    }

    pub fn cur_checkpoint_list(&self) -> Vec<Vec<(u32, u32, u32)>> {
        let mut levels_logs = vec![];
        for level in &self.states.levels {
            let mut runs = vec![];
            for run in level.run_vec.iter() {
                // println!("[{} {}]", run.block_range_low, run.block_range_high);
                runs.push((run.block_range_low, run.block_range_high, run.run_id));
            }
            levels_logs.push(runs);
        }
        return levels_logs;
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RunStateCacheIndex {
    level_id: usize,
    run_index: usize,
}

impl RunStateCacheIndex {
    pub fn new(level_id: usize, run_index: usize) -> Self {
        Self {
            level_id,
            run_index,
        }
    }
}

pub struct RunStateCacheManager {
    pub run_state_cache: RawLRU<RunStateCacheIndex, Vec<(CompoundKey, StateValue)>>,
}

impl RunStateCacheManager {
    pub fn new() -> Self {
        let run_state_cache = RawLRU::<RunStateCacheIndex, Vec<(CompoundKey, StateValue)>>::new(1).unwrap();
        Self {
            run_state_cache,
        }
    }

    pub fn read_state_cache(&mut self, level_id: usize, run_index: usize) -> Option<Vec<(CompoundKey, StateValue)>> {
        let index = RunStateCacheIndex::new(level_id, run_index);
        self.run_state_cache.get(&index).cloned()
    }

    pub fn set_state_cache(&mut self, level_id: usize, run_index: usize, states: Vec<(CompoundKey, StateValue)>) {
        let index = RunStateCacheIndex::new(level_id, run_index);
        self.run_state_cache.put(index, states); 
    }
}

fn check_overlap(search_block_range_low: u32, search_block_range_high: u32, cur_block_range_low: u32, cur_block_range_high: u32) -> bool {
    // Range 1: [search_block_range_low, search_block_range_high]
    // Range 2: [cur_block_range_low, cur_block_range_high]

    // They overlap if the first range doesn't end before the second one starts,
    // AND the second range doesn't end before the first one starts.
    search_block_range_low <= cur_block_range_high && cur_block_range_low <= search_block_range_high
}

impl<'a> Backend for ColePlusBackend<'a> {
    fn get_code(&self, acc_address: Address) -> Result<Code> {
        if self.code_map.contains_key(&acc_address) {
            Ok(self.code_map.get(&acc_address).unwrap().clone())
        } else {
            Ok(Code::default())
        }
    }

    fn get_nonce(&self, acc_address: Address) -> Result<Nonce> {
        if self.nonce_map.contains_key(&acc_address) {
            return Ok(self.nonce_map.get(&acc_address).unwrap().clone());
        } else {
            return Ok(Nonce::default());
        }
    }

    fn get_value(&self, acc_address: Address, key: StateKey) -> Result<StateValue> {
        let addr_key = AddrKey::new(acc_address, key);
        let v = self.get_mut_total_tree().search_latest_state_value(addr_key);
        match v {
            Some((_, _, value)) => {
                Ok(value)
            },
            None => {
                return Ok(StateValue::default());
            }
        }
    }
}

impl<'a> BackendWriteTrait for ColePlusBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, block_id: u32) {
        self.states.insert((addr_key, block_id, v));
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, block_id: u32) {
        for (addr_key, value) in &states {
            self.states.insert((*addr_key, block_id, *value));
        }
        // for reorg
        if self.states.configs.test_in_mem_roll ==  true {
            self.latest_block_id = block_id;
            // push addr_key to the stashed block
            let addr_keys: Vec<AddrKey> = states.into_keys().collect();
            self.stashed_blocks.insert(block_id, addr_keys);
            // clean the out-of-date block
            if self.stashed_blocks.len() > self.kept_num_blocks as usize {
                let removed_block_id = self.latest_block_id - self.kept_num_blocks;
                self.stashed_blocks.remove(&removed_block_id).unwrap();
            }
        }
    }

    fn set_acc_nonce(&mut self, contract_addr: &Address, contract_nonce: Nonce) {
        self.nonce_map.insert(*contract_addr, contract_nonce);
    }

    fn get_acc_nonce(&self, contract_addr: &Address) -> Nonce {
        match self.nonce_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Nonce::default()
            }
        }
    }

    fn set_acc_code(&mut self, contract_addr: &Address, contract_code: Code) {
        self.code_map.insert(*contract_addr, contract_code);
    }

    fn get_acc_code(&self, contract_addr: &Address) -> Code {
        match self.code_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Code::default()
            }
        }
    }

    fn memory_cost(&self,) -> MemCost {
        self.states.memory_cost()
    }

    fn index_stucture_output(&self,) -> String {
        format!("")
    }
    fn flush(&mut self) {

    }
    fn commit(&mut self) {
        
    }

    fn print_in_mem_tree(&self) {
        println!("{:?}", self.states.in_mem_group[0]);
        println!("{:?}", self.states.in_mem_group[0]);
    }

    fn index_size(&self) -> usize {
        let disk_size = disk_usage_check_directory_storage(&self.path);
        let mem_size = self.memory_cost().size();
        return disk_size + mem_size;
    }
}

#[cfg(test)]
mod tests {
    use crate::send_tx::{create_deploy_tx, create_call_tx, ContractArg};
    use super::super::tx_executor::{exec_tx, test_batch_exec_tx};
    use super::super::common::tx_req::TxRequest;
    use super::*;
    use rand::prelude::*;
    use primitive_types::H160;
    use cole_plus::verify_and_collect_result;
    use utils::{compute_cole_size_breakdown, disk_usage_check_directory_storage};

    #[test]
    fn test_cole_plus_prune_backend() {
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 500;
        let size_ratio = 5;
        
        let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, false, false, false);
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = ColePlusBackend::new(&configs, dir_name, 0);

        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);

        let n = 1000;
        let small_bank_n = n / 100;
        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = i % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i as i32), &mut rng, small_bank_n as usize);
            requests.push(call_tx_req);
        }
        let block_size = 100;
        let blocks: Vec<Vec<TxRequest>> = requests.chunks(block_size).into_iter().map(|v| v.to_owned()).collect();
        let mut i = 1;
        let mut states = BTreeMap::<AddrKey, StateValue>::new();
        let start = std::time::Instant::now();
        for block in blocks {
            println!("block {}", i);
            let s = test_batch_exec_tx(block, caller_address, i, &mut backend);
            states.extend(s);
            i += 1;
        }
        let elapse = start.elapsed().as_nanos();
        println!("time: {}", elapse / n as u128);
        
        // println!("sleep");
        // std::thread::sleep(std::time::Duration::from_secs(30));
        let mut search_latest = 0;
        for (k, v) in states {
            let start = std::time::Instant::now();
            let (_, _, read_v) = backend.states.search_latest_state_value(k).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(read_v, v);
        }
        println!("search latest: {}", search_latest / n as u128);
        let before_flush_storage = disk_usage_check_directory_storage(dir_name) + backend.memory_cost().size();
        println!("{:?}", backend.memory_cost());
        println!("before flush storage: {}", before_flush_storage);
        drop(backend);
        let storage_size = compute_cole_size_breakdown(dir_name);
        println!("storage size: {:?}", storage_size);
    }

    #[test]
    fn test_cole_plus_backend() {
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 100;
        let size_ratio = 5;
        let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, false, false, false);
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = ColePlusBackend::new(&configs, dir_name, 0);

        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);

        let n = 5000;
        let small_bank_n = n / 100;
        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = i % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i as i32), &mut rng, small_bank_n as usize);
            requests.push(call_tx_req);
        }
        let block_size = 100;
        let blocks: Vec<Vec<TxRequest>> = requests.chunks(block_size).into_iter().map(|v| v.to_owned()).collect();
        let mut i = 1;
        let mut states = BTreeMap::<AddrKey, StateValue>::new();
        let start = std::time::Instant::now();
        for block in blocks {
            let s = test_batch_exec_tx(block, caller_address, i, &mut backend);
            states.extend(s);
            i += 1;
        }
        let elapse = start.elapsed().as_nanos();
        println!("time: {}", elapse / n as u128);
        
        // println!("sleep");
        // std::thread::sleep(std::time::Duration::from_secs(30));
        let digest = backend.states.compute_digest();
        let latest_version = n as u32 / block_size as u32;
        let mut search_latest = 0;
        let mut search_prove = 0;
        for (k, v) in states {
            let start = std::time::Instant::now();
            let (_, _, read_v) = backend.states.search_latest_state_value(k).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(read_v, v);
            for version in 1..= latest_version {
                let start = std::time::Instant::now();
                let p = backend.states.search_with_proof(k, version, version);
                let (b, _) = verify_and_collect_result(k, version, version, digest, &p, fanout);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                if b == false {
                    println!("false");
                }
            }
        }
        println!("search latest: {}", search_latest / n as u128);
        println!("search prove: {}", search_prove / (n * latest_version) as u128);

        drop(backend);
        let storage_size = compute_cole_size_breakdown(dir_name);
        println!("storage size: {:?}", storage_size);
    }
}