pub mod run;
use run::LevelRun;
pub mod async_level;
use async_level::AsyncLevel;
use cole_plus::in_memory_postree::InMemoryPOSTree;
use pattern_oridented_split_tree::traits::POSTreeNodeIO;
use std::thread::JoinHandle;
use utils::{cacher::CacheManager, config::Configs, pager::{cdc_mht::CDCTreeReader, state_pager::{StateIterator, StatePageReader}, upper_mht::UpperMHTReader}, types::{compute_concatenate_hash, AddrKey, CompoundKey, StateValue}, OpenOptions, Read, Write};
use primitive_types::H256;
use std::fmt::{Debug, Formatter, Error};

pub const EVAL_STORAGE_AFTER_DROP: bool = true;
use utils::DEFAULT_MAX_NODE_CAPACITY;
pub struct InMemGroup {
    pub mem_mht: InMemoryPOSTree, // in-mem POS-Tree
    pub thread_handle: Option<JoinHandle<LevelRun>>, // object related to the asynchronous merge thread,
}

impl Debug for InMemGroup {
    fn fmt(&self, _: &mut Formatter<'_>) -> Result<(), Error> {
        println!("mem mht: {:?}", self.mem_mht);
        println!("thread is some: {}", self.thread_handle.is_some());
        Ok(())
    }
}

impl InMemGroup {
    pub fn new(exp_fanout: usize, max_fanout: usize) -> Self {
        Self {
            mem_mht: InMemoryPOSTree::new(exp_fanout, max_fanout),
            thread_handle: None,
        }
    }
    pub fn clear(&mut self) {
        self.mem_mht.clear();
        self.thread_handle = None;
    }
}

pub struct ColePlusBinary<'a> {
    pub configs: &'a Configs,
    pub in_mem_group: [InMemGroup; 2],
    pub in_mem_write_group_flag: bool,
    pub levels: Vec<AsyncLevel>,
    pub run_id_cnt: u32, // this helps generate a new run_id
    pub cache_manager: CacheManager,
}

impl<'a> ColePlusBinary<'a> {
    // create a new index using configs,
    pub fn new(configs: &'a Configs) -> Self {
        Self {
            configs,
            in_mem_group: [InMemGroup::new(configs.fanout, DEFAULT_MAX_NODE_CAPACITY), InMemGroup::new(configs.fanout, DEFAULT_MAX_NODE_CAPACITY)],
            in_mem_write_group_flag: true,
            levels: Vec::new(), // empty levels' vector
            run_id_cnt: 0, // initiate the counter to be 0
            cache_manager: CacheManager::new(),
        }
    }

    fn get_tree_meta_path(&self) -> String {
        format!("{}/mht", &self.configs.dir_name)
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

    // load a new index using configs
    pub fn load(configs: &'a Configs) -> Self {
        let mut ret = Self::new(configs);
        let level_len = ret.get_meta();
        // load levels
        for i in 0..level_len {
            let level = AsyncLevel::load(i as u32, configs);
            ret.levels.push(level);
        }
        return ret;
    }

    fn new_run_id(&mut self) -> u32 {
        // increment the run_id and return it
        self.run_id_cnt += 1;
        return self.run_id_cnt;
    }

    fn switch_in_mem_group(&mut self) {
        // reverse the flag of write group
        if self.in_mem_write_group_flag == true {
            self.in_mem_write_group_flag = false;
        } else {
            self.in_mem_write_group_flag = true;
        }
    }

    // add the new run to the level's write group with level_id
    fn add_run_to_level_write_group(&mut self, level_id: u32, new_run: LevelRun) {
        match self.levels.get_mut(level_id as usize) {
            Some(level_ref) => {
                let write_index = level_ref.get_write_group_index();
                level_ref.run_groups[write_index].run_vec.push(new_run); 
                // level_ref.run_groups[write_index].run_vec.insert(0, new_run); // always insert the new run to the front, so that the latest states are at the front of the level
            },
            None => {
                let mut level = AsyncLevel::new(level_id); // the level with level_id does not exist, so create a new one
                let write_index = level.get_write_group_index();
                level.run_groups[write_index].run_vec.push(new_run);
                // level.run_groups[write_index].run_vec.insert(0, new_run);
                self.levels.push(level); // push the new level to the level vector
            }
        }
    }

    fn prepare_thread_input(&mut self, level_id: u32) -> (u32, String, usize, usize, usize, usize, bool) {
        let new_run_id = self.new_run_id();
        let dir_name = self.configs.dir_name.clone();
        let fanout = self.configs.fanout;
        let size_ratio = self.configs.size_ratio;
        let max_num_of_states = self.configs.max_num_of_states_in_a_run(level_id);
        let level_num_of_run = match self.levels.get(level_id as usize) {
            Some(level) => {
                let write_index = level.get_write_group_index();
                level.run_groups[write_index].run_vec.len()
            },
            None => 0
        };
        let is_pruned = self.configs.is_pruned;
        (new_run_id, dir_name, fanout, max_num_of_states, level_num_of_run, size_ratio, is_pruned)
    }

    pub fn insert(&mut self, state: (AddrKey, u32, StateValue)) {
        let (addr_key, ver, value) = state;
        // compute the in-memory threshold
        let in_mem_thres = (self.configs.base_state_num as f64 * 0.5) as usize;
        // get the write in_mem group index
        let write_index = self.get_write_in_mem_group_index();
        // insert the state to the tree of write group
        let tree_ref = &mut self.in_mem_group[write_index].mem_mht;
        pattern_oridented_split_tree::insert(tree_ref, CompoundKey::new(addr_key, ver), value);
        // check wheither the write group tree is full
        if tree_ref.key_num as usize == in_mem_thres {
            // get the merge group index
            let merge_index = self.get_merge_in_mem_group_index();
            let level_id = 0; // the first on-disk level's id is 0
            // check if the merge group has thread
            if let Some(handle) = self.in_mem_group[merge_index].thread_handle.take() {
                // get the merged new run
                let new_run = handle.join().unwrap();
                // add the new run to the first disk level's write group
                self.add_run_to_level_write_group(level_id, new_run);
                // clear the mb-tree of the merge group and set thread_handle to be None
                self.in_mem_group[merge_index].clear();
            }
            // switch the write group and merge group
            self.switch_in_mem_group();
            // get the updated merge index
            let merge_index = self.get_merge_in_mem_group_index();
            // merge group is full, the data should be merged to the run in the disk-level
            let state_vec = self.in_mem_group[merge_index].mem_mht.load_all_key_values();
            // prepare for the thread input
            let (new_run_id, dir_name, fanout, max_num_of_states, level_num_of_run, size_ratio, is_pruned) = self.prepare_thread_input(level_id);
            // create a merge thread
            let handle = std::thread::spawn(move|| {
                let run = LevelRun::construct_run_by_in_memory_collection(state_vec, new_run_id, level_id, &dir_name, fanout, max_num_of_states, level_num_of_run, size_ratio, is_pruned);
                return run;
            });
            // assign the thread_handle to the merge group
            self.in_mem_group[merge_index].thread_handle = Some(handle);
            // check and merge the disk levels
            self.check_and_merge();
        }
    }

    fn check_and_merge(&mut self) {
        let mut level_id = 0; // start from 0 disk level
        // iteratively check each level's write group is full or not
        while level_id < self.levels.len() {
            if self.levels[level_id].level_write_group_reach_capacity(&self.configs) {
                // get the merge group's index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // get the next level id
                let next_level_id = level_id + 1;
                // check whether the merge group has thread_handle
                if let Some(handle) = self.levels[level_id].run_groups[merge_index].thread_handle.take() {
                    // get the merged new run
                    let new_run = handle.join().unwrap();
                    // add the new run to the next level
                    self.add_run_to_level_write_group(next_level_id as u32, new_run);
                    let merge_group_ref = &mut self.levels[level_id as usize].run_groups[merge_index];
                    // set the thread_handle to be None
                    merge_group_ref.thread_handle = None;
                    // remove all the runs in the merge group in levels[level_id]
                    let run_id_vec: Vec<u32> = merge_group_ref.run_vec.drain(..).map(|run| run.run_id).collect();
                    // remove the merged files in level_id by using multi-threads; note that we do not need to wait for the ending of the thread.
                    AsyncLevel::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
                }
                // switch the write group and merge group
                self.levels[level_id].switch_group();
                // get the updated merge group index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // prepare for run_ids of the merged runs, which will be used during the background merge thread
                let merge_group_run_id_vec: Vec<u32> = self.levels[level_id].run_groups[merge_index].run_vec.iter().map(|run| run.run_id).collect();
                // prepare for the input parameters of the merging thread
                let (new_run_id, dir_name, fanout, max_num_of_states, level_num_of_run, size_ratio, is_pruned) = self.prepare_thread_input(next_level_id as u32);
                let handle = std::thread::spawn(move|| {
                    let mut state_iters = Vec::<StateIterator>::new();
                    let mut lower_cdc_tree_readers = Vec::<CDCTreeReader>::new();
                    let mut upper_mht_readers: Vec::<(u32, UpperMHTReader)> = Vec::<(u32, UpperMHTReader)>::new();
                    for merge_run_id in merge_group_run_id_vec {
                        let state_file_name = LevelRun::file_name(merge_run_id, level_id as u32, &dir_name, "s");
                        let upper_offset_file_name = LevelRun::file_name(merge_run_id, level_id as u32,  &dir_name, "uo");
                        let upper_hash_file_name = LevelRun::file_name(merge_run_id, level_id as u32,  &dir_name, "uh");
                        let lower_cdc_file_name = LevelRun::file_name(merge_run_id, level_id as u32,  &dir_name, "lh");
                        let state_iter = StatePageReader::load(&state_file_name).to_state_iter();
                        state_iters.push(state_iter);
                        let lower_cdc_tree_reader = CDCTreeReader::new(&lower_cdc_file_name);
                        lower_cdc_tree_readers.push(lower_cdc_tree_reader);
                        let upper_mht_reader = UpperMHTReader::new(&upper_offset_file_name, &upper_hash_file_name);
                        upper_mht_readers.push((merge_run_id, upper_mht_reader));
                    }
                    let new_run = LevelRun::construct_run_by_merge(state_iters, lower_cdc_tree_readers, upper_mht_readers, new_run_id, next_level_id as u32, &dir_name, fanout, max_num_of_states, level_num_of_run, size_ratio, is_pruned);
                    return new_run;
                });
                // assign the merge thread handle to level_id's merge group
                self.levels[level_id].run_groups[merge_index].thread_handle = Some(handle);
                level_id += 1;
            } else {
                break;
            }
        }
    }

    pub fn search_latest_state_value(&mut self, addr_key: AddrKey) -> Option<(AddrKey, u32, StateValue)> {
        // compute the boundary compound key
        let upper_key = CompoundKey {
            addr: addr_key,
            version: u32::MAX,
        };
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
            // first search the write group
            let write_index = level.get_write_group_index();
            for run in level.run_groups[write_index].run_vec.iter_mut().rev() {
                let res = run.search_run(&upper_key, &mut self.cache_manager);
                if let Some(inner_res) = res {
                    if inner_res.0 == addr_key {
                        return Some(inner_res);
                    }
                }
            }
            // then search the merge group
            let merge_index = level.get_merge_group_index();
            for run in level.run_groups[merge_index].run_vec.iter_mut().rev() {
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

    // compute the digest of COLE*
    pub fn compute_digest(&self) -> H256 {
        let mut hash_vec = vec![];
        // collect the write and merge group of in_mem_mht
        let write_index = self.get_write_in_mem_group_index();
        hash_vec.push(self.in_mem_group[write_index].mem_mht.get_root_hash());
        let merge_index = self.get_merge_in_mem_group_index();
        hash_vec.push(self.in_mem_group[merge_index].mem_mht.get_root_hash());
        let disk_hash_vec: Vec<H256> = self.levels.iter().map(|level| level.compute_digest()).collect();
        hash_vec.extend(&disk_hash_vec);
        compute_concatenate_hash(&hash_vec)
    }
}

impl<'a> Drop for ColePlusBinary<'a> {
    fn drop(&mut self) {
        if EVAL_STORAGE_AFTER_DROP == false {
            // first handle the in mem level's merge thread
            let merge_index = self.get_merge_in_mem_group_index();
            let mut level_id = 0;
            if let Some(handle) = self.in_mem_group[merge_index].thread_handle.take() {
                // get the merged new run
                let new_run = handle.join().unwrap();
                // add the new run to the first disk level's write group
                self.add_run_to_level_write_group(level_id as u32, new_run);
                // clear the mb-tree of the merge group and set thread_handle to be None
                self.in_mem_group[merge_index].clear();
            }
            // then handle the disk level's merge thread, note that here we do not need to care about whether the write group is full or not after committing the merge thread
            while level_id < self.levels.len() {
                // get the merge group's index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // get the next level id
                let next_level_id = level_id + 1;
                // check whether the merge group has thread_handle
                if let Some(handle) = self.levels[level_id].run_groups[merge_index].thread_handle.take() {
                    // get the merged new run
                    let new_run = handle.join().unwrap();
                    // add the new run to the next level
                    self.add_run_to_level_write_group(next_level_id as u32, new_run);
                    let merge_group_ref = &mut self.levels[level_id as usize].run_groups[merge_index];
                    // set the thread_handle to be None
                    merge_group_ref.thread_handle = None;
                    // remove all the runs in the merge group in levels[level_id]
                    let run_id_vec: Vec<u32> = merge_group_ref.run_vec.drain(..).map(|run| run.run_id).collect();
                    // remove the merged files in level_id by using multi-threads; note that we do not need to wait for the ending of the thread.
                    AsyncLevel::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
                }
                level_id += 1;
            }
        }
        // lastly persist the manifest
        self.update_manifest();
    }
}
