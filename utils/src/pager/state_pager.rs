use std::{io::{Read, Seek, Write}, os::unix::prelude::FileExt};
use crate::{cacher::CacheManager, types::{AddrKey, CompoundKey, StateValue}, File, OpenOptions};
use crate::pager::{Page, PAGE_SIZE};

use super::MAX_NUM_OLD_STATE_IN_PAGE;

pub struct StatePageWriter {
    pub file: File,
    pub vec_in_latest_update_page: Vec<(CompoundKey, StateValue)>,
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
    pub num_states: usize,
}

impl StatePageWriter {
    /* Initialize the writer using a given file name
    Storage Layout
    AddrKey, ver, value
     */
    pub fn create(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            vec_in_latest_update_page: vec![],
            num_stored_pages: 0,
            num_states: 0,
        }
    }

    /* Streamingly add the state to the file, state: (AddrKey, version, value)
     */
    pub fn append(&mut self, state: (AddrKey, u32, StateValue)) -> Option<CompoundKey> {
        let (cur_addr, cur_ver, state_value) = (state.0, state.1, state.2);
        let mut ret_key: Option<CompoundKey> = None; // the first key of the page, used for model learning
        self.vec_in_latest_update_page.push((CompoundKey::new(cur_addr, cur_ver), state_value));
        if self.vec_in_latest_update_page.len() == 1 && cur_addr != AddrKey::default() {
            let (first_key_in_page, _) = self.vec_in_latest_update_page[0];
            ret_key = Some(first_key_in_page);
        }
        if self.vec_in_latest_update_page.len() == 2 && self.vec_in_latest_update_page[0].0.addr == AddrKey::default() {
            let (first_key_in_page, _) = self.vec_in_latest_update_page[1];
            ret_key = Some(first_key_in_page);
        }
        if self.vec_in_latest_update_page.len() == MAX_NUM_OLD_STATE_IN_PAGE {
            self.flush();
        }
        self.num_states += 1;
        return ret_key;
    }

    pub fn flush(&mut self) {
        if self.vec_in_latest_update_page.len() != 0 {
            // first put the vector into a page
            let page = Page::from_state_vec_old_design(&self.vec_in_latest_update_page);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
            self.file.write_all(&page.block).unwrap();
            self.vec_in_latest_update_page.clear();
            self.num_stored_pages += 1;
        }
    }

    pub fn to_state_reader(self) -> StatePageReader {
        let file = self.file;
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        StatePageReader {
            file,
            num_stored_pages,
        }
    }
}

pub struct StatePageReader {
    pub file: File, // file object of the corresponding value file
    pub num_stored_pages: usize, // num of stored pages
}

impl StatePageReader {
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        Self {
            file,
            num_stored_pages,
        }
    }

    pub fn read_deser_states_at(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> Vec<(CompoundKey, StateValue)> {
        // first check whether the cache contains the page
        let r = cache_manager.read_state_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let page = r.unwrap();
            page.to_state_vec_old_design()
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.read_exact_at(&mut bytes, offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let v = page.to_state_vec_old_design();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_state_cache(run_id, page_id, page);
            return v;
        }
    }

    pub fn query_merkle_index(&mut self, search_addr_key: &AddrKey, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> (bool, Vec<(AddrKey, usize)>) {
        let collection = self.read_deser_states_at(run_id, page_id, cache_manager);
        let mut addr_keys: Vec<AddrKey> = Vec::new();
        let mut base_index = page_id * MAX_NUM_OLD_STATE_IN_PAGE;
        // get first and last key
        let first_addr_key = collection.first().unwrap().0.addr;
        let last_addr_key = collection.last().unwrap().0.addr;
        if search_addr_key < &first_addr_key {
            // load the prevoius page if exist
            if page_id >= 1 {
                let prev_page_id = page_id - 1;
                let prev_collection = self.read_deser_states_at(run_id, prev_page_id, cache_manager);
                let first_key_in_prev = prev_collection.first().unwrap().0.addr;
                if search_addr_key < &first_key_in_prev && page_id >= 2 {
                    // due to the precision error, we should look up the page_id - 2's page
                    base_index = (page_id - 2) * MAX_NUM_OLD_STATE_IN_PAGE;
                    let prev_prev_collection = self.read_deser_states_at(run_id, prev_page_id - 1, cache_manager);
                    addr_keys.extend(prev_prev_collection.iter().map(|(k, _)| k.addr));
                    addr_keys.extend(prev_collection.iter().map(|(k, _)| k.addr));
                } else {
                    base_index = (page_id - 1) * MAX_NUM_OLD_STATE_IN_PAGE;
                    addr_keys.extend(prev_collection.iter().map(|(k, _)| k.addr));
                    addr_keys.extend(collection.iter().map(|(k, _)| k.addr));
                }
            } else {
                addr_keys.extend(collection.iter().map(|(k, _)| k.addr));
            }
        } else if search_addr_key > &last_addr_key{
            // load the next page if exist
            if page_id + 1 < self.num_stored_pages {
                let next_page_id = page_id + 1;
                let next_collection = self.read_deser_states_at(run_id, next_page_id, cache_manager);
                let last_key_in_next = next_collection.last().unwrap().0.addr;
                if search_addr_key > &last_key_in_next && page_id + 1 < self.num_stored_pages - 1 {
                    // due to the precision error, we should look up the page_id + 2's page
                    base_index = (page_id + 1) * MAX_NUM_OLD_STATE_IN_PAGE;
                    let next_next_collection = self.read_deser_states_at(run_id, next_page_id + 1, cache_manager);
                    addr_keys.extend(next_collection.iter().map(|(k, _)| k.addr));
                    addr_keys.extend(next_next_collection.iter().map(|(k, _)| k.addr));
                } else {
                    addr_keys.extend(collection.iter().map(|(k, _)| k.addr));
                    addr_keys.extend(next_collection.iter().map(|(k, _)| k.addr));
                }
            } else {
                addr_keys.extend(collection.iter().map(|(k, _)| &k.addr));
            }
        } else {
            addr_keys.extend(collection.iter().map(|(k, _)| &k.addr));
        }
        let result_index = binary_search_of_addr(&addr_keys, search_addr_key);
        let result = addr_keys[result_index];
        if &result == search_addr_key {
            // match addr key
            return (true, vec![(result, result_index + base_index)]);
        } else {
            // mismatch addr key
            if result_index + 1 < addr_keys.len() {
                let next_result = addr_keys[result_index + 1];
                return (false, vec![(result, result_index + base_index), (next_result, result_index + 1 + base_index)]);
            } else {
                return (true, vec![(result, result_index + base_index)]);
            }
        }
    }

    pub fn query_page_state(&mut self, search_key: &CompoundKey, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> (CompoundKey, StateValue) {
        let mut state_v = Vec::<(CompoundKey, StateValue)>::new();
        // first load page_id's states
        let state_collection = self.read_deser_states_at(run_id, page_id, cache_manager);
        // get first and last model's key
        let first_compound_key = state_collection.first().unwrap().0;
        let last_compound_key = state_collection.last().unwrap().0;
        if search_key < &first_compound_key {
            // load the previous page if exist
            if page_id >= 1 {
                // println!("load prev page");
                let prev_page_id = page_id - 1;
                let prev_collection = self.read_deser_states_at(run_id, prev_page_id, cache_manager);
                let first_key_in_prev = prev_collection.first().unwrap().0;
                if search_key < &first_key_in_prev && page_id >= 2 {
                    // due to the precision error, we should look up the page_id - 2's page
                    let prev_prev_collection = self.read_deser_states_at(run_id, prev_page_id - 1, cache_manager);
                    state_v.extend(&prev_prev_collection);
                    state_v.extend(&prev_collection);
                } else {
                    state_v.extend(&prev_collection);
                    state_v.extend(&state_collection);
                }
                // println!("state_v: {:?}", state_v);
            } else {
                state_v.extend(&state_collection);
            }
        } else if search_key > &last_compound_key {
            // load the next page if exist
            if page_id + 1 < self.num_stored_pages {
                // println!("load next page");
                let next_page_id = page_id + 1;
                let next_collection = self.read_deser_states_at(run_id, next_page_id, cache_manager);
                let last_key_in_next = next_collection.last().unwrap().0;
                if search_key > &last_key_in_next && page_id + 1 < self.num_stored_pages - 1 {
                    // due to the precision error, we should look up the page_id + 2's page
                    let next_next_collection = self.read_deser_states_at(run_id, next_page_id + 1, cache_manager);
                    state_v.extend(&next_collection);
                    state_v.extend(&next_next_collection);
                } else {
                    state_v.extend(&state_collection);
                    state_v.extend(&next_collection);
                }
                // println!("state_v: {:?}", state_v);
            } else {
                state_v.extend(&state_collection);
            }
        } else {
            state_v.extend(&state_collection);
            // println!("inner page state_v: {:?}", state_v);
        }

        let (_, r) = binary_search_of_key(&state_v, search_key);
        return r.unwrap();
    }

    pub fn read_all_state(&mut self, run_id: u32) -> Vec<(CompoundKey, StateValue)> {
        let mut r = Vec::new();
        let mut cache_manager = CacheManager::new();
        for i in 0..self.num_stored_pages {
            let collections = self.read_deser_states_at(run_id, i, &mut cache_manager);
            r.extend(collections);
        }
        return r;
    }

    pub fn to_state_iter(self) -> StateIterator {
        let mut file =  self.file;
        let num_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut cur_vec_of_page = Vec::new();
        if num_pages != 0 {
            // load the first page
            let mut bytes = [0u8; PAGE_SIZE];
            let offset = 0;
            file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
            file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            // deserialize the page to state vector
            cur_vec_of_page = page.to_state_vec_old_design();
        }
        StateIterator {
            file,
            cur_vec_of_page,
            cur_state_pos_in_page: 0,
            cur_page_id: 0,
            num_pages,
        }
    }
}

pub fn binary_search_of_addr(v: &Vec<AddrKey>, key: &AddrKey) -> usize {
    let mut index: usize;
    let len = v.len();
    let mut l: i32 = 0;
    let mut r: i32 = len as i32 - 1;
    if len == 0 {
        return 0;
    }

    while l <= r && l >=0 && r <= len as i32 - 1{
        let m = l + (r - l) / 2;
        if &v[m as usize] < key {
            l = m + 1;
        }
        else if &v[m as usize] > key {
            r = m - 1;
        }
        else {
            index = m as usize;
            return index;
        }
    }
    
    index = l as usize;
    if index == len {
        index -= 1;
    }

    if key < &v[index] && index > 0 {
        index -= 1;
    }
    return index;
}

pub fn binary_search_of_key(v: &Vec<(CompoundKey, StateValue)>, key: &CompoundKey) -> (usize, Option<(CompoundKey, StateValue)>) {
    let mut index: usize;
    let len = v.len();
    let mut l: i32 = 0;
    let mut r: i32 = len as i32 - 1;
    if len == 0 {
        return (0, None);
    }

    while l <= r && l >=0 && r <= len as i32 - 1{
        let m = l + (r - l) / 2;
        if &v[m as usize].0 < key {
            l = m + 1;
        }
        else if &v[m as usize].0 > key {
            r = m - 1;
        }
        else {
            index = m as usize;
            return (index, Some(v[index].clone()));
        }
    }
    
    index = l as usize;
    if index == len {
        index -= 1;
    }

    if key < &v[index].0 && index > 0 {
        index -= 1;
    }
    return (index, Some(v[index].clone()));
}

pub struct InMemStateIterator {
    pub states: Vec<(CompoundKey, StateValue)>,
    pub cur_state_pos: usize, // position of current state
}

impl InMemStateIterator {
    // create a new in-memory state iterator using the input state vector
    pub fn create(states: Vec<(CompoundKey, StateValue)>) -> Self {
        Self {
            states,
            cur_state_pos: 0,
        }
    }
}

/* Implementation of the iterator trait.
 */
impl Iterator for InMemStateIterator {
    // the data type of each iterated item is the state (compound key-value pair)
    type Item = (CompoundKey, StateValue);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_state_pos >= self.states.len() {
            // already reached the last state, return None 
            return None;
        } else {
            let r = self.states[self.cur_state_pos];
            self.cur_state_pos += 1;
            return Some(r);
        }
    }
}

pub struct StateIterator {
    pub file: File,
    pub cur_vec_of_page: Vec<(CompoundKey, StateValue)>, // cache of the current deserialized vector of page
    pub cur_state_pos_in_page: usize, // position of current state
    pub cur_page_id: usize,
    pub num_pages: usize,
}

impl StateIterator {
    /* Create a new state iterator by given the file handler.
     */
    pub fn create(file_name: &str) -> Self {
        let mut file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut cur_vec_of_page = Vec::new();
        if num_pages != 0 {
            // load the first page
            let mut bytes = [0u8; PAGE_SIZE];
            let offset = 0;
            file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
            file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            // deserialize the page to state vector
            cur_vec_of_page = page.to_state_vec_old_design();
        }
        Self {
            file,
            cur_vec_of_page,
            cur_state_pos_in_page: 0,
            cur_page_id: 0,
            num_pages,
        }
    }
}

/* Implementation of the iterator trait.
 */
impl Iterator for StateIterator {
    // the data type of each iterated item is the vector of states with a shared AddrKey
    type Item = (CompoundKey, StateValue);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_page_id >= self.num_pages {
            // already reached the end, return None
            return None;
        } else {
            if self.cur_state_pos_in_page == self.cur_vec_of_page.len() {
                self.cur_page_id += 1;
                self.cur_state_pos_in_page = 0;
                if self.cur_page_id < self.num_pages {
                    // load the next page
                    let mut bytes = [0u8; PAGE_SIZE];
                    let offset = self.cur_page_id * PAGE_SIZE;
                    self.file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
                    self.file.read_exact(&mut bytes).unwrap();
                    let page = Page::from_array(bytes);
                    // deserialize the page to state vector
                    self.cur_vec_of_page = page.to_state_vec_old_design();
                    let r = self.cur_vec_of_page[self.cur_state_pos_in_page];
                    self.cur_state_pos_in_page += 1;
                    return Some(r);
                } else {
                    return None;
                }
            } else {
                let r = self.cur_vec_of_page[self.cur_state_pos_in_page];
                self.cur_state_pos_in_page += 1;
                return Some(r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use primitive_types::{H160, H256};
    use crate::types::{Address, StateKey};
    use rand::{rngs::StdRng, SeedableRng};
    use super::*;
    
    #[test]
    fn test_write_and_read_state() {
        let n = 12345;
        let mut rng = StdRng::seed_from_u64(1);
        let mut addr_vec: Vec<AddrKey> = (1..=n).map(|_| generate_addr(&mut rng)).collect();
        addr_vec.sort();

        let ver_num: u32 = 1;
        let mut states = Vec::<(CompoundKey, StateValue)>::new();
        for addr in &addr_vec {
            for i in 1..=ver_num {
                let ver = i;
                let value = StateValue(H256::from_low_u64_be(i as u64));
                states.push((CompoundKey::new(*addr, ver), value));
            }
        }
        let file_name = "state.dat";
        let mut state_page_writer = StatePageWriter::create(&file_name);
        for (key, value) in &states {
            state_page_writer.append((key.addr, key.version, *value));
        }
        state_page_writer.flush();

        // read
        let mut state_page_reader = state_page_writer.to_state_reader();
        let num_of_pages = state_page_reader.num_stored_pages;
        let mut total_state_read = Vec::<(CompoundKey, StateValue)>::new();

        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..num_of_pages {
            let v = state_page_reader.read_deser_states_at(run_id, i, &mut cache_manager);
            total_state_read.extend(v);
        }
        
        assert_eq!(states, total_state_read);
    }

    fn generate_addr(rng: &mut StdRng) -> AddrKey {
        AddrKey { addr: Address(H160::random_using(rng)), state_key: StateKey(H256::random_using(rng)) }
    }
}