use growable_bloom_filter::GrowableBloom;
use primitive_types::H256;
use utils::{cacher::CacheManager, merge_sort::{in_memory_collect, merge_state}, pager::{cdc_mht::CDCTreeReader, model_pager::ModelPageReader, state_pager::{StateIterator, StatePageReader}, upper_mht::UpperMHTReader}, types::{bytes_hash, AddrKey, CompoundKey, StateValue}, OpenOptions, Read, Write};
use utils::config::Configs;
use std::fmt::{Debug, Formatter, Error};

const FILTER_FP_RATE: f64 = 0.1; // false positive rate of a filter
const MAX_FILTER_SIZE: usize = 1024 * 1024; // 1M
// define a run in a level
pub struct LevelRun {
    pub run_id: u32, // id of this run
    pub state_reader: StatePageReader, // state's reader
    pub model_reader: ModelPageReader, // model's reader
    pub upper_mht_reader: UpperMHTReader, // upper mht's reader (read each addr's tree offset)
    pub lower_cdc_tree_reader: CDCTreeReader, // lower cdc tree's reader (read each addr's versions)
    pub filter: Option<GrowableBloom>, // bloom filter
    pub filter_hash: Option<H256>, // filter's hash
    pub digest: H256, // a digest of mht_root and filter's hash if filter exists
}

impl LevelRun {
    // load a run using the run's id, level's id and the configuration reference
    pub fn load(run_id: u32, level_id: u32, configs: &Configs) -> Self {
        // first define the file names of the state, model, mht, and filter
        let state_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "lh");
        let filer_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "f");

        // load the readers using their file names
        let state_reader = StatePageReader::load(&state_file_name);
        let model_reader = ModelPageReader::load(&model_file_name);
        let upper_mht_reader = UpperMHTReader::new(&upper_offset_file_name, &upper_hash_file_name);
        let lower_cdc_tree_reader = CDCTreeReader::new(&lower_cdc_file_name);

        // initiate the filter using None object
        let mut filter = None;
        // if the filter's file exists, read the filter from the file
        match OpenOptions::new().read(true).open(&filer_file_name) {
            Ok(mut file) => {
                let mut len_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut len_bytes).unwrap();
                let len = u32::from_be_bytes(len_bytes);
                let mut v = vec![0u8; len as usize];
                file.read_exact(&mut v).unwrap();
                let filter_obj: GrowableBloom = bincode::deserialize(&v).unwrap();
                filter = Some(filter_obj);
            },
            Err(_) => {
            }
        }

        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    fn estimate_all_filter_size(level_id: u32, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> usize {
        let mut total_size = 0;
        let mut cur_level = level_id as i32;
        let mut cur_num_of_state = max_num_of_state;
        let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * level_num_of_run;
        total_size += cur_level_filter_size;
        while cur_level >= 0 {
            cur_level -= 1;
            cur_num_of_state /= size_ratio;
            let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * size_ratio;
            total_size += cur_level_filter_size;
        }
        return total_size;
    }

    // use the in-memory iterator to process the merge operation
    pub fn construct_run_by_in_memory_collection(state_vec: Vec<(CompoundKey, StateValue)>, run_id: u32, level_id: u32, dir_name: &str, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize, is_pruned: bool) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id,  &dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id,  &dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id,  &dir_name, "lh");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_reader, model_reader, upper_mht_reader, lower_cdc_tree_reader,  filter) = in_memory_collect(state_vec, &state_file_name, &model_file_name, &upper_offset_file_name, &upper_hash_file_name, &lower_cdc_file_name, fanout, filter, is_pruned);
        
        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);

        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    // use the state iterator to process the merge operation
    pub fn construct_run_by_merge(input_states: Vec<StateIterator>, lower_cdc_tree_readers: Vec::<CDCTreeReader>, upper_mht_readers: Vec::<(u32, UpperMHTReader)>, run_id: u32, level_id: u32, dir_name: &str, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize, is_pruned: bool) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id,  &dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id,  &dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id,  &dir_name, "lh");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_reader, model_reader, upper_mht_reader, lower_cdc_tree_reader, filter) = merge_state(input_states, lower_cdc_tree_readers, upper_mht_readers, &state_file_name, &model_file_name, &upper_offset_file_name, &upper_hash_file_name, &lower_cdc_file_name, fanout, filter, is_pruned);
        
        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);

        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    // helper function to generate the file name of different file types: "s", "m", "h"
    pub fn file_name(run_id: u32, level_id: u32, dir_name: &str, file_type: &str) -> String {
        format!("{}/{}_{}_{}.dat", dir_name, file_type, level_id, run_id)
    }

    // persist the filter if it exists
    pub fn persist_filter(&self, level_id: u32, configs: &Configs) {
        if self.filter.is_some() {
            // init the filter's file name
            let filer_file_name = Self::file_name(self.run_id, level_id, &configs.dir_name, "f");
            // serialize the filter using bincode
            let bytes = bincode::serialize(self.filter.as_ref().unwrap()).unwrap();
            // get the length of the serialized bytes
            let bytes_len = bytes.len() as u32;
            // v is a vector that will be persisted to the filter's file
            let mut v = bytes_len.to_be_bytes().to_vec();
            v.extend(&bytes);
            // write v to the file
            let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&filer_file_name).unwrap();
            file.write_all(&mut v).unwrap();
        }
    }

    pub fn search_run(&mut self, compound_key: &CompoundKey, cache_manager: &mut CacheManager) -> Option<(AddrKey, u32, StateValue)> {
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&compound_key.addr) {
                return None;
            }
        }
        let mut low = 0;
        let mut high = self.state_reader.num_stored_pages;
        let mut candidate_page_id: Option<usize> = None;

        while low < high {
            let mid = low + (high - low) / 2;
            let states_on_mid_page = self.state_reader.read_deser_states_at(self.run_id, mid, cache_manager);
            let last_key_on_page = &states_on_mid_page.last().unwrap().0; 
            if compound_key < last_key_on_page {
                candidate_page_id = Some(mid);
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        let target_page_id = match candidate_page_id {
            Some(id) => id,
            None => return None, // If no candidate page was found, the key is not in any page.
        };

        // 3. Search within the identified target page
        let states_on_target_page = self.state_reader.read_deser_states_at(self.run_id, target_page_id, cache_manager);
        return binary_search_of_key(&states_on_target_page, compound_key).1;
    }
    
    // pub fn search_run(&mut self, compound_key: &CompoundKey, cache_manager: &mut CacheManager) -> Option<(AddrKey, u32, StateValue)> {
    //     // try to use filter to test whether the key exists
    //     if self.filter.is_some() {
    //         if !self.filter.as_ref().unwrap().contains(&compound_key.addr) {
    //             return None;
    //         }
    //     }
    //     // use the model file to predict the position in the state file
    //     // use model file to predict the page_id
    //     let pred_page_id = self.model_reader.get_pred_state_page_id(self.run_id, compound_key, false, cache_manager);
    //     let (compound_key, state_value) = self.state_reader.query_page_state(compound_key, self.run_id, pred_page_id, cache_manager);
    //     return Some((compound_key.addr, compound_key.version, state_value));
    // }

    // derive the digest of the LevelRun
    pub fn compute_digest(&self) -> H256 {
        let mht_root = self.upper_mht_reader.mht_reader.root.unwrap();
        let mut bytes = mht_root.as_bytes().to_vec();
        if self.filter_hash.is_some() {
            bytes.extend(self.filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    // compute the digest of the run according to the MHT root and the filter if it exists
    pub fn load_digest(mht_root: H256, filter_hash: &Option<H256>) -> H256 {
        let mut bytes = mht_root.as_bytes().to_vec();
        if filter_hash.is_some() {
            bytes.extend(filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    pub fn filter_cost(&self) -> RunFilterSize {
        // filter cost
        let filter_ref = self.filter.as_ref();
        let filter_size;
        if filter_ref.is_some() {
            filter_size = filter_ref.unwrap().memory_size();
        } else {
            filter_size = 0;
        }
        return RunFilterSize::new(filter_size);
    }
}

pub fn binary_search_of_key(v: &Vec<(CompoundKey, StateValue)>, key: &CompoundKey) -> (usize, Option<(AddrKey, u32, StateValue)>) {
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
            let return_value = (v[index].0.addr, v[index].0.version, v[index].1);
            return (index, Some(return_value));
        }
    }
    
    index = l as usize;
    if index == len {
        index -= 1;
    }

    if key < &v[index].0 && index > 0 {
        index -= 1;
    }
    let return_value = (v[index].0.addr, v[index].0.version, v[index].1);
    return (index, Some(return_value));
}

#[derive(Debug, Clone, Copy)]
pub struct RunFilterSize {
    pub filter_size: usize,
}

impl RunFilterSize {
    pub fn new(filter_size: usize) -> Self {
        Self {
            filter_size,
        }
    }

    pub fn add(&mut self, other: &RunFilterSize) {
        self.filter_size += other.filter_size;
    }
}

impl Debug for LevelRun {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "<Level Run Info> run_id: {}, filter is some: {:?}, page len: {}", self.run_id, self.filter.is_some(), self.state_reader.num_stored_pages)
    }
}

/*
Compute a recommended bitmap size for items_count items
and a fp_p rate of false positives.
fp_p obviously has to be within the ]0.0, 1.0[ range.
 */
pub fn compute_bitmap_size_in_bytes(items_count: usize, fp_p: f64) -> usize {
    assert!(items_count > 0);
    assert!(fp_p > 0.0 && fp_p < 1.0);
    // We're using ceil instead of round in order to get an error rate <= the desired.
    // Using round can result in significantly higher error rates.
    let num_slices = ((1.0 / fp_p).log2()).ceil() as u64;
    let slice_len_bits = (items_count as f64 / 2f64.ln()).ceil() as u64;
    let total_bits = num_slices * slice_len_bits;
    // round up to the next byte
    let buffer_bytes = ((total_bits + 7) / 8) as usize;
    buffer_bytes
}



#[cfg(test)]
mod tests {
    
}