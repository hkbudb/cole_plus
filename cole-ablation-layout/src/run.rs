use utils::{cacher::CacheManagerOld, config::{compute_bitmap_size_in_bytes, Configs}, old_design_merge_sort::MergeElement, pager::{model_pager::{ModelPageReader, ModelPageWriter, StreamModelConstructor}, old_mht_pager::{HashPageReaderOld, HashPageWriterOld, StreamMHTConstructor}, old_state_pager::{InMemStateIteratorOld, StateIteratorOld, StatePageReaderOld, StatePageWriterOld}}, types::{bytes_hash, AddrKey, CompoundKey, StateValue}, OpenOptions, Read, Write};
use primitive_types::{H160, H256};
use std::{cmp::{max, min}, collections::BinaryHeap};
use growable_bloom_filter::GrowableBloom;
use std::fmt::Debug;
const FILTER_FP_RATE: f64 = 0.1; // false positive rate of a filter
const MAX_FILTER_SIZE: usize = 1024 * 1024; // 1M

// define a run in a level
pub struct LevelRun {
    pub run_id: u32, // id of this run
    pub state_reader: StatePageReaderOld, // state's reader
    pub latest_state_reader: StatePageReaderOld, // latest value reader
    pub model_reader: ModelPageReader, // model's reader
    pub latest_state_model_reader: ModelPageReader, // latest value's model reader
    pub mht_reader: HashPageReaderOld, // mht's reader
    pub filter: Option<GrowableBloom>, // bloom filter
    pub filter_hash: Option<H256>, // filter's hash
    pub digest: H256, // a digest of mht_root and filter's hash if filter exists
}

impl LevelRun {
    // load a run using the run's id, level's id and the configuration reference
    pub fn load(run_id: u32, level_id: u32, configs: &Configs) -> Self {
        // first define the file names of the state, model, mht, and filter
        let state_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "s");
        let latest_state_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "ls");
        let model_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "m");
        let latest_state_model_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "lm");
        let mht_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "h");
        let filer_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "f");

        // load the three readers using their file names
        let state_reader = StatePageReaderOld::load(&state_file_name);
        let latest_state_reader = StatePageReaderOld::load(&latest_state_file_name);
        let model_reader = ModelPageReader::load(&model_file_name);
        let latest_state_model_reader = ModelPageReader::load(&latest_state_model_file_name);
        let mht_reader = HashPageReaderOld::load(&mht_file_name);
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

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            latest_state_reader,
            model_reader,
            latest_state_model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
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
    pub fn construct_run_by_in_memory_merge(inputs: Vec<InMemStateIteratorOld>, run_id: u32, level_id: u32, dir_name: &str, epsilon: i64, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let latest_state_file_name = Self::file_name(run_id, level_id, &dir_name, "ls");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let latest_state_model_file_name = Self::file_name(run_id, level_id, &dir_name, "lm");
        let mht_file_name = Self::file_name(run_id, level_id, &dir_name, "h");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_writer, latest_state_writer, model_writer, latest_state_model_writer, mht_writer, filter) = in_memory_merge_ablation_design(inputs, &state_file_name, &latest_state_file_name, &model_file_name, &latest_state_model_file_name, &mht_file_name, epsilon, fanout, filter);

        let state_reader = state_writer.to_state_reader_old();
        let latest_state_reader = latest_state_writer.to_state_reader_old();
        let model_reader = model_writer.to_model_reader();
        let latest_state_model_reader = latest_state_model_writer.to_model_reader();
        let mht_reader = mht_writer.to_hash_reader_old();

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            latest_state_reader,
            model_reader,
            latest_state_model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
        }
    }

    // use the state iterator to process the merge operation
    pub fn construct_run_by_merge(inputs: Vec<StateIteratorOld>, run_id: u32, level_id: u32, dir_name: &str, epsilon: i64, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let latest_state_file_name = Self::file_name(run_id, level_id, &dir_name, "ls");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let latest_state_model_file_name = Self::file_name(run_id, level_id, &dir_name, "lm");
        let mht_file_name = Self::file_name(run_id, level_id, &dir_name, "h");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_writer, latest_state_writer, model_writer, latest_state_model_writer, mht_writer, filter) = merge_ablation_design(inputs, &state_file_name, &latest_state_file_name, &model_file_name, &latest_state_model_file_name, &mht_file_name, epsilon, fanout, filter);

        let state_reader = state_writer.to_state_reader_old();
        let latest_state_reader = latest_state_writer.to_state_reader_old();
        let model_reader = model_writer.to_model_reader();
        let latest_state_model_reader = latest_state_model_writer.to_model_reader();
        let mht_reader = mht_writer.to_hash_reader_old();

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            latest_state_reader,
            model_reader,
            latest_state_model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
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

    pub fn search_run(&mut self, addr_key: AddrKey, configs: &Configs, cache_manager: &mut CacheManagerOld) -> Option<(CompoundKey, StateValue)> {
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&addr_key) {
                return None;
            }
        }
        // use the model file to predict the position in the state file
        // compute the boundary compound key
        let upper_key = CompoundKey {
            addr: addr_key,
            version: u32::MAX,
        };
        let epsilon = configs.epsilon;
        /* use additional files for ablation study */
        // use model file to predict the pos
        let pred_pos = self.latest_state_model_reader.get_pred_state_pos(self.run_id, &upper_key, epsilon, cache_manager) as i64;
        let num_of_states = self.latest_state_reader.num_states;
        // compute the lower position and upper position according to the pred_pos and epsilon
        let pos_l = min(max(pred_pos - epsilon - 1, 0), num_of_states as i64 - 1) as usize;
        let pos_r = min(pred_pos + epsilon + 2, num_of_states as i64 - 1) as usize;
        // load the states from the value file given the range [pos_l, pos_r]
        let states = self.latest_state_reader.read_deser_states_range(self.run_id, pos_l, pos_r, cache_manager);
        // binary search the loaded vector using the upper key
        let (_, res) = binary_search_of_key(&states, upper_key);
        if res.is_some() {
            let res = res.unwrap();
            let res_addr = res.0.addr;
            if res_addr == addr_key {
                return Some(res);
            }
        }
        return None;
    }

    /* Omit the code for prove and verification for ablation study */
    // derive the digest of the LevelRun
    pub fn compute_digest(&self) -> H256 {
        let mht_root = self.mht_reader.root.unwrap();
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

pub fn in_memory_merge_ablation_design(mut inputs: Vec<InMemStateIteratorOld>, output_state_file_name: &str, output_latest_state_file_name: &str, output_model_file_name: &str, output_latest_state_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriterOld, StatePageWriterOld, ModelPageWriter, ModelPageWriter, HashPageWriterOld, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriterOld::create(output_state_file_name);
    let mut latest_state_writer = StatePageWriterOld::create(output_latest_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
    let mut latest_state_model_constructor = StreamModelConstructor::new(output_latest_state_model_file_name, epsilon);
    let mut mht_constructor = StreamMHTConstructor::new(output_mht_file_name, fanout);
    let mut minheap = BinaryHeap::<MergeElement>::new();
    let k = inputs.len();

    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = (CompoundKey::new(AddrKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into()), 0), StateValue(H256::from_low_u64_be(0)));
    let max_state = (CompoundKey::new(AddrKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into()), u32::MAX), StateValue(H256::from_low_u64_be(0)));
    // add the state's key to the model constructor
    // model_constructor.append_state_key(&min_state.key);
    // add the state's hash to the mht constructor
    let mut min_state_bytes = vec![];
    min_state_bytes.extend(min_state.0.to_bytes());
    min_state_bytes.extend(min_state.1.to_bytes());
    mht_constructor.append_hash(bytes_hash(&min_state_bytes));
    // add the smallest state to the writer
    state_writer.append(min_state);
    
    // add the first states from each iterator
    for i in 0..k {
        let r = inputs[i].next().unwrap();
        let elem = MergeElement {
            state: r,
            i,
        };
        minheap.push(elem);
    }
    // flag of number of full iterators
    let mut full_cnt = 0;
    // let mut model_timer = 0;
    let mut latest_compound_key = CompoundKey::default();
    let mut latest_value = StateValue::default();
    // let mut total_learned_keys = 0;
    // let mut latest_learned_keys = 0;
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            // let start = std::time::Instant::now();
            model_constructor.append_state_key(&state.0);
            // total_learned_keys += 1;
            // let elapse = start.elapsed().as_nanos();
            // model_timer += elapse;
            // insert the state's key to the bloom filter
            if filter.is_some() {
                let addr_key = state.0.addr;
                filter.as_mut().unwrap().insert(addr_key);
            }
            // add the state's hash to the mht constructor
            let mut state_bytes = vec![];
            state_bytes.extend(state.0.to_bytes());
            state_bytes.extend(state.1.to_bytes());
            mht_constructor.append_hash(bytes_hash(&state_bytes));
            // add the smallest state to the writer
            state_writer.append(state);


            /* additional files for ablation study */
            if state.0.addr != latest_compound_key.addr {
                // new addr key, write latest_compound_key and value to the latest_state_file and add the compound key to the latest state model
                if latest_compound_key != CompoundKey::default() {
                    latest_state_model_constructor.append_state_key(&latest_compound_key);
                    latest_state_writer.append((latest_compound_key, latest_value));
                    // latest_learned_keys += 1;
                }
            }
            latest_compound_key = state.0;
            latest_value = state.1;
        }

        let i = elem.i;
        let r = inputs[i].next();
        if r.is_some() {
            // load the next smallest state from the iterator
            let state = r.unwrap();
            // create a new merge element with the previously loaded next smallest state
            let elem = MergeElement {
                state,
                i,
            };
            // push the element to the heap
            minheap.push(elem);
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
    }

    // add the max state as the upper boundary to help the completeness check
    // add the state's hash to the mht constructor
    let mut state_bytes = vec![];
    state_bytes.extend(max_state.0.to_bytes());
    state_bytes.extend(max_state.1.to_bytes());
    mht_constructor.append_hash(bytes_hash(&state_bytes));
    // add the max state to the writer
    state_writer.append(max_state);

    // flush the state writer
    state_writer.flush();
    // finalize the model constructor
    // let start = std::time::Instant::now();
    model_constructor.finalize_append();
    // let elapse = start.elapsed().as_nanos();
    // model_timer += elapse;
    // println!("model timer: {}", model_timer);
    mht_constructor.build_mht();

    /* additional files for ablation study */
    if latest_compound_key != CompoundKey::default() {
        latest_state_model_constructor.append_state_key(&latest_compound_key);
        latest_state_writer.append((latest_compound_key, latest_value));
        // latest_learned_keys += 1;
    }
    latest_state_writer.flush();
    latest_state_model_constructor.finalize_append();
    // print!("total learned keys: {}, latest learned keys: {}", total_learned_keys, latest_learned_keys);
    return (state_writer, latest_state_writer, model_constructor.output_model_writer, latest_state_model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
}

pub fn merge_ablation_design(mut inputs: Vec<StateIteratorOld>, output_state_file_name: &str, output_latest_state_file_name: &str, output_model_file_name: &str, output_latest_state_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriterOld, StatePageWriterOld, ModelPageWriter, ModelPageWriter, HashPageWriterOld, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriterOld::create(output_state_file_name);
    let mut latest_state_writer = StatePageWriterOld::create(output_latest_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
    let mut latest_state_model_constructor = StreamModelConstructor::new(output_latest_state_model_file_name, epsilon);
    let mut mht_constructor = StreamMHTConstructor::new(output_mht_file_name, fanout);
    let mut minheap = BinaryHeap::<MergeElement>::new();
    let k = inputs.len();
    
    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = (CompoundKey::new(AddrKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into()), 0), StateValue(H256::from_low_u64_be(0)));
    let max_state = (CompoundKey::new(AddrKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into()), u32::MAX), StateValue(H256::from_low_u64_be(0)));
    // add the state's key to the model constructor
    // model_constructor.append_state_key(&min_state.key);
    // add the state's hash to the mht constructor
    let mut min_state_bytes = vec![];
    min_state_bytes.extend(min_state.0.to_bytes());
    min_state_bytes.extend(min_state.1.to_bytes());
    mht_constructor.append_hash(bytes_hash(&min_state_bytes));
    // add the smallest state to the writer
    state_writer.append(min_state);

    // add the first states from each iterator
    for i in 0..k {
        let r = inputs[i].next().unwrap();
        let elem = MergeElement {
            state: r,
            i,
        };
        minheap.push(elem);
    }
    
    // flag of number of full iterators
    let mut full_cnt = 0;
    let mut latest_compound_key = CompoundKey::default();
    let mut latest_value = StateValue::default();
    // let mut total_learned_keys = 0;
    // let mut latest_learned_keys = 0;
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            model_constructor.append_state_key(&state.0);
            // total_learned_keys += 1;
            // insert the state's key to the bloom filter
            if filter.is_some() {
                let addr_key = state.0.addr;
                filter.as_mut().unwrap().insert(addr_key);
            }
            // add the state's hash to the mht constructor
            let mut state_bytes = vec![];
            state_bytes.extend(state.0.to_bytes());
            state_bytes.extend(state.1.to_bytes());
            mht_constructor.append_hash(bytes_hash(&state_bytes));
            // add the smallest state to the writer
            state_writer.append(state);

            /* additional files for ablation study */
            if state.0.addr != latest_compound_key.addr {
                // new addr key, write latest_compound_key and value to the latest_state_file and add the compound key to the latest state model
                if latest_compound_key != CompoundKey::default() {
                    latest_state_model_constructor.append_state_key(&latest_compound_key);
                    latest_state_writer.append((latest_compound_key, latest_value));
                    // latest_learned_keys += 1
                }
            }
            latest_compound_key = state.0;
            latest_value = state.1;
        }

        let i = elem.i;
        let r = inputs[i].next();
        if r.is_some() {
            // load the next smallest state from the iterator
            let state = r.unwrap();
            // create a new merge element with the previously loaded next smallest state
            let elem = MergeElement {
                state,
                i,
            };
            // push the element to the heap
            minheap.push(elem);
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
    }

    // add the max state as the upper boundary to help the completeness check
    // add the state's hash to the mht constructor
    let mut state_bytes = vec![];
    state_bytes.extend(max_state.0.to_bytes());
    state_bytes.extend(max_state.1.to_bytes());
    mht_constructor.append_hash(bytes_hash(&state_bytes));
    // add the max state to the writer
    state_writer.append(max_state);
    
    // flush the state writer
    state_writer.flush();
    // finalize the model constructor
    model_constructor.finalize_append();
    mht_constructor.build_mht();

    /* additional files for ablation study */
    if latest_compound_key != CompoundKey::default() {
        latest_state_model_constructor.append_state_key(&latest_compound_key);
        latest_state_writer.append((latest_compound_key, latest_value));
        // latest_learned_keys += 1
    }
    latest_state_writer.flush();
    latest_state_model_constructor.finalize_append();
    // print!("total learned keys: {}, latest learned keys: {}", total_learned_keys, latest_learned_keys);
    return (state_writer, latest_state_writer, model_constructor.output_model_writer, latest_state_model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
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

pub fn binary_search_of_key(v: &Vec<(CompoundKey, StateValue)>, key: CompoundKey) -> (usize, Option<(CompoundKey, StateValue)>) {
    let mut index: usize;
    let len = v.len();
    let mut l: i32 = 0;
    let mut r: i32 = len as i32 - 1;
    if len == 0 {
        return (0, None);
    }

    while l <= r && l >=0 && r <= len as i32 - 1{
        let m = l + (r - l) / 2;
        if v[m as usize].0 < key {
            l = m + 1;
        }
        else if v[m as usize].0 > key {
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

    if key < v[index].0 && index > 0 {
        index -= 1;
    }
    return (index, Some(v[index].clone()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use utils::types::{CompoundKey, StateValue};
    use primitive_types::{H160, H256};
    use utils::config::Configs;

    #[test]
    fn test_in_memory_merge_and_run_construction_abaltion() {
        let k: usize = 2;
        let n: usize = 10;
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 46;
        let fanout = 2;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }

        std::fs::create_dir(dir_name).unwrap_or_default();
        let mut iters = Vec::new();
        for _ in 0..k {
            let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
            for i in 0..n {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let version = i as u32;
                let value = H256::random_using(&mut rng);
                let key = CompoundKey::new(AddrKey::new(acc_addr.into(), state_addr.into()), version);
                let value = StateValue(value);
                state_vec.push((key, value));
            }
            let min_key = CompoundKey::new(AddrKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into()), 0);
            let max_key = CompoundKey::new(AddrKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into()), u32::MAX);
            state_vec.push((min_key, StateValue(H256::default())));
            state_vec.push((max_key, StateValue(H256::default())));
            state_vec.sort();
            
            
            let it = InMemStateIteratorOld::create(state_vec);
            iters.push(it);
        }

        let run_id = 1;
        let level_id = 0;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n as usize,
            size_ratio: k as usize,
            is_pruned: false,
            test_in_mem_roll: false,
            test_disk_roll: false,
        };

        let run = LevelRun::construct_run_by_in_memory_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, k as usize);
        run.persist_filter(level_id, &configs);
        drop(run);
    }

}