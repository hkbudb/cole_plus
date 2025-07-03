use std::cmp::{Ordering, PartialOrd, Ord};
use std::collections::BinaryHeap;
use crate::pager::old_mht_pager::{StreamMHTConstructor, HashPageWriterOld};
use crate::pager::model_pager::{StreamModelConstructor, ModelPageWriter};
use crate::pager::old_state_pager::{StateIteratorOld, StatePageWriterOld, InMemStateIteratorOld};
use crate::types::{bytes_hash, AddrKey, CompoundKey, StateValue};
use primitive_types::{H160, H256};
use growable_bloom_filter::GrowableBloom;

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct MergeElement {
    pub state: (CompoundKey, StateValue), // state
    pub i: usize, // the index of the file
}

impl PartialOrd for MergeElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return other.state.0.partial_cmp(&self.state.0);
    }
}

impl Ord for MergeElement {
    fn cmp(&self, other: &MergeElement) -> Ordering {
        return other.state.0.cmp(&self.state.0);
    }
}

pub fn merge_old_design(mut inputs: Vec<StateIteratorOld>, output_state_file_name: &str, output_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriterOld, ModelPageWriter, HashPageWriterOld, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriterOld::create(output_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
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
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            model_constructor.append_state_key(&state.0);
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
    return (state_writer, model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
}

pub fn in_memory_merge_old_design(mut inputs: Vec<InMemStateIteratorOld>, output_state_file_name: &str, output_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriterOld, ModelPageWriter, HashPageWriterOld, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriterOld::create(output_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
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
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            model_constructor.append_state_key(&state.0);
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
    return (state_writer, model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
}

#[cfg(test)]
mod tests {
    use crate::{OpenOptions, cacher::CacheManagerOld};
    use rand::{rngs::StdRng, SeedableRng};
    use crate::{pager::PAGE_SIZE, models::CompoundKeyModel};
    use super::*;

    #[test]
    fn generate_states() {
        let k: usize = 10;
        let n: usize = 1000;
        let mut rng = StdRng::seed_from_u64(1);
        let mut pagers = Vec::<StatePageWriterOld>::new();
        for i in 0..k {
            let mut pager = StatePageWriterOld::create(&format!("data{}.dat", i));
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
            state_vec.sort();
            for s in state_vec {
                pager.append(s);
            }
            pager.flush();
            pagers.push(pager);
        }
    }

    #[test]
    fn test_total_merge_old_design() {
        let k = 10;
        let n = 1000;
        let epsilon = 2;
        let fanout = 2;
        let mut iters = Vec::<StateIteratorOld>::new();
        for i in 0..k {
            let file = OpenOptions::new().create(true).read(true).write(true).open(&format!("data{}.dat", i)).unwrap();
            let iter = StateIteratorOld::create_with_num_states(file, n);
            iters.push(iter);
        }
        let filter = Some(GrowableBloom::new(0.1, n as usize));
        let r = merge_old_design(iters, "out_state.dat", "out_model.dat", "out_mht.dat", epsilon, fanout, filter);
        let (out_state_writer, out_model_writer, out_mht_writer, _) = r;
        let (mut out_state_reader, mut out_model_reader, mut out_mht_reader) = (out_state_writer.to_state_reader_old(), out_model_writer.to_model_reader(), out_mht_writer.to_hash_reader_old());
        let state_page_num = out_state_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut states = Vec::<(CompoundKey, StateValue)>::new();
        let mut cache_manager = CacheManagerOld::new();
        for page_id in 0..state_page_num {
            let v = out_state_reader.read_deser_page_at(0, page_id, &mut cache_manager);
            states.extend(&v);
        }
        println!("states len: {}", states.len());
        let mut models = Vec::<CompoundKeyModel>::new();
        let model_page_num = out_model_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..model_page_num {
            let v = out_model_reader.read_deser_page_at_old_version(0, page_id, &mut cache_manager).v;
            models.extend(&v);
        }
        println!("model len: {}", models.len());

        let mut hashes = Vec::<H256>::new();
        let hash_page_num = out_mht_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..hash_page_num {
            let v = out_mht_reader.read_deser_page_at(0, page_id, &mut cache_manager);
            hashes.extend(&v);
        }
        println!("hash len: {}", hashes.len());
    }
}