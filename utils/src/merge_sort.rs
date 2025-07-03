use std::cmp::{Ordering, PartialOrd, Ord};
use std::collections::{BinaryHeap, HashMap};
use primitive_types::{H160, H256};
use crate::cacher::CacheManager;
use crate::pager::cdc_mht::{merge_two_cdc_trees, CDCTree, CDCTreeReader, CDCTreeWriter, VerObject};
use crate::pager::model_pager::{ModelConstructor, ModelPageReader};
use crate::pager::state_pager::{StateIterator, StatePageReader, StatePageWriter};
use crate::pager::upper_mht::{UpperMHTReader, UpperMHTWriter};
use crate::types::{bytes_hash, AddrKey, Address, CompoundKey, StateKey, StateValue};
use growable_bloom_filter::GrowableBloom;
use cdc_hash::{DEFAULT_GEAR_HASH_LEVEL, DEFAULT_MAX_NODE_CAPACITY, DEFAULT_FANOUT};
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
    fn cmp(&self, other: &Self) -> Ordering {
        return other.state.0.cmp(&self.state.0);
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct InMemMergeElement {
    pub state: (AddrKey, u32, StateValue),
    pub i: usize,
}

impl PartialOrd for InMemMergeElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // use AddrKey as the sort key
        return other.state.0.partial_cmp(&self.state.0);
    }
}

impl Ord for InMemMergeElement {
    fn cmp(&self, other: &Self) -> Ordering {
        return other.state.0.cmp(&self.state.0);
    }
}

pub fn merge_state(mut input_states: Vec<StateIterator>, mut lower_cdc_tree_readers: Vec::<CDCTreeReader>, mut upper_mht_readers: Vec::<(u32, UpperMHTReader)>, output_state_file_name: &str, output_model_file_name: &str, output_upper_offset_file_name: &str, output_upper_hash_file_name: &str, output_lower_cdc_file_name: &str, fanout: usize, mut filter: Option<GrowableBloom>, is_pruned: bool) -> (StatePageReader, ModelPageReader, UpperMHTReader, CDCTreeReader, Option<GrowableBloom>) {
    // create a writer for recording the states
    let mut state_writer = StatePageWriter::create(output_state_file_name);
    // create a writer for models
    let mut model_constructor = ModelConstructor::new(output_model_file_name);
    // create upper mht writer and lower cdc tree writer
    let mut upper_mht_writer = UpperMHTWriter::new(output_upper_offset_file_name, output_upper_hash_file_name, fanout);
    let mut lower_cdc_tree_writer = CDCTreeWriter::new(output_lower_cdc_file_name);
    // create an offset counter map to record the offset of k merging cdc-trees
    let mut offset_index_counter = HashMap::<usize, usize>::new();
    
    let mut min_heap = BinaryHeap::<MergeElement>::new();
    let k = input_states.len();
    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = (AddrKey::new(Address(H160::from_low_u64_be(0)), StateKey(H256::from_low_u64_be(0))), 0u32, StateValue(H256::from_low_u64_be(0)));
    let max_state = (AddrKey::new(Address(H160::from_slice(&vec![255u8;20])), StateKey(H256::from_slice(&vec![255u8;32]))), u32::MAX, StateValue(H256::from_low_u64_be(0)));
    // println!("write {:?}", min_state);
    state_writer.append(min_state);

    // add the first elems from each iterator, init the offset counter map
    for i in 0..k {
        // get the first element of the i-th input iterator
        let r = input_states[i].next().unwrap();
        let elem = MergeElement {
            state: r,
            i,
        };
        min_heap.push(elem);
        offset_index_counter.insert(i, 0); // set each offset of cdc tree to be 0
    }
    // create a temp var
    let mut temp_var: (AddrKey, Vec<(u32, StateValue)>) = (min_state.0, vec![]);
    // create a temp cdc_tree
    let mut lower_cdc_tree = CDCTree::new(DEFAULT_FANOUT, DEFAULT_GEAR_HASH_LEVEL, DEFAULT_MAX_NODE_CAPACITY);
    // flag of number of full iterators
    let mut full_cnt = 0;
    let mut prev_input_index = 0;
    let mut temp_cache_manager = CacheManager::new();
    while full_cnt < k {
        // pop the smallest elem from the heap
        let elem = min_heap.pop().unwrap();
        let state = elem.state;
        let input_index = elem.i;
        // check whether the states' addr_key is the same as the temp_var's addr_key
        if state.0.addr != temp_var.0 {
            prev_input_index = input_index;
            // different addr_key
            // avoid duplication of adding the min and max state
            if temp_var.0 != min_state.0 && temp_var.0 != max_state.0 {
                // select the maximum version and its state value
                let value_vec = &mut temp_var.1;
                let mut max_version: u32 = 0;
                let mut state_value: StateValue = H256::default().into();
                for value in value_vec {
                    let cur_ver = value.0;
                    if cur_ver > max_version {
                        max_version = cur_ver;
                        state_value = value.1;
                    }
                }
                // write the max_version and its state_value to the state_file
                let model_key = state_writer.append((temp_var.0, max_version, state_value));
                if let Some(inner_model_key) = model_key {
                    model_constructor.append_state_key(&inner_model_key);
                }
                // insert the state's addr to the bloom filter
                if let Some(inner_filter) = &mut filter {
                    inner_filter.insert(temp_var.0);
                }

                // write the lower_cdc_tree
                let addr_key = temp_var.0;
                // prune states if is_pruned is true
                if is_pruned {
                    lower_cdc_tree.prune_tree_with_latest_version();
                }
                write_lower_cdc_tree(addr_key, &mut lower_cdc_tree, &mut lower_cdc_tree_writer, &mut upper_mht_writer);
            }
            temp_var = (state.0.addr, vec![(state.0.version, state.1)]);

            if temp_var.0 != min_state.0 && temp_var.0 != max_state.0 {
                // not the last addr_key, read states.0's cdc_tree as lower_cdc_tree
                lower_cdc_tree = read_lower_cdc_tree(input_index, &mut offset_index_counter, &mut upper_mht_readers, &mut lower_cdc_tree_readers, &mut temp_cache_manager);
            }
        } else if state.0.addr != min_state.0 && state.0.addr != max_state.0 {
            // same addr_key
            temp_var.1.push((state.0.version, state.1)); // append the states' value vec to temp_var's value vec
            if prev_input_index != input_index {
                // read the same addr_key's corresponding lower_cdc_tree in the input_index's run
                let another_lower_cdc_tree = read_lower_cdc_tree(input_index, &mut offset_index_counter, &mut upper_mht_readers, &mut lower_cdc_tree_readers, &mut temp_cache_manager);
                // merge another_lower_cdc_tree to lower_cdc_tree
                lower_cdc_tree = merge_two_cdc_trees(lower_cdc_tree, another_lower_cdc_tree).unwrap();
                prev_input_index = input_index;
            }
        }

        let i = elem.i;
        let r = input_states[i].next();
        if r.is_some() {
            // load the next smallest elem from the iterator
            let new_states = r.unwrap();
            let elem = MergeElement {
                state: new_states,
                i,
            };
            // push the element to the heap
            min_heap.push(elem);
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
    }

    // handle the last element, temp_var
    if temp_var.0 != min_state.0 && temp_var.0 != max_state.0 {
        // select the maximum version and its state value
        let value_vec = &mut temp_var.1;
        let mut max_version: u32 = 0;
        let mut state_value: StateValue = H256::default().into();
        for value in value_vec {
            let cur_ver = value.0;
            if cur_ver > max_version {
                max_version = cur_ver;
                state_value = value.1;
            }
        }
        // write the max_version and its state_value to the state_file
        let model_key = state_writer.append((temp_var.0, max_version, state_value));
        if let Some(inner_model_key) = model_key {
            model_constructor.append_state_key(&inner_model_key);
        }
        // insert the state's addr to the bloom filter
        if let Some(inner_filter) = &mut filter {
            inner_filter.insert(temp_var.0);
        }

        // write the lower_cdc_tree
        let addr_key = temp_var.0;
        // prune states if is_pruned is true
        if is_pruned {
            lower_cdc_tree.prune_tree_with_latest_version();
        }
        write_lower_cdc_tree(addr_key, &mut lower_cdc_tree, &mut lower_cdc_tree_writer, &mut upper_mht_writer);
    }

    lower_cdc_tree_writer.finalize();
    upper_mht_writer.finalize_write_mht_offset();

    // add the max state to the writer 
    state_writer.append(max_state);
    // flush the state writer
    state_writer.flush();
    // flush the model writer
    model_constructor.finalize_append();
    (state_writer.to_state_reader(), model_constructor.output_model_writer.to_model_reader(), upper_mht_writer.to_upper_mht_reader(), lower_cdc_tree_writer.to_cdc_tree_reader(), filter)
}

pub fn in_memory_collect(state_vec: Vec<(CompoundKey, StateValue)>, output_state_file_name: &str, output_model_file_name: &str, output_upper_offset_file_name: &str, output_upper_hash_file_name: &str, output_lower_cdc_file_name: &str, fanout: usize, mut filter: Option<GrowableBloom>, is_pruned: bool) -> (StatePageReader, ModelPageReader, UpperMHTReader, CDCTreeReader, Option<GrowableBloom>) {
    // create a writer for recording the states
    let mut state_writer = StatePageWriter::create(output_state_file_name);
    // create a writer for models
    let mut model_constructor = ModelConstructor::new(output_model_file_name);
    let mut upper_mht_writer = UpperMHTWriter::new(output_upper_offset_file_name, output_upper_hash_file_name, fanout);
    let mut cdc_tree_writer = CDCTreeWriter::new(output_lower_cdc_file_name);
    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = (AddrKey::new(Address(H160::from_low_u64_be(0)), StateKey(H256::from_low_u64_be(0))), 0u32, StateValue(H256::from_low_u64_be(0)));
    let max_state = (AddrKey::new(Address(H160::from_slice(&vec![255u8;20])), StateKey(H256::from_slice(&vec![255u8;32]))), u32::MAX, StateValue(H256::from_low_u64_be(0)));

    state_writer.append(min_state);

    // collect each addr and their (ver, state_value), then build CDC-Tree and upper MHT
    let mut prev_addr_key = AddrKey::default();
    let mut prev_ver_objs = Vec::<VerObject>::new();
    for state in state_vec {
        let cur_addr_key = state.0.addr;
        let ver = state.0.version;
        let value = state.1;
        if cur_addr_key != min_state.0 && cur_addr_key != max_state.0 {
            // try to remove min and max addr_key's value
            if cur_addr_key != prev_addr_key {
                // different addr_key, first handle the prev addr
                if prev_addr_key != AddrKey::default() {
                    let mut cdc_tree = CDCTree::new(DEFAULT_FANOUT, DEFAULT_GEAR_HASH_LEVEL, DEFAULT_MAX_NODE_CAPACITY);
                    // choose to only store the latest ver_obj to the state file
                    let latest_ver_obj = prev_ver_objs.last().unwrap().clone();
                    let model_key = state_writer.append((prev_addr_key, latest_ver_obj.ver, latest_ver_obj.value));
                    if let Some(inner_model_key) = model_key {
                        model_constructor.append_state_key(&inner_model_key);
                    }
                    // insert the state's addr to the bloom filter
                    if let Some(inner_filter) = &mut filter {
                        inner_filter.insert(prev_addr_key);
                    }
                    // keep all the versions in the Merkle tree
                    cdc_tree.bulk_load(prev_ver_objs);
                    // prune states if is_pruned is true
                    if is_pruned {
                        cdc_tree.prune_tree_with_latest_version();
                    }
                    // write cdc_tree
                    write_lower_cdc_tree(prev_addr_key, &mut cdc_tree, &mut cdc_tree_writer, &mut upper_mht_writer);
                }
                prev_ver_objs = vec![VerObject::new(ver, value)];
                prev_addr_key = cur_addr_key;
            } else {
                // same addr_key
                prev_ver_objs.push(VerObject::new(ver, value));
            }
        }
    }
    // handle the last addr_key case
    if !prev_ver_objs.is_empty() && prev_addr_key != AddrKey::default() {
        let mut cdc_tree = CDCTree::new(DEFAULT_FANOUT, DEFAULT_GEAR_HASH_LEVEL, DEFAULT_MAX_NODE_CAPACITY);
        // choose to only store the latest ver_obj to the state file
        let latest_ver_obj = prev_ver_objs.last().unwrap().clone();
        let model_key = state_writer.append((prev_addr_key, latest_ver_obj.ver, latest_ver_obj.value));
        if let Some(inner_model_key) = model_key {
            model_constructor.append_state_key(&inner_model_key);
        }
        // insert the state's addr to the bloom filter
        if let Some(inner_filter) = &mut filter {
            inner_filter.insert(prev_addr_key);
        }
        // keep all the versions in the Merkle tree
        cdc_tree.bulk_load(prev_ver_objs);
        // prune states if is_pruned is true
        if is_pruned {
            cdc_tree.prune_tree_with_latest_version();
        }
        // write cdc_tree
        write_lower_cdc_tree(prev_addr_key, &mut cdc_tree, &mut cdc_tree_writer, &mut upper_mht_writer);
    }
    cdc_tree_writer.finalize();
    upper_mht_writer.finalize_write_mht_offset();

    // add the max state to the writer 
    state_writer.append(max_state);
    // flush the state writer
    state_writer.flush();
    // flush the model writer
    model_constructor.finalize_append();
    (state_writer.to_state_reader(), model_constructor.output_model_writer.to_model_reader(), upper_mht_writer.to_upper_mht_reader(), cdc_tree_writer.to_cdc_tree_reader(), filter)
}

fn write_lower_cdc_tree(addr_key: AddrKey, lower_cdc_tree: &mut CDCTree, lower_cdc_tree_writer: &mut CDCTreeWriter, upper_mht_writer: &mut UpperMHTWriter) {
    let cdc_tree_root = lower_cdc_tree.get_root_hash(); // get CDC-Tree's root
    let mut bytes = Vec::<u8>::new(); // help compute the lower tree root
    bytes.extend(addr_key.to_bytes());
    bytes.extend(cdc_tree_root.as_bytes());
    let lower_tree_root = bytes_hash(&mut bytes); // lower tree root = H(addr_key | cdc_tree_root)
    let cdc_tree_file_addr = lower_cdc_tree_writer.write_tree(lower_cdc_tree);
    upper_mht_writer.append_upper_mht_offset_and_hash(cdc_tree_file_addr, lower_tree_root);
}

fn read_lower_cdc_tree(input_index: usize, offset_index_counter: &mut HashMap<usize, usize>, upper_mht_readers: &mut Vec::<(u32, UpperMHTReader)>, lower_cdc_tree_readers: &mut Vec::<CDCTreeReader>, cache_manager: &mut CacheManager) -> CDCTree {
    let offset_index = *offset_index_counter.get(&input_index).unwrap();
    let run_id = upper_mht_readers[input_index].0;
    let merkle_offset_reader = &mut upper_mht_readers[input_index].1.merkle_offset_reader;
    let cdc_tree_addr = merkle_offset_reader.read_merkle_offset(run_id, offset_index, cache_manager);
    let lower_cdc_tree = lower_cdc_tree_readers[input_index].read_tree_at(cdc_tree_addr, run_id, cache_manager).unwrap();
    // update offset_index for input_index
    *offset_index_counter.get_mut(&input_index).unwrap() += 1;
    return lower_cdc_tree;
}

#[cfg(test)]
mod tests {
    use std::{collections::BinaryHeap, slice::Iter};
    #[derive(PartialEq, Eq, Clone, Hash, Debug)]
    struct T {
        pub key: u32,
        pub v: Vec<u32>,
    }
    impl PartialOrd for T {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            return other.key.partial_cmp(&self.key);
        }
    }
    impl Ord for T {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            return other.key.cmp(&self.key);
        }
    }

    #[derive(PartialEq, Eq, Clone, Debug)]
    struct Element {
        pub states: T,
        pub i: usize,
    }

    impl PartialOrd for Element {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            return other.states.key.partial_cmp(&self.states.key);
        }
    }

    impl Ord for Element {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            return other.states.key.cmp(&self.states.key);
        }
    }

    fn merge_sort(mut inputs: Vec<Iter<T>>) -> Vec<(u32, u32)>{
        let mut r = Vec::<(u32, u32)>::new();
        let mut minheap = BinaryHeap::<Element>::new();
        let k = inputs.len();
        for i in 0..k {
            let r = inputs[i].next().unwrap();
            println!("add r: {:?}, i: {}", r, i);
            let elem = Element {
                states: r.clone(),
                i,
            };
            minheap.push(elem);
        }
        let mut full_cnt = 0;
        let mut temp_var: T = T {
            key: u32::MIN,
            v: vec![],
        };
        while full_cnt < k {
            println!("heap size: {}", minheap.len());
            let elem = minheap.pop().unwrap();
            println!("elem: {:?}", elem);
            println!("temp_var: {:?}", temp_var);
            let states = elem.states;
            if states.key != temp_var.key {
                // different key
                println!("different key");
                if temp_var.key != 0 && temp_var.key != u32::MAX {
                    println!("not boundary");
                    let value_vec = &mut temp_var.v;
                    value_vec.sort_by(|a, b| a.cmp(&b));
                    println!("value_vec");
                    for value in value_vec {
                        r.push((temp_var.key, *value));
                    }
                }
                temp_var = states;
            } else {
                // same key
                temp_var.v.extend(states.v);
            }
            let i = elem.i;
            let r = inputs[i].next();
            if r.is_some() {
                let new_states = r.unwrap().clone();
                let elem = Element {
                    states: new_states,
                    i,
                };
                minheap.push(elem);
            } else {
                full_cnt += 1;
            }
        }

        // handle the last element temp_var
        let value_vec = &mut temp_var.v;
        value_vec.sort_by(|a, b| a.cmp(&b));
        for value in value_vec {
            r.push((temp_var.key, *value));
        }
        return r;
    }

    #[test]
    fn test_simple_merge_sort() {
        /* let v1 = vec![
            Element { states: T { key: 1, v: vec![0, 2, 3]}, i: 0}, 
            Element { states: T { key: 2, v: vec![0, 1]}, i: 1},
            Element { states: T { key: 1, v: vec![5, 6]}, i: 2},
        ];
        let mut minheap = BinaryHeap::new();
        for elem in v1 {
            minheap.push(elem);
        }

        while minheap.len() > 0 {
            let r = minheap.pop().unwrap();
            println!("{:?}", r);
        } */
        let v1 = vec![
            T { key: 1, v: vec![0, 2, 3]}, 
            T { key: 4, v: vec![1]},
            T { key: 5, v: vec![1, 3, 4]},
        ];
        let v2 = vec![
            T { key: 2, v: vec![0, 1]}, 
            T { key: 4, v: vec![2, 4]},
            T { key: 6, v: vec![2]},
        ];
        let v3 = vec![
            T { key: 1, v: vec![5, 6]}, 
            T { key: 2, v: vec![4]},
            T { key: 5, v: vec![5, 7, 9, 10]},
        ];
        let v4 = vec![
            T { key: 5, v: vec![11]},
        ];
        let v5 = vec![
            T { key: 15, v: vec![12]},
            T { key: 17, v: vec![12]},
        ];
        let vecs = vec![v1, v2, v3, v4, v5];
        let inputs: Vec<_> = vecs.iter().map(|v| v.iter()).collect();
        let r = merge_sort(inputs);
        println!("{:?}", r);
    }
}