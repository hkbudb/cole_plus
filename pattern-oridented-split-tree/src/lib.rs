pub mod traits;
pub mod nodes;
use std::collections::VecDeque;

use primitive_types::H256;
use traits::POSTreeNodeIO;
use serde::{Serialize, Deserialize};
use utils::types::{bytes_hash, Digestible, LoadAddr, Num, Value};
use nodes::{POSTreeLeafNode, POSTreeInternalNode, POSTreeNode};
use cdc_hash::{CDCHash, CDCResult, DEFAULT_GEAR_HASH_LEVEL};

#[derive(Debug)]
pub struct SearchNodeId {
    pub node_id: u32, // searched node's id
    pub child_idx: Option<usize>, // the child node's position of the searched node; if the node is an internal node, then it is Some(idx), otherwise, it is None
}

// given the root_id and the search key, get the node ids of the search path
pub fn get_search_path_ids<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>, root_id: u32, key: K) -> Option<Vec<SearchNodeId>> {
    let mut cur_id = root_id;
    let curnode = tree.load_node(cur_id);
    match curnode {
        Some(node) => {
            let mut node = node;
            let mut node_ids = Vec::new();
            // iterative until the node is a leaf node
            while !node.is_leaf() {
                let internal_node = node.get_internal().unwrap();
                let index = internal_node.search_key_idx(key);
                // node id vec consists of Some(index) of the child node, and the current node id
                let search_node_id = SearchNodeId {
                    node_id: cur_id,
                    child_idx: Some(index),
                };
                node_ids.push(search_node_id);
                // update cur_id to the child node's id
                cur_id = internal_node.childs[index];
                // update the node to the child node
                node = tree.load_node(cur_id).unwrap();
            }
            let search_node_id = SearchNodeId {
                node_id: cur_id,
                child_idx: None,
            };
            node_ids.push(search_node_id);
            return Some(node_ids);
        },
        None => {
            return None; // no node exist under node_id
        }
    }
}

// given a search key, return None if the root does not exist; return Some(key_exist, leaf_node_id) otherwise
fn search_key<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>, key: K) -> Option<(bool, Vec<SearchNodeId>)> {
    // try to get the search path from the root_id
    let r = get_search_path_ids(tree, tree.get_root_id(), key);
    match r {
        Some(search_path) => {
            let leaf_node_id = search_path.last().unwrap().node_id;
            let leaf = tree.load_node(leaf_node_id).unwrap().to_leaf().unwrap();
            // search key in leaf
            let index = leaf.search_prove_idx(key);
            if leaf.key_values[index].0 == key {
                // find the key, then update the index of the last node and return true
                let mut node_ids = search_path;
                let last = node_ids.last_mut().unwrap();
                last.child_idx = Some(index);
                return Some((true, node_ids));
            }
            // does not find the key, then return false
            return Some((false, search_path));
        },
        None => {
            return None;
        }
    }
}

pub fn search_without_proof<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>, key: K) -> Option<(K, V)> {
    let mut node_id = tree.get_root_id();
    if let Some(n) = tree.load_node(node_id) {
        let mut node = n;
        let mut value = V::default();
        let mut result_key = K::default();
        // iteratively traverse the tree using key, until a leaf is reached
        while node.is_leaf() != true {
            let index = node.search_prove_idx(key);
            node_id = node.get_internal_child(index).unwrap();
            node = tree.load_node(node_id).unwrap();
        }
        // a leaf node
        if node.is_leaf() == true {
            let mut index = node.search_prove_idx(key);
            let leaf = node.get_leaf().unwrap();
            // last index is found
            if index == leaf.get_n() {
                // out of bound
                index -= 1;
            }

            result_key = leaf.key_values[index].0;
            value = leaf.key_values[index].1;
        }
        return Some((result_key, value));
    }
    else {
        return None;
    }
}

pub fn search_with_upper_key<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>, key: K) -> Option<(K, V)> {
    let mut node_id = tree.get_root_id();
    if let Some(n) = tree.load_node(node_id) {
        let mut node = n;
        let mut value = V::default();
        let mut result_key = K::default();
        // iteratively traverse the tree using key, until a leaf is reached
        while node.is_leaf() != true {
            let index = node.search_prove_idx(key);
            node_id = node.get_internal_child(index).unwrap();
            node = tree.load_node(node_id).unwrap();
        }
        // a leaf node
        if node.is_leaf() == true {
            let mut index = node.search_prove_idx(key); // might read the previoud node to check largest value
            let leaf = node.get_leaf().unwrap();
            // last index is found
            if index == leaf.get_n() {
                // out of bound
                index -= 1;
            }
            result_key = leaf.key_values[index].0;
            value = leaf.key_values[index].1;
            if index == 0 && result_key.addr().unwrap() != key.addr().unwrap() {
                // check the previous leaf's last element
                let prev_leaf_id = node.get_leaf().unwrap().prev;
                if prev_leaf_id != 0 {
                    let prev_leaf = tree.load_node(prev_leaf_id).unwrap();
                    let (last_key, last_value) = *prev_leaf.get_leaf().unwrap().key_values.last().unwrap();
                    if last_key.addr().unwrap() == key.addr().unwrap() {
                        result_key = last_key;
                        value = last_value;
                    }
                }
            }
        }
        return Some((result_key, value));
    }
    else {
        return None;
    }
}

/* Proof of a range query, each level consist of a vector of MB-Tree nodes
 */
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct POSTreeRangeProof<K: Num + LoadAddr, V: Value> {
    pub levels: Vec<Vec<((usize, usize), POSTreeNode<K, V>)>>, // the first two usize store the start_idx and end_idx of the searched entries in the node
}

pub fn get_range_proof<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>, lb: K, ub: K) -> (Option<Vec<(K, V)>>, POSTreeRangeProof<K, V>) {
    // init a range proof
    let mut proof = POSTreeRangeProof::default();
    if tree.get_key_num() == 0 {
        // tree is empty
        return (None, proof);
    } else {
        // load root node
        let node = tree.load_node(tree.get_root_id()).unwrap();
        // init a result value vector
        let mut value_vec = Vec::<(K, V)>::new();
        // create a queue to help traverse the tree
        let mut queue = VecDeque::<POSTreeNode<K, V>>::new();
        // push the root node to the queue
        queue.push_back(node);
        // some counter to help determine the number of nodes in the level
        let mut prev_cnt = 1;
        let mut cur_cnt = 0;
        // a temporary proof for the current level
        let mut cur_level_proof = Vec::<((usize, usize), POSTreeNode<K, V>)>::new();
        // traverse the tree in a while loop until the queue is empty
        while !queue.is_empty() {
            let cur_node = queue.pop_front().unwrap();
            prev_cnt -= 1; // decrease the node counter of the previous level
            if !cur_node.is_leaf() {
                // the node is an internal node, retrieve the reference of the internal node
                let internal = cur_node.get_internal().unwrap();
                // given the lb and ub, get the position range of the child nodes
                let (start_idx, end_idx) = cur_node.search_prove_idx_range(lb, ub);
                // update the node counter for the level
                cur_cnt += end_idx - start_idx + 1;
                // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                // add the corresponding child nodes to the queue
                for idx in start_idx ..=end_idx {
                    let id = internal.childs[idx];
                    let child_node = tree.load_node(id).unwrap();
                    queue.push_back(child_node);
                }
            } else {
                // the node is a leaf node, retrieve the reference of the leaf node
                let leaf = cur_node.get_leaf().unwrap();
                // get the position range of the leaf node
                let (start_idx, end_idx) = cur_node.search_prove_idx_range(lb, ub);
                // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                // add the corresponding searched entries to the value_vec
                for id in start_idx ..= end_idx {
                    let key_value = leaf.key_values[id].clone();
                    value_vec.push(key_value);
                }
            }

            if prev_cnt == 0 {
                // if prev_cnt = 0, start a new level by assigning cur_cnt to prev_cnt and reset cur_cnt to 0
                prev_cnt = cur_cnt;
                cur_cnt = 0;
                // add the temporary proof of the current level to the final proof
                proof.levels.push(cur_level_proof.clone());
                cur_level_proof.clear();
            }
        }
        return (Some(value_vec), proof);
    }
}

// reconstruct the range proof to the root digest 
pub fn reconstruct_range_proof<K: Num + LoadAddr, V: Value>(lb: K, ub: K, result: &Option<Vec<(K, V)>>, proof: &POSTreeRangeProof<K, V>) -> H256 {
    if result.is_none() && proof == &POSTreeRangeProof::default() {
        return H256::default();
    } else {
        // a flag to determine whethere there is an verification error
        let mut validate = true;
        // the root hash from the proof should be the first level's single node's digest
        let compute_root_hash = proof.levels[0][0].1.to_digest();
        // a temporary vector to store the hash values of the next level
        let mut next_level_hashes= Vec::<H256>::new();
        // iterate each of the levels in the proof
        for (i, level_proof) in proof.levels.iter().enumerate() {
            // check whether the hash valeus in the next_level_hashes vector (constructed during the prevous level) match the re-computed one for the current level or not
            if i != 0 {
                let mut computed_hashes = Vec::<H256>::new();
                for (_, cur_level_node) in level_proof {
                    computed_hashes.push(cur_level_node.to_digest());
                }
                if computed_hashes != next_level_hashes {
                    // not match
                    validate = false;
                    break;
                }
                // start another level by clearing the hashes
                next_level_hashes.clear();
            }
            // id of the result in the vector of the proof
            let mut leaf_id: usize = 0;
            for inner_proof in level_proof {
                // retrieve the node reference from the level proof
                let node = &inner_proof.1;
                // retrieve the start and end positions from the level proof
                let (start_idx, end_idx) = &inner_proof.0;
                if !node.is_leaf() {
                    // node is an internal node
                    let internal = node.get_internal().unwrap();
                    // add the hash values of the traversed child nodes to the next_level_hashes
                    for id in *start_idx..= *end_idx {
                        let h = internal.child_hashes[id];
                        next_level_hashes.push(h);
                    }
                } else {
                    // node is a leaf node
                    let result = result.as_ref().unwrap();
                    let leaf = node.get_leaf().unwrap();
                    // check the values in the result vector against the values in the proof
                    for (i, id) in (*start_idx..= *end_idx).into_iter().enumerate() {
                        let key_value = leaf.key_values[id];
                        if i == 0 {
                            if result[leaf_id] != key_value {
                                validate = false;
                                break;
                            }
                        } else {
                            let k = key_value.0;
                            if k < lb || k > ub || result[leaf_id] != key_value {
                                validate = false;
                                break;
                            }
                        }
                        leaf_id += 1;
                    }
                }
            }
        }
        if validate == false {
            return H256::default();
        }
        return compute_root_hash;
    }
}

fn check_leaf_node_cdc_and_fanout<K: Num, V: Value>(list: &Vec<(K, V)>, exp_fanout: usize, max_fanout: usize) -> Vec<i32> {
    let mut cut_points: Vec<i32> = vec![-1];
    let mut cdc_hash = CDCHash::new(exp_fanout, DEFAULT_GEAR_HASH_LEVEL, max_fanout);
    let list_len = list.len();
    for cur_index in 0..list_len {
        let mut input_bytes = bincode::serialize(&list[cur_index].0).unwrap();
        input_bytes.extend(bincode::serialize(&list[cur_index].1).unwrap());
        let r = cdc_hash.generate_cut_point(bytes_hash(&input_bytes).as_bytes());
        if let CDCResult::PatternFound | CDCResult::ReachCapacity = r {
            if cur_index != 0 && cur_index != list_len - 1 {
                cut_points.push(cur_index as i32);
            }
        }
    }
    
    cut_points.push(list.len() as i32 -1); // add last split point len - 1
    return cut_points;
}

fn check_internal_node_cdc_and_fanout(list: &Vec<H256>, exp_fanout: usize, max_fanout: usize) -> Vec<i32> {
    let mut cut_points: Vec<i32> = vec![-1];
    let mut cdc_hash = CDCHash::new(exp_fanout, DEFAULT_GEAR_HASH_LEVEL, max_fanout);
    let list_len = list.len();
    for index in 0..list_len {
        let r = cdc_hash.generate_cut_point(list[index].as_bytes());
        if let CDCResult::PatternFound | CDCResult::ReachCapacity = r {
            if index != 0 && index != list_len - 1 {
                cut_points.push(index as i32);
            }
        }
    }
    cut_points.push(list.len() as i32 -1); // add last split point len - 1
    return cut_points;
}

#[derive(Debug, Default)]
struct PushUpEntries<K: Num> {
    keys: Vec::<K>,
    node_hashes: Vec::<H256>,
    node_ids: Vec::<u32>,
}

impl<K: Num> PushUpEntries<K> {
    pub fn new(keys: Vec::<K>, node_hashes: Vec::<H256>, node_ids: Vec::<u32>) -> Self {
        Self {
            keys,
            node_hashes,
            node_ids,
        }
    }

    pub fn entry_num(&self, ) -> usize {
        assert_eq!(self.keys.len(), self.node_hashes.len());
        assert_eq!(self.keys.len(), self.node_ids.len());
        self.keys.len()
    }

    pub fn first_node_id(&self, ) -> u32 {
        *self.node_ids.first().unwrap()
    }
}

pub fn insert<K: Num + LoadAddr, V: Value>(tree: &mut impl POSTreeNodeIO<K, V>, key: K, value: V) {
    // println!("insert key: {:?}, value: {:?}", key, value);
    // add 1 to the number of data
    tree.increment_key_num();
    // get exp and max fanout
    let exp_fanout = tree.get_exp_fanout();
    let max_fanout = tree.get_max_fanout();
    match search_key(tree, key) {
        Some((key_exist, node_id_vec)) => {
            let mut node_id_vec = node_id_vec;
            let leaf_id_obj = node_id_vec.pop().unwrap();
            let leaf_id = leaf_id_obj.node_id;
            let mut leaf = tree.load_node(leaf_id).unwrap().to_leaf().unwrap();
            let index;
            if !key_exist {
                // get the insert index
                index = leaf.search_insert_idx(key);
                // insert (key, value) to the leaf
                leaf.key_values.insert(index, (key, value));
            } else {
                // get updated index
                index = leaf_id_obj.child_idx.unwrap();
                // update the (k, v)
                leaf.key_values[index] = (key, value);
            }
            let leaf_n = leaf.key_values.len();
            let collection = &mut leaf.key_values;
            let mut recorded_next_node = leaf.next;
            let recorded_prev_node = leaf.prev;
            let mut collect_num_nodes: usize = 1;
            let mut pointers = vec![recorded_prev_node, leaf_id];
            if index == leaf_n - 1 || leaf_n > max_fanout {
                // index indicates the last element, or leaf reach the maximum capacity
                // should load the next nodes until the node has less maximum fanout's elements
                let mut less_flag = false;
                let mut next_node_id = leaf.next;
                while next_node_id != 0 && !less_flag {
                    let next_leaf_node = tree.remove_node(next_node_id).unwrap().to_leaf().unwrap();
                    if next_leaf_node.get_n() < max_fanout {
                        // next leaf node has less than maximum fanout's element
                        less_flag = true;
                    }
                    recorded_next_node = next_leaf_node.next;
                    // collect (k, v) of next_leaf_node
                    collection.extend(next_leaf_node.key_values);
                    next_node_id = next_leaf_node.next;
                    collect_num_nodes += 1; 
                }
            }
            // check cdc in collection
            let cut_points = check_leaf_node_cdc_and_fanout(&collection, exp_fanout, max_fanout);
            let cut_points_len = cut_points.len();
            for _ in 1 .. cut_points_len - 1 {
                let new_id = tree.new_counter();
                pointers.push(new_id);
            }
            pointers.push(recorded_next_node);
            let mut pushed_last_keys = Vec::<K>::new();
            let mut pushed_node_ids = Vec::<u32>::new();
            let mut pushed_node_hashes = Vec::<H256>::new();
            for i in 1..=cut_points_len - 1 {
                let start_idx = (cut_points[i-1] + 1) as usize;
                let end_idx = cut_points[i] as usize;
                let mut new_node = POSTreeLeafNode::new(collection[start_idx..=end_idx].to_vec());
                let new_node_id = pointers[i];
                new_node.next = pointers[i+1];
                new_node.prev = pointers[i-1];
                pushed_last_keys.push(new_node.key_values.last().unwrap().0);
                pushed_node_ids.push(new_node_id);
                pushed_node_hashes.push(new_node.to_digest());
                tree.store_node(new_node_id, POSTreeNode::from_leaf(new_node));
            }
            let last_pointer = pointers[pointers.len() - 1];
            if last_pointer != 0 {
                tree.update_node_prev(last_pointer, pointers[pointers.len() - 2]);
                // let mut last_node = tree.load_node(last_pointer).unwrap().to_leaf().unwrap();
                // last_node.prev = pointers[pointers.len() - 2];
                // tree.store_node(last_pointer, POSTreeNode::from_leaf(last_node));
            }
            let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
            handle_insert_parent_nodes(tree, entries, node_id_vec, collect_num_nodes);
        },
        None => {
            // create a new node_id
            let node_id = tree.new_counter();
            // create a key_value pair vector using the (key, value)
            let key_values = vec![(key, value)];
            // create a new leaf node using the vector
            let leaf = POSTreeLeafNode::new(key_values);
            let node = POSTreeNode::from_leaf(leaf);
            // store the node to the storage
            tree.store_node(node_id, node);
            // update the root id
            tree.set_root_id(node_id);
        }
    }
}

fn handle_insert_parent_nodes<K: Num + LoadAddr, V: Value>(tree: &mut impl POSTreeNodeIO<K, V>, entries: PushUpEntries<K>, node_id_vec: Vec<SearchNodeId>, collect_num_nodes: usize) {
    // get exp and max fanout
    let exp_fanout = tree.get_exp_fanout();
    let max_fanout = tree.get_max_fanout();
    if node_id_vec.is_empty() {
        if entries.entry_num() >= 2 {
            // should check the pushed-up entries have pattern or not
            let key_collection = entries.keys;
            let child_id_collection = entries.node_ids;
            let child_hash_collection = entries.node_hashes;
            // check pattern in collection
            let cut_points = check_internal_node_cdc_and_fanout(&child_hash_collection, exp_fanout, max_fanout);
            let mut pointers = vec![0];
            let cut_points_len = cut_points.len();
            for _ in 0..cut_points_len - 1 {
                let new_id = tree.new_counter();
                pointers.push(new_id);
            }
            let recorded_next_node = 0;
            pointers.push(recorded_next_node);
            let mut pushed_last_keys = Vec::<K>::new();
            let mut pushed_node_ids = Vec::<u32>::new();
            let mut pushed_node_hashes = Vec::<H256>::new();
            for i in 1..=cut_points_len - 1 {
                let start_idx = (cut_points[i-1] + 1) as usize;
                let end_idx = cut_points[i] as usize;
                let mut new_node = POSTreeInternalNode::new(key_collection[start_idx..=end_idx].to_vec(), child_id_collection[start_idx..=end_idx].to_vec(), child_hash_collection[start_idx..=end_idx].to_vec());
                new_node.next = pointers[i+1];
                new_node.prev = pointers[i-1];
                let new_node_id = pointers[i];
                pushed_last_keys.push(*new_node.keys.last().unwrap());
                pushed_node_ids.push(new_node_id);
                pushed_node_hashes.push(new_node.to_digest());
                tree.store_node(new_node_id, POSTreeNode::from_internal(new_node));
            }
            let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
            handle_insert_parent_nodes(tree, entries, node_id_vec, 1);
        } else if entries.entry_num() == 1 {
            // reach the root level, update the root id
            let root_id = entries.first_node_id();
            tree.set_root_id(root_id);
            // check if the root is an internal node and contains at least two entries
            let root = tree.load_node(root_id).unwrap();
            if !root.is_leaf() {
                let root_internal = root.to_internal().unwrap();
                if root_internal.get_n() == 1 {
                    // set its first child as the root and remove the node with root_id
                    tree.set_root_id(root_internal.childs[0]);
                    tree.remove_node(root_id).unwrap();
                }
            }
        }
    } else {
        let mut node_id_vec = node_id_vec;
        let node_id_obj = node_id_vec.pop().unwrap();
        let node_id = node_id_obj.node_id;
        let node = tree.load_node(node_id).unwrap().to_internal().unwrap();
        let index = node_id_obj.child_idx.unwrap();
        let mut key_collection = node.keys;
        let mut child_id_collection = node.childs;
        let mut child_hash_collection = node.child_hashes;
        let mut recorded_next_node = node.next;
        let recorded_prev_node = node.prev;
        let mut new_collect_num_nodes: usize = 1;
        let mut pointers = vec![recorded_prev_node, node_id];
        let mut next_node_id = node.next;
        let mut less_flag = false;
        while index + collect_num_nodes >= key_collection.len() && next_node_id != 0 {
            // current node is not enough, should incorporate next node
            let node_next = tree.remove_node(next_node_id).unwrap().to_internal().unwrap();
            if node_next.get_n() < max_fanout {
                less_flag = true;
            } else {
                less_flag = false;
            }
            recorded_next_node = node_next.next;
            key_collection.extend(node_next.keys);
            child_id_collection.extend(node_next.childs);
            child_hash_collection.extend(node_next.child_hashes);
            next_node_id = node_next.next;
            new_collect_num_nodes += 1; 
        }
        while next_node_id != 0 && !less_flag {
            // previous extended node ends with maximum size
            let node_next = tree.remove_node(next_node_id).unwrap().to_internal().unwrap();
            if node_next.get_n() < max_fanout {
                less_flag = true;
            }
            recorded_next_node = node_next.next;
            key_collection.extend(node_next.keys);
            child_id_collection.extend(node_next.childs);
            child_hash_collection.extend(node_next.child_hashes);
            next_node_id = node_next.next;
            new_collect_num_nodes += 1; 
        }
        // update collections
        for _ in 0..collect_num_nodes {
            key_collection.remove(index);
            child_id_collection.remove(index);
            child_hash_collection.remove(index);
        }
        // insert some elements to index
        let mut inserted_index = index;
        for ((insert_key, insert_node_id), insert_node_h) in entries.keys.into_iter().zip(entries.node_ids.into_iter()).zip(entries.node_hashes) {
            key_collection.insert(inserted_index, insert_key);
            child_id_collection.insert(inserted_index, insert_node_id);
            child_hash_collection.insert(inserted_index, insert_node_h);
            inserted_index += 1;
        }
        // check pattern in collection
        let cut_points = check_internal_node_cdc_and_fanout(&child_hash_collection, exp_fanout, max_fanout);
        let cut_points_len = cut_points.len();
        for _ in 1 .. cut_points_len - 1 {
            let new_id = tree.new_counter();
            pointers.push(new_id);
        }
        pointers.push(recorded_next_node);
        let mut pushed_last_keys = Vec::<K>::new();
        let mut pushed_node_ids = Vec::<u32>::new();
        let mut pushed_node_hashes = Vec::<H256>::new();
        for i in 1..=cut_points_len - 1 {
            let start_idx = (cut_points[i-1] + 1) as usize;
            let end_idx = cut_points[i] as usize;
            let mut new_node = POSTreeInternalNode::new(key_collection[start_idx..=end_idx].to_vec(), child_id_collection[start_idx..=end_idx].to_vec(), child_hash_collection[start_idx..=end_idx].to_vec());
            new_node.next = pointers[i+1];
            let new_node_id = pointers[i];
            new_node.prev = pointers[i-1];
            pushed_last_keys.push(*new_node.keys.last().unwrap());
            pushed_node_ids.push(new_node_id);
            pushed_node_hashes.push(new_node.to_digest());
            tree.store_node(new_node_id, POSTreeNode::from_internal(new_node));
        }
        // update last node's prev pointer
        let pointers_len = pointers.len();
        if pointers[pointers_len-1] != 0 {
            tree.update_node_prev(pointers[pointers_len-1], pointers[pointers_len-2]);
        }
        let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
        handle_insert_parent_nodes(tree, entries, node_id_vec, new_collect_num_nodes);
    }
}

pub fn remove<K: Num + LoadAddr, V: Value>(tree: &mut impl POSTreeNodeIO<K, V>, key: K) {
    // get exp and max fanout
    let exp_fanout = tree.get_exp_fanout();
    let max_fanout = tree.get_max_fanout();
    match search_key(tree, key) {
        Some((key_exist, node_id_vec)) => {
            if key_exist {
                // deduct 1 to the number of data
                tree.deduct_key_num();
                let mut node_id_vec = node_id_vec;
                let leaf_id_obj = node_id_vec.pop().unwrap();
                let leaf_id = leaf_id_obj.node_id;
                let mut leaf = tree.load_node(leaf_id).unwrap().to_leaf().unwrap();
                let leaf_n = leaf.key_values.len();
                let index = leaf_id_obj.child_idx.unwrap();
                leaf.key_values.remove(index); // remove the entry from the node
                let collection = &mut leaf.key_values;
                let mut recorded_next_node = leaf.next;
                let recorded_prev_node = leaf.prev;
                let mut collect_num_nodes: usize = 1;
                let mut pointers = vec![recorded_prev_node, leaf_id];
                if index == leaf_n - 1 || leaf_n == max_fanout {
                    // index indicates the last element, or leaf reach the maximum capacity
                    // should load the next nodes until the node has less maximum fanout's elements
                    let mut less_flag = false;
                    let mut next_node_id = leaf.next;
                    while next_node_id != 0 && !less_flag {
                        let next_leaf_node = tree.remove_node(next_node_id).unwrap().to_leaf().unwrap();
                        if next_leaf_node.get_n() < max_fanout {
                            // next leaf node has less than maximum fanout's element
                            less_flag = true;
                        }
                        recorded_next_node = next_leaf_node.next;
                        // collect (k, v) of next_leaf_node
                        collection.extend(next_leaf_node.key_values);
                        next_node_id = next_leaf_node.next;
                        collect_num_nodes += 1; 
                    }
                }

                if collection.len() == 0 {
                    // remove the node (the node should be the right-most one)
                    tree.remove_node(leaf_id).unwrap();
                    // update previous node's next pointer as null
                    tree.update_node_next(recorded_prev_node, 0);
                    let entries = PushUpEntries::default();
                    handle_delete_parent_nodes(tree, entries, node_id_vec, 0);
                } else {
                    // check cdc in collection
                    let cut_points = check_leaf_node_cdc_and_fanout(&collection, exp_fanout, max_fanout);
                    let cut_points_len = cut_points.len();
                    for _ in 1 .. cut_points_len - 1 {
                        let new_id = tree.new_counter();
                        pointers.push(new_id);
                    }
                    pointers.push(recorded_next_node);
                    let mut pushed_last_keys = Vec::<K>::new();
                    let mut pushed_node_ids = Vec::<u32>::new();
                    let mut pushed_node_hashes = Vec::<H256>::new();
                    for i in 1..=cut_points_len - 1 {
                        let start_idx = (cut_points[i-1] + 1) as usize;
                        let end_idx = cut_points[i] as usize;
                        let mut new_node = POSTreeLeafNode::new(collection[start_idx..=end_idx].to_vec());
                        let new_node_id = pointers[i];
                        new_node.next = pointers[i+1];
                        new_node.prev = pointers[i-1];
                        pushed_last_keys.push(new_node.key_values.last().unwrap().0);
                        pushed_node_ids.push(new_node_id);
                        pushed_node_hashes.push(new_node.to_digest());
                        tree.store_node(new_node_id, POSTreeNode::from_leaf(new_node));
                    }
                    let last_pointer = pointers[pointers.len() - 1];
                    if last_pointer != 0 {
                        tree.update_node_prev(last_pointer, pointers[pointers.len() - 2]);
                        // let mut last_node = tree.load_node(last_pointer).unwrap().to_leaf().unwrap();
                        // last_node.prev = pointers[pointers.len() - 2];
                        // tree.store_node(last_pointer, POSTreeNode::from_leaf(last_node));
                    }
                    let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
                    handle_delete_parent_nodes(tree, entries, node_id_vec, collect_num_nodes);
                }

                if tree.get_key_num() == 0 {
                    // empty tree
                    tree.set_counter(0);
                    tree.set_root_id(0);
                }
            }
        },
        None => {}
    }
}

fn handle_delete_parent_nodes<K: Num + LoadAddr, V: Value>(tree: &mut impl POSTreeNodeIO<K, V>, entries: PushUpEntries<K>, node_id_vec: Vec<SearchNodeId>, collect_num_nodes: usize) {
    // get exp and max fanout
    let exp_fanout = tree.get_exp_fanout();
    let max_fanout = tree.get_max_fanout();
    if node_id_vec.is_empty() {
        if entries.entry_num() >= 2 {
            // should check the pushed-up entries have pattern or not
            let key_collection = entries.keys;
            let child_id_collection = entries.node_ids;
            let child_hash_collection = entries.node_hashes;
            // check pattern in collection
            let cut_points = check_internal_node_cdc_and_fanout(&child_hash_collection, exp_fanout, max_fanout);
            let mut pointers = vec![0];
            let cut_points_len = cut_points.len();
            for _ in 0..cut_points_len - 1 {
                let new_id = tree.new_counter();
                pointers.push(new_id);
            }
            let recorded_next_node = 0;
            pointers.push(recorded_next_node);
            let mut pushed_last_keys = Vec::<K>::new();
            let mut pushed_node_ids = Vec::<u32>::new();
            let mut pushed_node_hashes = Vec::<H256>::new();
            for i in 1..=cut_points_len - 1 {
                let start_idx = (cut_points[i-1] + 1) as usize;
                let end_idx = cut_points[i] as usize;
                let mut new_node = POSTreeInternalNode::new(key_collection[start_idx..=end_idx].to_vec(), child_id_collection[start_idx..=end_idx].to_vec(), child_hash_collection[start_idx..=end_idx].to_vec());
                new_node.next = pointers[i+1];
                new_node.prev = pointers[i-1];
                let new_node_id = pointers[i];
                pushed_last_keys.push(*new_node.keys.last().unwrap());
                pushed_node_ids.push(new_node_id);
                pushed_node_hashes.push(new_node.to_digest());
                tree.store_node(new_node_id, POSTreeNode::from_internal(new_node));
            }
            let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
            handle_delete_parent_nodes(tree, entries, node_id_vec, 1);
        } else if entries.entry_num() == 1 {
            // reach the root level, update the root id
            let root_id = entries.first_node_id();
            tree.set_root_id(root_id);
            // check if the root is an internal node and contains at least two entries
            let root = tree.load_node(root_id).unwrap();
            if !root.is_leaf() {
                let root_internal = root.to_internal().unwrap();
                if root_internal.get_n() == 1 {
                    // set its first child as the root and remove the node with root_id
                    tree.set_root_id(root_internal.childs[0]);
                    tree.remove_node(root_id).unwrap();
                }
            }
        }
    } else {
        let mut node_id_vec = node_id_vec;
        let node_id_obj = node_id_vec.pop().unwrap();
        let node_id = node_id_obj.node_id;
        let mut node = tree.load_node(node_id).unwrap().to_internal().unwrap();
        let index = node_id_obj.child_idx.unwrap();
        if collect_num_nodes == 0 {
            // remove index from node
            node.keys.remove(index);
            node.childs.remove(index);
            node.child_hashes.remove(index);
            if node.get_n() == 0 {
                // empty node, should remove the node
                tree.remove_node(node_id).unwrap();
                // update the previous node's next pointer
                let recorded_prev_node = node.prev;
                tree.update_node_next(recorded_prev_node, 0);
                
                let entries = PushUpEntries::default();
                handle_delete_parent_nodes(tree, entries, node_id_vec, 0);
            } else {
                // update the node to the tree
                let mut pushed_last_keys = Vec::<K>::new();
                let mut pushed_node_ids = Vec::<u32>::new();
                let mut pushed_node_hashes = Vec::<H256>::new();
                pushed_last_keys.push(*node.keys.last().unwrap());
                pushed_node_ids.push(node_id);
                pushed_node_hashes.push(node.to_digest());
                tree.store_node(node_id, POSTreeNode::from_internal(node));
                let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
                handle_delete_parent_nodes(tree, entries, node_id_vec, 1);
            }
        } else {
            let mut key_collection = node.keys;
            let mut child_id_collection = node.childs;
            let mut child_hash_collection = node.child_hashes;
            let mut recorded_next_node = node.next;
            let recorded_prev_node = node.prev;
            let mut new_collect_num_nodes: usize = 1;
            let mut pointers = vec![recorded_prev_node, node_id];
            let mut next_node_id = node.next;
            let mut less_flag = false;
            while index + collect_num_nodes >= key_collection.len() && next_node_id != 0 {
                // current node is not enough, should incorporate next node
                let node_next = tree.remove_node(next_node_id).unwrap().to_internal().unwrap();
                if node_next.get_n() < max_fanout {
                    less_flag = true;
                } else {
                    less_flag = false;
                }
                recorded_next_node = node_next.next;
                key_collection.extend(node_next.keys);
                child_id_collection.extend(node_next.childs);
                child_hash_collection.extend(node_next.child_hashes);
                next_node_id = node_next.next;
                new_collect_num_nodes += 1; 
            }
            while next_node_id != 0 && !less_flag {
                // previous extended node ends with maximum size
                let node_next = tree.remove_node(next_node_id).unwrap().to_internal().unwrap();
                if node_next.get_n() < max_fanout {
                    less_flag = true;
                }
                recorded_next_node = node_next.next;
                key_collection.extend(node_next.keys);
                child_id_collection.extend(node_next.childs);
                child_hash_collection.extend(node_next.child_hashes);
                next_node_id = node_next.next;
                new_collect_num_nodes += 1; 
            }
            // update collections
            for _ in 0..collect_num_nodes {
                key_collection.remove(index);
                child_id_collection.remove(index);
                child_hash_collection.remove(index);
            }
            // insert some elements to index
            let mut inserted_index = index;
            for ((insert_key, insert_node_id), insert_node_h) in entries.keys.into_iter().zip(entries.node_ids.into_iter()).zip(entries.node_hashes) {
                key_collection.insert(inserted_index, insert_key);
                child_id_collection.insert(inserted_index, insert_node_id);
                child_hash_collection.insert(inserted_index, insert_node_h);
                inserted_index += 1;
            }
            // check pattern in collection
            let cut_points = check_internal_node_cdc_and_fanout(&child_hash_collection, exp_fanout, max_fanout);
            let cut_points_len = cut_points.len();
            for _ in 1 .. cut_points_len - 1 {
                let new_id = tree.new_counter();
                pointers.push(new_id);
            }
            pointers.push(recorded_next_node);
            let mut pushed_last_keys = Vec::<K>::new();
            let mut pushed_node_ids = Vec::<u32>::new();
            let mut pushed_node_hashes = Vec::<H256>::new();
            for i in 1..=cut_points_len - 1 {
                let start_idx = (cut_points[i-1] + 1) as usize;
                let end_idx = cut_points[i] as usize;
                let mut new_node = POSTreeInternalNode::new(key_collection[start_idx..=end_idx].to_vec(), child_id_collection[start_idx..=end_idx].to_vec(), child_hash_collection[start_idx..=end_idx].to_vec());
                new_node.next = pointers[i+1];
                let new_node_id = pointers[i];
                new_node.prev = pointers[i-1];
                pushed_last_keys.push(*new_node.keys.last().unwrap());
                pushed_node_ids.push(new_node_id);
                pushed_node_hashes.push(new_node.to_digest());
                tree.store_node(new_node_id, POSTreeNode::from_internal(new_node));
            }
            // update last node's prev pointer
            let pointers_len = pointers.len();
            if pointers[pointers_len-1] != 0 {
                tree.update_node_prev(pointers[pointers_len-1], pointers[pointers_len-2]);
            }
            let entries = PushUpEntries::new(pushed_last_keys, pushed_node_hashes, pushed_node_ids);
            handle_insert_parent_nodes(tree, entries, node_id_vec, new_collect_num_nodes);
        }
    }
}

// get the left mode leaf's id for traversing all the leaf nodes to collect the data in batch
pub fn get_left_most_leaf_id<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> u32 {
    // get root id
    let mut node_id = tree.get_root_id();
    // retrieve root node
    let mut node = tree.load_node(node_id).unwrap();
    // iteratively traverse the tree along the left-most sub-path
    while node.is_leaf() == false {
        let internal = node.get_internal().unwrap();
        // get the left-most child node id
        node_id = *internal.childs.first().unwrap();
        node = tree.load_node(node_id).unwrap();
    }
    return node_id;
}

pub fn get_left_path<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> Vec<u32> {
    // get root id
    let mut node_id = tree.get_root_id();
    // retrieve root node
    let mut node = tree.load_node(node_id).unwrap();
    // iteratively traverse the tree along the left-most sub-path
    let mut res: Vec<u32> = Vec::new();
    res.push(node_id);
    while node.is_leaf() == false {
        let internal = node.get_internal().unwrap();
        // get the left-most child node id
        node_id = *internal.childs.first().unwrap();
        node = tree.load_node(node_id).unwrap();
        res.push(node_id);
    }
    return res;
}

pub fn get_tree_height<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> usize {
    let mut height = 0;
    // get root id
    let mut node_id = tree.get_root_id();
    height += 1;
    // retrieve root node
    let mut node = tree.load_node(node_id).unwrap();
    // iteratively traverse the tree along the left-most sub-path
    while node.is_leaf() == false {
        let internal = node.get_internal().unwrap();
        // get the left-most child node id
        node_id = *internal.childs.first().unwrap();
        node = tree.load_node(node_id).unwrap();
        height += 1;
    }
    return height;
}

pub fn check_prev_next_pointers<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> bool {
    let mut flag = true;
    let mut leaf_id = get_left_most_leaf_id(tree);
    let mut cur_prev = 0;
    while leaf_id != 0 {
        let leaf = tree.load_node(leaf_id).unwrap().to_leaf().unwrap();
        if leaf.prev != cur_prev {
            flag = false;
            return flag;
        }
        cur_prev = leaf_id;
        leaf_id = leaf.next;
    }
    return flag;
}

pub fn check_pointers<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> bool {
    let mut flag = true;
    let node_ids = get_left_path(tree);
    for node_id in node_ids {
        let mut cur_node_id = node_id;
        let mut cur_prev = 0;
        while cur_node_id != 0 {
            let node = tree.load_node(cur_node_id).unwrap();
            if node.is_leaf() {
                let leaf = node.to_leaf().unwrap();
                if leaf.prev != cur_prev {
                    println!("leaf prev: {}, cur prev: {}", leaf.prev, cur_prev);
                    flag = false;
                    return flag;
                }
                cur_prev = cur_node_id;
                cur_node_id = leaf.next;
            } else {
                let internal = node.to_internal().unwrap();
                if internal.prev != cur_prev {
                    println!("internal prev: {}, cur prev: {}", internal.prev, cur_prev);
                    flag = false;
                    return flag;
                }
                cur_prev = cur_node_id;
                cur_node_id = internal.next;
            }
        }
    }
    return flag;
}

pub fn print_tree<K: Num + LoadAddr, V: Value>(tree: &impl POSTreeNodeIO<K, V>) -> usize {
    let root_id = tree.get_root_id();
    let root_node = tree.load_node(root_id).unwrap();
    let mut queue = VecDeque::new();
    queue.push_back((root_id, root_node));
    let mut node_size = 0;
    let mut cnt = 0;
    while !queue.is_empty() {
        let (node_id, node) = queue.pop_front().unwrap();
        node_size += node.get_n();
        cnt += 1;
        // println!("h: {:?}, id: {}, is leaf: {}, node size: {:?}", node.to_digest(), node_id, node.is_leaf(), node.get_n());
        println!("h: {:?}, id: {}, is leaf: {}, node size: {:?} node: {:?}", node.to_digest(), node_id, node.is_leaf(), node.get_n(), node);
        match node {
            POSTreeNode::Leaf(_) => {},
            POSTreeNode::NonLeaf(internal) => {
                for child_id in &internal.childs {
                    let child_node = tree.load_node(*child_id).unwrap();
                    queue.push_back((*child_id, child_node));
                }
            }
        }
    }
    return node_size / cnt;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use rand::prelude::*;
    use utils::types::bytes_hash;
    use super::*;
    use serde::{Serialize, Deserialize};
    #[test]
    fn test_split() {
        let mut v = vec![1, 2, 3, 4, 5];
        let v_2 = v.split_off(0);
        println!("v: {:?}, v_2: {:?}", v, v_2);
    }

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Serialize, Deserialize, Debug)]
    pub struct IKey(u32);
    impl Digestible for IKey {
        fn to_digest(&self) -> H256 {
            H256::from_low_u64_be(self.0 as u64)
        }
    }

    impl LoadAddr for IKey {
        fn addr(&self) -> Option<&utils::types::AddrKey> {
            None
        }
    }

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Serialize, Deserialize, Debug)]
    pub struct IValue(u32);
    impl Digestible for IValue {
        fn to_digest(&self) -> H256 {
            let v = H256::from_low_u64_be(self.0 as u64);
            bytes_hash(v.as_bytes())
        }
    }
    #[derive(Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
    pub struct TestPOSTree {
        pub root: u32,
        pub counter: u32,
        pub key_num: u32,
        pub nodes: BTreeMap<u32, POSTreeNode<IKey, IValue>>,
        pub exp_fanout: usize,
        pub max_fanout: usize,
    }

    impl TestPOSTree {
        pub fn new(exp_fanout: usize, max_fanout: usize) -> Self {
            Self {
                root: 0,
                counter: 0,
                key_num: 0,
                nodes: BTreeMap::new(),
                exp_fanout,
                max_fanout,
            }
        }
    }

    impl POSTreeNodeIO<IKey, IValue> for TestPOSTree {
        fn load_node(&self, node_id: u32) -> Option<POSTreeNode<IKey, IValue>> {
            match self.nodes.get(&node_id) {
                Some(n) => Some(n.clone()),
                None => None,
            }
        }
    
        fn store_node(&mut self, node_id: u32, node: POSTreeNode<IKey, IValue>) {
            self.nodes.insert(node_id, node);
        }
    
        fn store_nodes_batch(&mut self, nodes: BTreeMap::<u32, POSTreeNode<IKey, IValue>>) {
            self.nodes.extend(nodes);
        }
    
        fn remove_node(&mut self, node_id: u32) -> Option<POSTreeNode<IKey, IValue>> {
            self.nodes.remove(&node_id)
        }

        fn update_node_prev(&mut self, node_id: u32, prev: u32) {
            match self.nodes.get_mut(&node_id) {
                Some(node_mut_ref) => {
                    if node_mut_ref.is_leaf() {
                        node_mut_ref.get_mut_leaf().unwrap().prev = prev;
                    } else {
                        node_mut_ref.get_mut_internal().unwrap().prev = prev;
                    }
                },
                None => {},
            }
        }

        fn update_node_next(&mut self, node_id: u32, next: u32) {
            match self.nodes.get_mut(&node_id) {
                Some(node_mut_ref) => {
                    if node_mut_ref.is_leaf() {
                        node_mut_ref.get_mut_leaf().unwrap().next = next;
                    } else {
                        node_mut_ref.get_mut_internal().unwrap().next = next;
                    }
                },
                None => {},
            }
        }
    
        fn new_counter(&mut self) -> u32 {
            self.counter += 1;
            self.counter
        }
    
        fn set_counter(&mut self, counter: u32) {
            self.counter = counter
        }
    
        fn get_counter(&self) -> u32 {
            self.counter
        }
    
        fn get_root_id(&self) -> u32 {
            self.root
        }
    
        fn set_root_id(&mut self, root_id: u32) {
            self.root = root_id;
        }
    
        fn get_root_hash(&self) -> H256 {
            if self.key_num == 0 {
                return H256::default();
            } else {
                self.nodes.get(&self.root).unwrap().to_digest()
            }
        }
    
        fn increment_key_num(&mut self) {
            self.key_num += 1;
        }
    
        fn deduct_key_num(&mut self) {
            self.key_num -= 1;
        }
    
        fn set_key_num(&mut self, key_num: u32) {
            self.key_num = key_num;
        }
    
        fn get_key_num(&self) -> u32 {
            self.key_num
        }
    
        fn load_all_key_values(&self) -> Vec<(IKey, IValue)> {
            let mut values = Vec::<(IKey, IValue)>::new();
            if self.key_num != 0 {
                // get the left most leaf node's id
                let mut cur_leaf_id = get_left_most_leaf_id(self);
                // iteratively scan the leaf nodes from left to right until the leaf's next pointer is 0
                while cur_leaf_id != 0 {
                    let leaf = self.load_node(cur_leaf_id).unwrap().to_leaf().unwrap();
                    for i in 0..leaf.get_n() {
                        let (key, value) = leaf.key_values[i];
                        values.push((key, value));
                    }
                    cur_leaf_id = leaf.next;
                }
            }
            return values;
        }
    
        fn set_max_fanout(&mut self, max_fanout: usize) {
            self.max_fanout = max_fanout;
        }
    
        fn get_max_fanout(&self) -> usize {
            self.max_fanout
        }
    
        fn set_exp_fanout(&mut self, exp_fanout: usize) {
            self.exp_fanout = exp_fanout;
        }
    
        fn get_exp_fanout(&self) -> usize {
            self.exp_fanout
        }
    }

    #[test]
    fn test_in_memory_pos_tree() {
        let exp_fanout = 4;
        let max_fanout = 15;
        let mut tree = TestPOSTree::new(exp_fanout, max_fanout);
        let mut rng = StdRng::seed_from_u64(1);
        let n = 500000;
        let mut inputs: Vec<u32> = (1..=n).collect();
        inputs.shuffle(&mut rng);
        for (i, input) in inputs.iter().enumerate() {
            // println!("i: {}, input: {}", i, input);
            println!("i: {}", i);
            let key = IKey(*input);
            let value = IValue(*input);
            insert(&mut tree, key, value);
        }
        println!("--------------");
        let check_leaf_pointers = check_prev_next_pointers(&tree);
        assert!(check_leaf_pointers);
        let root_h = tree.get_root_hash();
        println!("root: {:?}", root_h);
        for input in inputs.iter() {
            let key = IKey(*input);
            // let (r_k, r_v) = search_without_proof(&tree, key).unwrap();
            // assert_eq!(r_k, key);
            // assert_eq!(r_v, IValue(*input));
            let (r, p) = get_range_proof(&tree, key, key);
            let h = reconstruct_range_proof(key, key, &r, &p);
            assert_eq!(h, root_h);
        }
    }

    #[test]
    fn test_pos_tree_delete() {
        let exp_fanout = 4;
        let max_fanout = 15;
        let mut tree = TestPOSTree::new(exp_fanout, max_fanout);
        let mut rng = StdRng::seed_from_u64(1);
        let n = 1000;
        let mut inputs: Vec<u32> = (1..=n).collect();
        inputs.shuffle(&mut rng);
        for (i, input) in inputs.iter().enumerate() {
            println!("i: {}, input: {}", i, input);
            let key = IKey(*input);
            let value = IValue(*input);
            insert(&mut tree, key, value);
            // let h = tree.get_root_hash();
            // println!("h: {:?}", h);
        }

        // print_tree(&tree);
        println!("--------------------------------");
        inputs.shuffle(&mut rng);
        for input in &inputs {
            println!("remove input: {}", input);
            let key = IKey(*input);
            remove(&mut tree, key);
            // println!("-----------tree--------------");
            // println!("tree: {:?}", tree);
            // if tree.counter != 0 {
            //     print_tree(&tree);
            // }
            // println!("tree root id: {}", tree.get_root_id());
            // println!("--------------------------------");
        }

        println!("end remove --------------------------------");
    }
}