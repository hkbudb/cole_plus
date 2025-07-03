use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, VecDeque};
use pattern_oridented_split_tree::{nodes::POSTreeNode, traits::POSTreeNodeIO, get_left_most_leaf_id};
use utils::types::{CompoundKey, StateValue, Digestible};
use primitive_types::H256;

#[derive(Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct InMemoryPOSTree {
    pub root: u32,
    pub counter: u32,
    pub key_num: u32,
    pub exp_fanout: usize,
    pub max_fanout: usize,
    pub nodes: BTreeMap<u32, POSTreeNode<CompoundKey, StateValue>>, // use a map to store the nodes in the memory
}

impl POSTreeNodeIO<CompoundKey, StateValue> for InMemoryPOSTree {
    fn load_node(&self, node_id: u32) -> Option<POSTreeNode<CompoundKey, StateValue>> {
        match self.nodes.get(&node_id) {
            Some(n) => Some(n.clone()),
            None => None,
        }
    }

    fn store_node(&mut self, node_id: u32, node: POSTreeNode<CompoundKey, StateValue>) {
        self.nodes.insert(node_id, node);
    }

    fn store_nodes_batch(&mut self, nodes: BTreeMap::<u32, POSTreeNode<CompoundKey, StateValue>>) {
        self.nodes.extend(nodes);
    }

    fn remove_node(&mut self, node_id: u32) -> Option<POSTreeNode<CompoundKey, StateValue>> {
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

    fn load_all_key_values(&self) -> Vec<(CompoundKey, StateValue)> {
        // initiate the data vector
        let mut key_values = Vec::<(CompoundKey, StateValue)>::new();
        // check whether the tree is empty
        if self.key_num != 0 {
            // get the left-most leaf's id
            let mut cur_leaf_id = get_left_most_leaf_id(self);
            // iterate the leaf nodes from the lest-most to the right-most, until the next pointer is 0
            while cur_leaf_id != 0 {
                let leaf = self.load_node(cur_leaf_id).unwrap().to_leaf().unwrap();
                for i in 0..leaf.get_n() {
                    let key_value = leaf.key_values[i];
                    key_values.push(key_value);
                }
                cur_leaf_id = leaf.next;
            }
        }
        return key_values;
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

impl InMemoryPOSTree {
    // construct a new in-memory POS-Tree using fanout
    pub fn new(exp_fanout: usize, max_fanout: usize) -> Self {
        Self {
            root: 0,
            counter: 0,
            key_num: 0,
            exp_fanout,
            max_fanout,
            nodes: BTreeMap::new(),
        }
    }
    // clear the MB-Tree space
    pub fn clear(&mut self) {
        self.root = 0;
        self.counter = 0;
        self.key_num = 0;
        self.nodes.clear();
    }

    pub fn print_tree(&self) {
        let node_id = self.root;
        let node = self.load_node(node_id).unwrap();
        let mut queue = VecDeque::new();
        queue.push_back((node_id, node));
        while !queue.is_empty() {
            let (node_id, node) = queue.pop_front().unwrap();
            println!("h: {:?}, id: {}, is leaf: {}, node size: {:?}, node: {:?}", node.to_digest(), node_id, node.is_leaf(), node.get_n(), node);
            match node {
                POSTreeNode::Leaf(_) => {},
                POSTreeNode::NonLeaf(internal) => {
                    for child_id in &internal.childs {
                        let child_node = self.load_node(*child_id).unwrap();
                        queue.push_back((*child_id, child_node));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_oridented_split_tree::print_tree;
    use primitive_types::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use utils::types::{AddrKey, Address, StateKey};
    #[test]
    fn test_insert_mem_postree() {
        let num_of_contract = 1;
        let num_of_addr = 50;
        let num_of_version = 50;
        // let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let fanout = 4;

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

        let mut tree = InMemoryPOSTree::new(fanout, 100);
        let mut i = 0;
        let start = std::time::Instant::now();
        for (key, value) in &state_vec {
            i += 1;
            pattern_oridented_split_tree::insert(&mut tree, *key, *value);
            println!("i: {}", i);
        }
        let elapse = start.elapsed().as_nanos();
        println!("insert time: {:?}", elapse);
        print_tree(&tree);
    }
}