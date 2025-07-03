use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use merkle_btree_storage::{nodes::BPlusTreeNode, traits::BPlusTreeNodeIO, get_left_most_leaf_id};
use utils::types::{CompoundKey, StateValue, Digestible};
use primitive_types::H256;

#[derive(Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct InMemoryMBTree {
    pub root: u32,
    pub counter: u32,
    pub key_num: u32,
    pub fanout: usize,
    pub nodes: BTreeMap<u32, BPlusTreeNode<CompoundKey, StateValue>>, // use a map to store the nodes in the memory
}

impl BPlusTreeNodeIO<CompoundKey, StateValue> for InMemoryMBTree {
    fn load_node(&self, node_id: u32) -> Option<BPlusTreeNode<CompoundKey, StateValue>> {
        match self.nodes.get(&node_id) {
            Some(n) => Some(n.clone()),
            None => None,
        }
    }

    fn store_node(&mut self, node_id: u32, node: BPlusTreeNode<CompoundKey, StateValue>) {
        self.nodes.insert(node_id, node);
    }

    fn store_nodes_batch(&mut self, nodes: &BTreeMap::<u32, BPlusTreeNode<CompoundKey, StateValue>>) {
        self.nodes.extend(nodes.clone());
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

    fn get_storage_id(&self,) -> u64 {
        0
    }

    fn get_root_storage_key(_: u64) -> Vec<u8> {
        vec![]
    }

    fn get_counter_storage_key(_: u64) -> Vec<u8> {
        vec![]
    }

    fn get_key_num_storage_key(_: u64) -> Vec<u8> {
        vec![]
    }

    fn set_fanout(&mut self, fanout: usize) {
        self.fanout = fanout;
    }

    fn get_fanout(&self) -> usize {
        self.fanout
    }
}

impl InMemoryMBTree {
    // construct a new in-memory MB-Tree using fanout
    pub fn new(fanout: usize) -> Self {
        Self {
            root: 0,
            counter: 0,
            key_num: 0,
            fanout,
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

    pub fn print_tree(&mut self) {
        let node_id = self.root;
        let node = self.load_node(node_id).unwrap();
        let mut queue = Vec::<(u32, BPlusTreeNode<CompoundKey, StateValue>)>::new();
        queue.push((node_id, node));
        while !queue.is_empty() {
            let (node_id, node) = queue.remove(0);
            println!("{:?} {:?}", node_id, node);
            if !node.is_leaf() {
                let internal = node.get_internal().unwrap();
                for c in &internal.childs {
                    let child = self.load_node(*c).unwrap();
                    queue.push((*c, child));
                }
            }
        }
    }
}