use std::collections::BTreeMap;
use utils::types::{LoadAddr, Num, Value};
use primitive_types::H256;
use crate::nodes::SIRINode;

/* Trait of the SIRI Storage
 */
pub trait SIRINodeIO<K: Num + LoadAddr, V: Value> {
    // given a node id, load a node from the storage, if the id does not exist, return none
    fn load_node(&self, node_id: u32) -> Option<SIRINode<K, V>>;
    // store a node with node_id
    fn store_node(&mut self, node_id: u32, node: SIRINode<K, V>);
    // store a batch of nodes in a map
    fn store_nodes_batch(&mut self, nodes: BTreeMap::<u32, SIRINode<K, V>>);
    // remove a node with node_id
    fn remove_node(&mut self, node_id: u32) -> Option<SIRINode<K, V>>;
    // update node prev
    fn update_node_prev(&mut self, node_id: u32, prev: u32);
    // update node next
    fn update_node_next(&mut self, node_id: u32, next: u32);
    // create a new counter of the node_id
    fn new_counter(&mut self) -> u32;
    // set the latest counter
    fn set_counter(&mut self, counter: u32);
    // get the latest counter of the storage
    fn get_counter(&self) -> u32;
    // get the id of the root node
    fn get_root_id(&self) -> u32;
    // set the id of the root node as root_id
    fn set_root_id(&mut self, root_id: u32);
    // get the root hash of the storage
    fn get_root_hash(&self) -> H256;
    // increment the number of keys in the storage
    fn increment_key_num(&mut self);
    // deduct the number of keys in the storage
    fn deduct_key_num(&mut self);
    // set the number of keys in the storage
    fn set_key_num(&mut self, key_num: u32);
    // get the number of keys in the storage
    fn get_key_num(&self) -> u32;
    // batchly load all the key-value pairs from the storage
    fn load_all_key_values(&self) -> Vec<(K, V)>;
    // set the maximum fanout
    fn set_max_fanout(&mut self, max_fanout: usize);
    // get the maximum fanout
    fn get_max_fanout(&self) -> usize;
    // set the exp fanout
    fn set_exp_fanout(&mut self, exp_fanout: usize);
    // get the exp fanout
    fn get_exp_fanout(&self) -> usize;
}