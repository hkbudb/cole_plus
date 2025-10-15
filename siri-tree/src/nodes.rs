use utils::types::{Digestible, LoadAddr, Num, Value};
use utils::types::bytes_hash;
use primitive_types::H256;
use serde::{Serialize, Deserialize};
/* Leaf node
    key_values: store the key-value pairs in the leaf node
    next: the pointer to the next leaf node
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SIRILeafNode<K: Num, V: Value> {
    pub key_values: Vec<(K, V)>,
    pub next: u32,
    pub prev: u32,
}

impl<K: Num, V: Value> Digestible for SIRILeafNode<K, V> {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        for (k, v) in &self.key_values {
            let key_bytes = bincode::serialize(k).unwrap();
            let value_bytes = bincode::serialize(v).unwrap();
            bytes.extend(&key_bytes);
            bytes.extend(&value_bytes);
        }
        bytes_hash(&bytes)
    }
}

impl<K: Num, V: Value> SIRILeafNode<K, V> {
    // create a new leaf node with key-value pairs
    pub fn new(key_values: Vec<(K, V)>) -> Self {
        Self {
            key_values,
            next: 0,
            prev: 0,
        }
    }
    // get number of keys
    pub fn get_n(&self) -> usize {
        self.key_values.len()
    }

    // [2, 4, 6, 8, 10], key: 5, return 1; key 6, return 2; key 7, return 2;
    pub fn search_prove_idx(&self, key: K) -> usize {
        let n = self.key_values.len();
        if key < self.key_values[0].0 {
            return 0;
        } else if key > self.key_values.last().unwrap().0 {
            return n - 1;
        } else {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.key_values[mid].0 < key {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            if self.key_values[low].0 != key {
                low -= 1; // should decrease 1 for the unmatched key to ensure [left, right) constraint
            }
            return low;
        }
    }

    pub fn search_prove_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        let l = self.search_prove_idx(lb);
        let r = self.search_prove_idx(ub);
        return (l, r);
    }

    pub fn search_insert_idx(&self, key: K) -> usize {
        let n = self.key_values.len();
        if key < self.key_values[0].0 {
            return 0;
        } 
        else {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.key_values[mid].0 < key {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            return low;
        }
    }
}

/*  Internal node
    keys: the direction keys to the child nodes [k1, k2, k3] [c1, c2, c3] -> all keys in c1 <= k1; keys in c2 <= k2; ...
    childs: the pointers of the child nodes
    child_hashes: the hash values of the child nodes
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SIRIInternalNode<K: Num + LoadAddr> {
    pub keys: Vec<K>,
    pub childs: Vec<u32>,
    pub child_hashes: Vec<H256>,
    pub next: u32,
    pub prev: u32,
}

impl<K: Num + LoadAddr> Digestible for SIRIInternalNode<K> {
    // digest of an internal node is computed from H(H(k_0) || H(k_1) || ... || H(k_f) || H(h_0) || H(h_1) || ... H(h_{f+1})), assume that there are f+1 child nodes
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        // for k in &self.keys {
        //     let key_bytes = bincode::serialize(k).unwrap();
        //     bytes.extend(key_bytes);
        // }

        for h in &self.child_hashes {
            bytes.extend(h.as_bytes());
        }
        bytes_hash(&bytes)
    }
}

impl<K: Num + LoadAddr> SIRIInternalNode<K> {
    // create an internal node
    pub fn new(keys: Vec<K>, childs: Vec<u32>, child_hashes: Vec<H256>) -> Self {
        Self {
            keys,
            childs,
            child_hashes,
            next: 0,
            prev: 0,
        }
    }
    // given an index, get the node id of the child node
    pub fn get_child_id(&self, idx: usize) -> u32 {
        *self.childs.get(idx).unwrap()
    }
    // given an index, get the hash value of the child node
    pub fn get_child_hash(&self, idx: usize) -> H256 {
        *self.child_hashes.get(idx).unwrap()
    }
    // get the number of keys in the internal node
    pub fn get_n(&self) -> usize {
        self.keys.len()
    }

    pub fn search_key_idx(&self, key: K) -> usize {
        let n = self.keys.len();
        if key < self.keys[0] {
            return 0;
        } else if &key > self.keys.last().unwrap() {
            return n - 1;
        } else {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.keys[mid] < key {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            return low;
        }
    }

    pub fn search_key_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        let l = self.search_key_idx(lb);
        let r = self.search_key_idx(ub);
        return (l, r);
    }

    pub fn search_insert_idx(&self, key: K) -> usize {
        let n = self.keys.len();
        if key < self.keys[0] {
            return 0;
        } else {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                if self.keys[mid] < key {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            return low;
        }
    }
}

/* The Enumerator of the node: either a leaf node or an internal node
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum SIRINode<K: Num + LoadAddr, V: Value> {
    Leaf(SIRILeafNode<K, V>),
    NonLeaf(SIRIInternalNode<K>),
}

impl<K: Num + LoadAddr, V: Value> Digestible for SIRINode<K, V> {
    fn to_digest(&self) -> H256 {
        match self {
            SIRINode::NonLeaf(n) => n.to_digest(),
            SIRINode::Leaf(n) => n.to_digest(),
        }
    }
}

// try to transform a node to a leaf node, if the node is not a leaf node, return None
impl<K: Num + LoadAddr, V: Value> Into<Option<SIRILeafNode<K, V>>> for SIRINode<K, V> {
    fn into(self) -> Option<SIRILeafNode<K, V>> {
        match self {
            SIRINode::Leaf(n) => Some(n),
            SIRINode::NonLeaf(_) => None,
        }
    }
}

impl<K: Num + LoadAddr, V: Value> SIRINode<K, V> {
    pub fn get_keys(&self) -> Vec<K> {
        match self {
            SIRINode::Leaf(n) => {
                let keys: Vec<K> = n.key_values.iter().map(|(k, _)| *k).collect();
                keys
            },
            SIRINode::NonLeaf(n) => n.keys.clone(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            SIRINode::Leaf(_) => true,
            SIRINode::NonLeaf(_) => false,
        }
    }

    pub fn to_leaf(self) -> Option<SIRILeafNode<K, V>> {
        match self {
            SIRINode::Leaf(n) => Some(n),
            SIRINode::NonLeaf(_) => None,
        }
    }

    pub fn get_leaf(&self) -> Option<&SIRILeafNode<K, V>> {
        match self {
            SIRINode::Leaf(n) => Some(n),
            SIRINode::NonLeaf(_) => None,
        }
    }

    pub fn get_internal(&self) -> Option<&SIRIInternalNode<K>> {
        match self {
            SIRINode::Leaf(_) => None,
            SIRINode::NonLeaf(n) => Some(n),
        }
    }

    pub fn to_internal(self) -> Option<SIRIInternalNode<K>> {
        match self {
            SIRINode::Leaf(_) => None,
            SIRINode::NonLeaf(n) => Some(n),
        }
    }

    pub fn from_leaf(leaf: SIRILeafNode<K, V>) -> Self {
        Self::Leaf(leaf)
    }

    pub fn from_internal(non_leaf: SIRIInternalNode<K>) -> Self {
        Self::NonLeaf(non_leaf)
    }

    pub fn search_prove_idx(&self, key: K) -> usize {
        match &self {
            SIRINode::Leaf(node) => node.search_prove_idx(key),
            SIRINode::NonLeaf(node) => node.search_key_idx(key),
        }
    }

    pub fn search_prove_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        match &self {
            SIRINode::Leaf(node) => node.search_prove_idx_range(lb, ub),
            SIRINode::NonLeaf(node) => node.search_key_idx_range(lb, ub),
        }
    }
    
    // get the child id given the index
    pub fn get_internal_child(&self, index: usize) -> Option<u32> {
        match &self {
            SIRINode::Leaf(_) => None,
            SIRINode::NonLeaf(node) => {
                return Some(node.childs[index])
            },
        }
    }

    pub fn get_n(&self,) -> usize {
        match &self {
            SIRINode::Leaf(leaf) => {
                leaf.get_n()
            },
            SIRINode::NonLeaf(node) => {
                node.get_n()
            },
        }
    }

    pub fn get_mut_leaf(&mut self) -> Option<&mut SIRILeafNode<K, V>> {
        match self {
            SIRINode::Leaf(leaf) => {
                Some(leaf)
            },
            _ => { 
                None 
            }
        }
    }

    pub fn get_mut_internal(&mut self) -> Option<&mut SIRIInternalNode<K>> {
        match self {
            SIRINode::NonLeaf(node) => {
                Some(node)
            },
            _ => { 
                None 
            }
        }
    }
}
