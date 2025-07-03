use std::{collections::{BTreeSet, HashMap}, fmt::Debug, fs::{File, OpenOptions}, io::{Read, Seek, SeekFrom, Write}};
use primitive_types::H256;
use serde::{Serialize, Deserialize};
use crate::{cacher::CacheManager, types::{bytes_hash, Digestible, StateValue}};
use cdc_hash::{CDCHash, CDCResult};
use std::collections::VecDeque;
use anyhow::{Result, anyhow};
use crate::pager::{Page, PAGE_SIZE};

// define some size constants to compute the MHT node size
pub const LEAF_FLAG_SIZE: usize = 1;
pub const VER_SIZE: usize = 4;
pub const VER_OBJECT_SIZE: usize = VER_SIZE + 32;
pub const N_SIZE: usize = 1;
pub const OFFSET_SIZE: usize = 8;
pub const NODE_SIZE_LEN: usize = 2;
// for each compound key, historical version and state value
#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub struct VerObject {
    pub ver: u32,
    pub value: StateValue,
}

impl VerObject {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let obj: Self = bincode::deserialize(&bytes).unwrap();
        return obj;
    }

    pub fn new(ver: u32, value: StateValue) -> Self {
        Self {
            ver,
            value,
        }
    }
}

impl Digestible for VerObject {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

impl PartialOrd for VerObject {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.ver.cmp(&other.ver).then(self.value.cmp(&other.value)))
    }
}

impl PartialEq for VerObject {
    fn eq(&self, other: &Self) -> bool {
        self.ver == other.ver && self.value == other.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CDCLeafNode {
    objs: Vec<VerObject>,
}

impl Digestible for CDCLeafNode {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        // compute leaf node's hash using each obj's hash
        for obj in &self.objs {
            bytes.extend(obj.to_digest().as_bytes());
        }
        return bytes_hash(&bytes);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CDCInternalNode {
    keys: Vec<u32>,
    child_hashes: Vec<H256>,
}

impl Digestible for CDCInternalNode {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        for child_hash in &self.child_hashes {
            bytes.extend(child_hash.as_bytes());
        }
        return bytes_hash(&bytes);
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum CDCNode {
    Leaf(CDCLeafNode),
    NonLeaf(CDCInternalNode),
}

impl Digestible for CDCNode {
    fn to_digest(&self) -> H256 {
        match self {
            CDCNode::Leaf(leaf) => leaf.to_digest(),
            CDCNode::NonLeaf(internal) => internal.to_digest(),
        }
    }
}

impl CDCNode {
    pub fn insert_to_leaf(&mut self, ver_obj: VerObject) {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.push(ver_obj);
            },
            Self::NonLeaf(_) => {

            }
        }
    }

    pub fn batch_load_to_leaf(&mut self, ver_obj_vec: Vec<VerObject>) {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.extend(ver_obj_vec);
            },
            Self::NonLeaf(_) => {

            }
        }
    }

    pub fn insert_to_internal(&mut self, key: u32, child_hash: H256) {
        match self {
            Self::Leaf(_) => {},
            Self::NonLeaf(node) => {
                node.keys.push(key);
                node.child_hashes.push(child_hash);
            }
        }
    }

    pub fn batch_load_to_internal(&mut self, key_hash_vec: Vec<(u32, H256)>) {
        match self {
            Self::Leaf(_) => {},
            Self::NonLeaf(node) => {
                for (key, child_hash) in key_hash_vec {
                    node.keys.push(key);
                    node.child_hashes.push(child_hash);
                }
            }
        }
    }

    pub fn get_node_last_key(&self) -> u32 {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.last().unwrap().ver
            },
            Self::NonLeaf(node) => {
                *node.keys.last().unwrap()
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.is_empty()
            },
            Self::NonLeaf(node) => {
                node.keys.is_empty() && node.child_hashes.is_empty()
            }
        }
    }

    pub fn num_of_keys(&self) -> usize {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.len()
            },
            Self::NonLeaf(node) => {
                node.keys.len()
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        if let &Self::Leaf(_) = self {
            true
        } else {
            false
        }
    }
    // with input of ver range l and r, return the lower_bound index minus 1 and plus 1 for completeness
    pub fn search_node_idx_range(&self, l: u32, r: u32) -> (usize, usize) {
        match self {
            Self::Leaf(leaf) => {
                let arr = &leaf.objs;
                let mut low_idx = general_lower_bound(arr, l);
                // reduce 1 for low_idx to ensure completeness
                if low_idx != 0 {
                    low_idx -= 1;
                }
                let mut up_idx = general_lower_bound(arr, r);
                // plus 1 for up_idx if ver equals to r, to ensure completeness
                if arr[up_idx].ver == r && up_idx != arr.len() - 1 {
                    up_idx += 1;
                }
                return (low_idx, up_idx);
            },
            Self::NonLeaf(node) => {
                let arr: &Vec<InnerVer> = &node.keys.iter().map(|elem| InnerVer(elem)).collect();
                let low_idx = general_lower_bound(arr, l);
                let up_idx = general_lower_bound(arr, r);
                return (low_idx, up_idx);
            }
        }
    }

    pub fn get_internal(&self) -> Option<&CDCInternalNode> {
        match self {
            Self::Leaf(_) => None,
            Self::NonLeaf(n) => Some(n),
        }
    }

    pub fn get_leaf(&self) -> Option<&CDCLeafNode> {
        match self {
            Self::NonLeaf(_) => None,
            Self::Leaf(n) => Some(n),
        }
    }

    pub fn get_size(&self) -> usize {
        let n = self.num_of_keys();
        match self {
            Self::Leaf(_) => {
                // leaf_flag 1 byte + num_of_keys 1 byte + n * ver_obj_size 36 bytes
                return LEAF_FLAG_SIZE + N_SIZE + n * VER_OBJECT_SIZE;
            },
            Self::NonLeaf(_) => {
                // leaf_flag 1 byte + num_of_keys 1 byte + n * ver_size 4 bytes + n * offset_size 8 bytes + n * hash_size 32 bytes
                return LEAF_FLAG_SIZE + N_SIZE + n * VER_SIZE +  n * OFFSET_SIZE + n * 32;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PersistCDCLeafNode {
    pub leaf_flag: bool,
    pub n: u8,
    pub objs: Vec<VerObject>,
}

impl PersistCDCLeafNode {
    pub fn from_cdc_leaf_node(node: CDCLeafNode) -> Self {
        Self {
            leaf_flag: true,
            n: node.objs.len() as u8,
            objs: node.objs,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        let b_num = 1u8; // is leaf
        bytes.extend(&b_num.to_be_bytes()); // write leaf_flag
        bytes.extend(self.n.to_be_bytes()); // write n
        for obj in &self.objs {
            bytes.extend(obj.to_bytes()); // write each obj
        }
        return bytes;
    }
}

impl Digestible for PersistCDCLeafNode {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        // compute leaf node's hash using each obj's hash
        for obj in &self.objs {
            bytes.extend(obj.to_digest().as_bytes());
        }
        return bytes_hash(&bytes);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PersistCDCInternalNode {
    pub leaf_flag: bool,
    pub n: u8,
    pub keys: Vec<u32>,
    pub child_offset: Vec<u64>,
    pub child_hashes: Vec<H256>,
}

impl PersistCDCInternalNode {
    pub fn from_cdc_internal_node(node: CDCInternalNode) -> Self {
        let l = node.keys.len();
        Self {
            leaf_flag: false,
            n: l as u8,
            keys: node.keys,
            child_offset: vec![0; l], // first init the offsets to with keys number of 0s, then fill them during the file persistence
            child_hashes: node.child_hashes,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        let b_num = 0u8; // is not leaf
        bytes.extend(&b_num.to_be_bytes()); // write leaf_flag
        bytes.extend(self.n.to_be_bytes()); // write n
        for key in &self.keys {
            bytes.extend(key.to_be_bytes());
        }
        for offset in &self.child_offset {
            bytes.extend(offset.to_be_bytes());
        }
        for h in &self.child_hashes {
            bytes.extend(h.as_bytes());
        }
        return bytes;
    }
}

impl Digestible for PersistCDCInternalNode {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        for child_hash in &self.child_hashes {
            bytes.extend(child_hash.as_bytes());
        }
        return bytes_hash(&bytes);
    }
}

#[derive(Debug)]
pub enum PersistCDCNode {
    Leaf(PersistCDCLeafNode),
    NonLeaf(PersistCDCInternalNode),
}

impl Digestible for PersistCDCNode {
    fn to_digest(&self) -> H256 {
        match self {
            PersistCDCNode::Leaf(leaf) => leaf.to_digest(),
            PersistCDCNode::NonLeaf(internal) => internal.to_digest(),
        }
    }
}

impl PersistCDCNode {
    pub fn from_cdc_node(node: CDCNode) -> Self {
        match node {
            CDCNode::Leaf(leaf) => {
                let persist_leaf = PersistCDCLeafNode::from_cdc_leaf_node(leaf);
                Self::Leaf(persist_leaf)
            },
            CDCNode::NonLeaf(internal) => {
                let persist_internal = PersistCDCInternalNode::from_cdc_internal_node(internal);
                Self::NonLeaf(persist_internal)
            },
        }
    }

    pub fn to_cdc_node(self) -> CDCNode {
        match self {
            PersistCDCNode::Leaf(leaf) => {
                let mut cdc_leaf = CDCLeafNode::default();
                cdc_leaf.objs = leaf.objs;
                CDCNode::Leaf(cdc_leaf)
            },
            PersistCDCNode::NonLeaf(internal) => {
                let mut cdc_internal = CDCInternalNode::default();
                cdc_internal.keys = internal.keys;
                cdc_internal.child_hashes = internal.child_hashes;
                CDCNode::NonLeaf(cdc_internal)
            }
        }
    }
     
    pub fn num_of_keys(&self) -> usize {
        match self {
            Self::Leaf(leaf) => {
                leaf.objs.len()
            },
            Self::NonLeaf(node) => {
                node.keys.len()
            }
        }
    }

    pub fn get_internal(&self) -> Option<&PersistCDCInternalNode> {
        match self {
            Self::Leaf(_) => None,
            Self::NonLeaf(n) => Some(n),
        }
    }

    pub fn get_leaf(&self) -> Option<&PersistCDCLeafNode> {
        match self {
            Self::NonLeaf(_) => None,
            Self::Leaf(n) => Some(n),
        }
    }

    pub fn get_size(&self) -> usize {
        let n = self.num_of_keys();
        match self {
            Self::Leaf(_) => {
                // leaf_flag 1 byte + num_of_keys 1 byte + n * ver_obj_size 36 bytes
                return LEAF_FLAG_SIZE + N_SIZE + n * VER_OBJECT_SIZE;
            },
            Self::NonLeaf(_) => {
                // leaf_flag 1 byte + num_of_keys 1 byte + n * ver_size 4 bytes + n * offset_size 8 bytes + n * hash_size 32 bytes
                return LEAF_FLAG_SIZE + N_SIZE + n * VER_SIZE +  n * OFFSET_SIZE + n * 32;
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Leaf(leaf) => leaf.to_bytes(),
            Self::NonLeaf(node) => node.to_bytes(),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let b_num: u8 = u8::from_be_bytes(bytes[0..1].try_into().unwrap());
        let leaf_flag = if b_num == 1 {
            true
        } else {
            false
        };
        if leaf_flag {
            let mut leaf = PersistCDCLeafNode::default();
            // read n
            let n = u8::from_be_bytes(bytes[1..2].try_into().unwrap());
            leaf.n = n;
            let mut cursor = 2usize;
            let mut objs: Vec<VerObject> = Vec::new();
            for _ in 0..n {
                // read objs
                let obj: VerObject = VerObject::from_bytes(&bytes[cursor .. cursor + VER_OBJECT_SIZE]);
                objs.push(obj);
                cursor += VER_OBJECT_SIZE;
            }
            leaf.objs = objs;
            return Self::Leaf(leaf);
        } else {
            let mut node = PersistCDCInternalNode::default();
            // read n
            let n = u8::from_be_bytes(bytes[1..2].try_into().unwrap());
            node.n = n;
            let mut cursor = 2usize;
            // read vers
            let mut keys = Vec::<u32>::new();
            for _ in 0..n {
                let ver: u32 = u32::from_be_bytes(bytes[cursor .. cursor + VER_SIZE].try_into().unwrap());
                keys.push(ver);
                cursor += VER_SIZE;
            }
            node.keys = keys;
            // read offsets
            let mut child_offset = Vec::<u64>::new();
            for _ in 0..n {
                let offset: u64 = u64::from_be_bytes(bytes[cursor .. cursor + OFFSET_SIZE].try_into().unwrap());
                child_offset.push(offset);
                cursor += OFFSET_SIZE;
            }
            node.child_offset = child_offset;
            // read child hash
            let mut child_hashes = Vec::<H256>::new();
            for _ in 0..n {
                let h = H256::from_slice(&bytes[cursor .. cursor + 32]);
                child_hashes.push(h);
                cursor += 32;
            }
            node.child_hashes = child_hashes;
            return Self::NonLeaf(node);
        }
    }

    // with input of ver range l and r, return the lower_bound index minus 1 and plus 1 for completeness
    pub fn search_node_idx_range(&self, l: u32, r: u32) -> (usize, usize) {
        match self {
            Self::Leaf(leaf) => {
                let arr = &leaf.objs;
                let mut low_idx = general_lower_bound(arr, l);
                // reduce 1 for low_idx to ensure completeness
                if low_idx != 0 {
                    low_idx -= 1;
                }
                let mut up_idx = general_lower_bound(arr, r);
                // plus 1 for up_idx if ver equals to r, to ensure completeness
                if arr[up_idx].ver == r && up_idx != arr.len() - 1 {
                    up_idx += 1;
                }
                return (low_idx, up_idx);
            },
            Self::NonLeaf(node) => {
                let arr: &Vec<InnerVer> = &node.keys.iter().map(|elem| InnerVer(elem)).collect();
                let low_idx = general_lower_bound(arr, l);
                let up_idx = general_lower_bound(arr, r);
                return (low_idx, up_idx);
            }
        }
    }
}

pub struct CDCTreeWriter {
    pub file: File, // file object of the stored Merkle Trees
    pub latest_activated_page: Page,
    pub unflushed_page_addr: u64, 
    pub new_tree_start_offset: u64, // starting offset of the current writing tree in the file
}

impl CDCTreeWriter {
    /* Initialize the writer with a given file name
     */
    pub fn new(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            latest_activated_page: Page::new(),
            unflushed_page_addr: 0,
            new_tree_start_offset: 0,
        }
    }

    pub fn write_tree(&mut self, tree: &mut CDCTree) -> u64 {
        let root_h = tree.root_h; // record the root hash, seen as a starting point when flushing the nodes to the file
        let node = tree.load_node(&root_h).unwrap();
        let mut queue = VecDeque::new();
        let mut offset_map = HashMap::<H256, u64>::new();
        queue.push_back(node);

        // file_cursor pointer
        let mut file_cursor = self.new_tree_start_offset;
        let mut tree_addr = file_cursor;
        // compute param's bytes
        let param_bytes = tree.serialize_params();
        let inner_page_cursor = file_cursor as usize % PAGE_SIZE;
        // check page capacity
        if PAGE_SIZE - inner_page_cursor < (param_bytes.len() + NODE_SIZE_LEN + node.get_size()) {
            // the rest of the space in this page cannot record tree params + node_size len (2 bytes) + the root node len
            // should set file_cursor to the next page addr
            file_cursor = (file_cursor / PAGE_SIZE as u64 + 1) * PAGE_SIZE as u64;
            // should update the tree addr 
            tree_addr = file_cursor;
        }
        // update file_cursor to the point after the params bytes
        file_cursor += param_bytes.len() as u64;
        // the following lines are used to compute each node's offset and put it into offset_map
        while !queue.is_empty() {
            let cur_node = queue.pop_front().unwrap();
            let cur_node_h = cur_node.to_digest();
            let node_size = cur_node.get_size();
            let inner_page_cursor = file_cursor as usize % PAGE_SIZE;
            if PAGE_SIZE - inner_page_cursor < (NODE_SIZE_LEN + node_size) {
                // no space to store this node, set file_cursor to the next page addr
                file_cursor = (file_cursor / PAGE_SIZE as u64 + 1) * PAGE_SIZE as u64;
            }
            offset_map.insert(cur_node_h, file_cursor); // record cur_node's offset
            file_cursor += (NODE_SIZE_LEN + node_size) as u64; // update the file_cursor
            if let CDCNode::NonLeaf(internal) = cur_node {
                // cur_node is an internal node, should add its child nodes
                for child_h in &internal.child_hashes {
                    // check whether child_h's node exists in the tree (can be partially suppressed)
                    if let Some(child_node) = tree.load_node(child_h) {
                        queue.push_back(child_node);
                    }
                }
            }
        }
        // write the tree params to the page
        let mut page_addr = tree_addr / PAGE_SIZE as u64 * PAGE_SIZE as u64;
        let mut inner_page_cursor: usize = tree_addr as usize % PAGE_SIZE;
        let mut block_bytes = &mut self.latest_activated_page.block;
        if page_addr == self.unflushed_page_addr + PAGE_SIZE as u64  {
            // should flush the activated page and create a new page
            self.file.seek(SeekFrom::Start(page_addr - PAGE_SIZE as u64)).unwrap();
            self.file.write_all(&self.latest_activated_page.block).unwrap();
            self.latest_activated_page = Page::new(); // create a new page
            block_bytes = &mut self.latest_activated_page.block;
            self.unflushed_page_addr += PAGE_SIZE as u64;
        }
    
        block_bytes[inner_page_cursor .. inner_page_cursor + param_bytes.len()].copy_from_slice(&param_bytes); // write tree params
        inner_page_cursor += param_bytes.len();
        // write tree nodes to file
        let node = tree.move_to_persist_node(tree.root_h).unwrap();
        let mut queue = VecDeque::new();
        queue.push_back(node);
        while !queue.is_empty() {
            let mut cur_node = queue.pop_front().unwrap();
            if let PersistCDCNode::NonLeaf(internal) = &mut cur_node {
                for (i, child_hash) in internal.child_hashes.iter_mut().enumerate() {
                    // check whether child_hash has offset in the offset_map (if child_hash is suppressed, it will not be in offset_map)
                    if let Some(child_offset) = offset_map.get(child_hash) {
                        internal.child_offset[i] = *child_offset;
                        let child_node = tree.move_to_persist_node(*child_hash).unwrap();
                        queue.push_back(child_node);
                    }
                }
            }
            // write persist node to bytes
            let mut buf: Vec<u8> = Vec::new();
            // write node size
            let node_size = cur_node.get_size() as u16; // cast size to 2 bytes
            buf.extend(node_size.to_be_bytes());
            // write node serialization
            buf.extend(cur_node.to_bytes());
            // check the correctness of the written buf size
            let buf_len = buf.len();
            assert_eq!(buf_len, node_size as usize + NODE_SIZE_LEN);
            // write (node_size, node_ser) to page.block
            if PAGE_SIZE - inner_page_cursor >= buf_len {
                // enough space in this page
                // check the correctness of the node's offset
                let retrieved_offset = *offset_map.get(&cur_node.to_digest()).unwrap();
                assert_eq!(retrieved_offset, page_addr + inner_page_cursor as u64);
                block_bytes[inner_page_cursor .. inner_page_cursor + buf_len].copy_from_slice(&buf);
                inner_page_cursor += buf_len;
            } else {
                // not enough space in this page, should flush the current page first
                self.file.seek(SeekFrom::Start(page_addr)).unwrap();
                self.file.write_all(&self.latest_activated_page.block).unwrap();
                page_addr += PAGE_SIZE as u64; // update the page_addr
                inner_page_cursor = 0; // reset the inner_page_cursor
                self.latest_activated_page = Page::new(); // create a new page
                self.unflushed_page_addr += PAGE_SIZE as u64;
                // check the correctness of the node's offset
                let retrieved_offset = *offset_map.get(&cur_node.to_digest()).unwrap();
                assert_eq!(retrieved_offset, page_addr + inner_page_cursor as u64);
                block_bytes = &mut self.latest_activated_page.block;
                block_bytes[inner_page_cursor .. inner_page_cursor + buf_len].copy_from_slice(&buf);
                inner_page_cursor += buf_len;
            }
        }
        self.new_tree_start_offset = file_cursor; // tree's last offset
        return tree_addr;
    }

    // finalize the latest_activated_page to the file
    pub fn finalize(&mut self) {
        let page_addr = self.unflushed_page_addr;
        self.file.seek(SeekFrom::Start(page_addr)).unwrap();
        self.file.write_all(&self.latest_activated_page.block).unwrap();
    }

    pub fn get_new_tree_start_offset(&self) -> u64 {
        self.new_tree_start_offset
    }

    pub fn to_cdc_tree_reader(self) -> CDCTreeReader {
        CDCTreeReader {
            file: self.file,
        }
    }
}

pub struct CDCTreeReader {
    pub file: File // file object of the stored Merkle Trees
}

impl CDCTreeReader {
    pub fn new(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        Self {
            file,
        }
    }

    pub fn search_range_at_tree_addr(&mut self, tree_addr: u64, run_id: u32, l: u32, r: u32, cache_manager: &mut CacheManager) -> (Option<Vec<VerObject>>, CDCRangeProof) {
        // first read tree params and the root node, note that the tree params and root node should always be in the single page
        let mut page_addr = tree_addr / PAGE_SIZE as u64 * PAGE_SIZE as u64;
        let mut page = self.load_page(run_id, page_addr as usize, cache_manager);
        let mut inner_page_cursor = tree_addr as usize % PAGE_SIZE;
        inner_page_cursor += 3; // skip exp_fanout, gear_hash_level, max_node_capacity
        // read len of min_keep_left_nodes
        let n: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        inner_page_cursor += n as usize; // skip min_keep_left_nodes
        
        // init a range proof
        let mut proof = CDCRangeProof::default();
        let mut value_vec = Vec::<VerObject>::new();
        // create a queue to help traverse the tree
        let mut queue = VecDeque::<PersistCDCNode>::new();
        let root_node = Self::load_tree_node(&page.block, &mut inner_page_cursor).unwrap();
        // push the root node to the queue
        queue.push_back(root_node);
        // some counter to help determine the number of nodes in the level
        let mut prev_cnt = 1;
        let mut cur_cnt = 0;
        // a temporary proof for the current level
        let mut cur_level_proof = Vec::<((usize, usize), CDCNode)>::new();
        // traverse the tree in a while loop until the queue is empty
        while !queue.is_empty() {
            let cur_node = queue.pop_front().unwrap();
            prev_cnt -= 1; // decrease the node counter of the previous level
            match &cur_node {
                PersistCDCNode::NonLeaf(internal) => {
                    // given the lb and ub, get the position range of the child nodes
                    let (start_idx, end_idx) = cur_node.search_node_idx_range(l, r);
                    // update the node counter for the level
                    cur_cnt += end_idx - start_idx + 1;
                    // add the corresponding child nodes to the queue
                    for idx in start_idx ..= end_idx {
                        let child_offset = internal.child_offset[idx];
                        // check whether the child_offset is in a new page
                        let child_page_addr = child_offset / PAGE_SIZE as u64 * PAGE_SIZE as u64;
                        if child_page_addr != page_addr {
                            // different page, should update page addr
                            page_addr = child_page_addr;
                            // update the page
                            page = self.load_page(run_id, page_addr as usize, cache_manager);
                        }
                        // read the node
                        inner_page_cursor = child_offset as usize % PAGE_SIZE;
                        let child_node = Self::load_tree_node(&page.block, &mut inner_page_cursor).unwrap();
                        queue.push_back(child_node);
                    }
                    // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                    let in_memory_node = cur_node.to_cdc_node();
                    cur_level_proof.push(((start_idx, end_idx), in_memory_node));
                },
                PersistCDCNode::Leaf(leaf) => {
                    // the node is a leaf node, retrieve the reference of the leaf node
                    // get the position range of the leaf node
                    let (start_idx, end_idx) = cur_node.search_node_idx_range(l, r);
                    // add the corresponding searched entries to the value_vec
                    for id in start_idx ..= end_idx {
                        let data = leaf.objs[id].clone();
                        value_vec.push(data);
                    }
                    // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                    let in_memory_node = cur_node.to_cdc_node();
                    cur_level_proof.push(((start_idx, end_idx), in_memory_node));
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
    
    pub fn read_tree_root_at(&mut self, tree_addr: u64, run_id: u32, cache_manager: &mut CacheManager) -> H256 {
        // first read tree params and the root node, note that the tree params and root node should always be in the single page
        let page_addr = tree_addr / PAGE_SIZE as u64 * PAGE_SIZE as u64;
        let page = self.load_page(run_id, page_addr as usize, cache_manager);
        let mut inner_page_cursor = tree_addr as usize % PAGE_SIZE;
        // skip exp_fanout
        // skip gear_hash_level
        // skip max_node_capacity
        inner_page_cursor += 3;
        // read len of min_keep_left_nodes
        let n: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        inner_page_cursor += n as usize;
        let root_node = Self::load_tree_node(&page.block, &mut inner_page_cursor).unwrap();
        let root_h = root_node.to_digest();
        return root_h;
    }

    pub fn read_tree_at(&mut self, tree_addr: u64, run_id: u32, cache_manager: &mut CacheManager) -> Result<CDCTree> {
        // first read tree params and the root node, note that the tree params and root node should always be in the single page
        let mut page_addr = tree_addr / PAGE_SIZE as u64 * PAGE_SIZE as u64;
        let mut page = self.load_page(run_id, page_addr as usize, cache_manager);
        let mut inner_page_cursor = tree_addr as usize % PAGE_SIZE;
        // read exp_fanout
        let exp_fanout: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        // read gear_hash_level
        let gear_hash_level: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        // read max_node_capacity
        let max_node_capacity: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        // read len of min_keep_left_nodes
        let n: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
        inner_page_cursor += 1;
        let mut min_keep_left_nodes: Vec<u8> = Vec::new();
        for _ in 0..n {
            // read each min_keep_left_nodes
            let keep_left: u8 = u8::from_be_bytes(page.block[inner_page_cursor .. inner_page_cursor + 1].try_into().unwrap());
            min_keep_left_nodes.push(keep_left);
            inner_page_cursor += 1;
        }
        // init tree
        let mut tree = CDCTree::new(exp_fanout as usize, gear_hash_level as usize, max_node_capacity as usize);
        tree.min_keep_left_nodes = min_keep_left_nodes;
    
        let root_node = Self::load_tree_node(&page.block, &mut inner_page_cursor).unwrap();
        let root_h = root_node.to_digest();
        tree.root_h = root_h; // assign root_h to tree
        let mut queue = VecDeque::new();
        queue.push_back(root_node);
        // iteratively read the tree node
        while !queue.is_empty() {
            let cur_persist_node = queue.pop_front().unwrap();
            if let PersistCDCNode::NonLeaf(internal) = &cur_persist_node {
                for child_offset in &internal.child_offset {
                    // check whether the child node exists
                    if *child_offset != 0 {
                        // check whether the child_offset is in a new page
                        let child_page_addr = *child_offset / PAGE_SIZE as u64 * PAGE_SIZE as u64;
                        if child_page_addr != page_addr {
                            // different page, should update page addr
                            page_addr = child_page_addr;
                            // update the page
                            page = self.load_page(run_id, page_addr as usize, cache_manager);
                        }
                        // read the node
                        inner_page_cursor = *child_offset as usize % PAGE_SIZE;
                        let child_node = Self::load_tree_node(&page.block, &mut inner_page_cursor).unwrap();
                        queue.push_back(child_node);
                    }
                }
            }
            let in_memory_node = cur_persist_node.to_cdc_node(); // change to in_memory CDCNode
            let node_h = in_memory_node.to_digest();
            tree.nodes.insert(node_h, in_memory_node); // insert the in_memory node to tree's node map
        }
        return Ok(tree);
    }

    fn load_tree_node(block: &[u8], inner_page_cursor: &mut usize) -> Result<PersistCDCNode> {
        // read root node size
        let node_size: usize = u16::from_be_bytes(block[*inner_page_cursor .. *inner_page_cursor + 2].try_into().unwrap()) as usize;
        *inner_page_cursor += 2;
        if node_size == 0 {
            return Err(anyhow!("wrong node size"));
        }
        // read node
        let node: PersistCDCNode = PersistCDCNode::from_bytes(block[*inner_page_cursor .. *inner_page_cursor + node_size].try_into().unwrap());
        *inner_page_cursor += node_size;
        return Ok(node);
    }

    fn load_page(&mut self, run_id: u32, page_addr: usize, cache_manager: &mut CacheManager) -> Page {
        // first check whether the cache contains the page
        let page_id = page_addr / PAGE_SIZE;
        let r = cache_manager.read_cdc_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let page = r.unwrap();
            return page;
        } else {
            // cache does not contain the page, should load the page from the file
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(page_addr as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            // before return, add it to the cache with page_id
            cache_manager.set_cdc_cache(run_id, page_id, page.clone());
            return page;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CDCTree {
    pub root_h: H256,
    pub exp_fanout: usize, // gear hash param, the expected fanout of a node
    pub gear_hash_level: usize, // gear hash param, the larger the level, the more '1's in the mask, which results in larger nodes
    pub max_node_capacity: usize, // node size param, to bound the node within max_node_capacity
    pub min_keep_left_nodes: Vec<u8>, // num of left nodes that should be kept during the merge
    pub nodes: HashMap<H256, CDCNode>,
}

impl CDCTree {
    pub fn new(exp_fanout: usize, gear_hash_level: usize, max_node_capacity: usize) -> Self {
        Self {
            root_h: H256::default(),
            exp_fanout,
            gear_hash_level,
            max_node_capacity,
            min_keep_left_nodes: Vec::new(),
            nodes: HashMap::new(),
        }
    }

    pub fn bulk_load(&mut self, objs: Vec<VerObject>) {
        // init cdc hash
        let mut cdc_hash = CDCHash::new(self.exp_fanout, self.gear_hash_level, self.max_node_capacity);
        let mut leaf = CDCNode::Leaf(CDCLeafNode::default());
        let mut pushed_up_entries: Vec<(u32, H256)> = vec![];
        // used to record the root hash
        let mut last_commit_h = H256::default();
        // used to determine the first pattern node
        let mut first_node_flag = true;
        // used to record num of left nodes that should be kept for the leaf level
        let mut min_keep_left_node = 0;
        for obj in objs {
            let obj_h = obj.to_digest();
            // put obj to current leaf
            leaf.insert_to_leaf(obj);
            // check pattern using obj's hash
            let r = cdc_hash.generate_cut_point(obj_h.as_bytes());
            if let CDCResult::PatternFound | CDCResult::ReachCapacity = r {
                // find a pattern or should be the end of the node, finish adding obj to this leaf
                let node_hash = leaf.to_digest();
                // insert push_up entries (last key in leaf, leaf's node hash)
                pushed_up_entries.push((leaf.get_node_last_key(), node_hash));
                last_commit_h = node_hash;
                self.nodes.insert(node_hash, leaf);
                // reset leaf
                leaf = CDCNode::Leaf(CDCLeafNode::default());
                if first_node_flag {
                    if let CDCResult::ReachCapacity = r {
                        min_keep_left_node += 1;
                    }
                    if let CDCResult::PatternFound = r {
                        first_node_flag = false;
                        min_keep_left_node += 1;
                    }
                }
            }
        }
        // handle the last leaf node
        if !leaf.is_empty() {
            let node_hash = leaf.to_digest();
            // insert push_up entries (last key in leaf, leaf's node hash)
            pushed_up_entries.push((leaf.get_node_last_key(), node_hash));
            last_commit_h = node_hash;
            self.nodes.insert(node_hash, leaf);
            if first_node_flag {
                // see this case as CDCResult::ReachCapacity
                min_keep_left_node += 1;
            }
        }
        self.min_keep_left_nodes.push(min_keep_left_node); // insert min_keep_left_node to the vector for the leaf level
        // build internal nodes
        while pushed_up_entries.len() > 1 {
            let prev_level_keep_left_nodes = *self.min_keep_left_nodes.last().unwrap();
            // used to record num of left nodes that should be kept for the leaf level
            cdc_hash.reset_hasher();
            let mut temp_pushed_up_entries: Vec<(u32, H256)> = vec![];
            let mut internal_node = CDCNode::NonLeaf(CDCInternalNode::default());
            let mut min_keep_internal_node = 0; // min_keep num of internal nodes for this level
            let mut accumulate_child_cnt = 0; // counter for accumulating child hashes
            let mut stop_flag = false; // flag to determine whether to continue accumulate child hashes
            let mut first_node_flag = true;
            for  (key, child_hash) in pushed_up_entries {
                // put entry to current internal node
                internal_node.insert_to_internal(key, child_hash);
                // check pattern using entry's child hash
                let r = cdc_hash.generate_cut_point(child_hash.as_bytes());
                if let CDCResult::PatternFound | CDCResult::ReachCapacity = r {
                    // find a pattern or should be the end of the node, finish adding entries to this node
                    let node_hash = internal_node.to_digest();
                    // insert push_up entries (last key in internal node, node's hash)
                    temp_pushed_up_entries.push((internal_node.get_node_last_key(), node_hash));
                    last_commit_h = node_hash;
                    let internal_node_num_keys = internal_node.num_of_keys();
                    self.nodes.insert(node_hash, internal_node);
                    // reset internal node
                    internal_node = CDCNode::NonLeaf(CDCInternalNode::default());

                    // add internal node's num of child to accumulator
                    accumulate_child_cnt += internal_node_num_keys;
                    if first_node_flag {
                        if let CDCResult::ReachCapacity = r {
                            min_keep_internal_node += 1;
                        }
                        if let CDCResult::PatternFound = r {
                            first_node_flag = false;
                            min_keep_internal_node += 1;
                        }
                    } else {
                        if !stop_flag {
                            if accumulate_child_cnt < prev_level_keep_left_nodes as usize {
                                min_keep_internal_node += 1;
                            } else {
                                stop_flag = true;
                            }
                        }
                    }
                }
            }
            // handle the last internal node
            if !internal_node.is_empty() {
                let node_hash = internal_node.to_digest();
                // insert push_up entries (last key in internal node, node's hash)
                temp_pushed_up_entries.push((internal_node.get_node_last_key(), node_hash));
                last_commit_h = node_hash;
                let internal_node_num_keys = internal_node.num_of_keys();
                self.nodes.insert(node_hash, internal_node);

                // add internal node's num of child to accumulator
                accumulate_child_cnt += internal_node_num_keys;
                if first_node_flag {
                    // see this case as CDCResult::ReachCapacity
                    min_keep_internal_node += 1;
                } else {
                    if !stop_flag {
                        if accumulate_child_cnt < prev_level_keep_left_nodes as usize {
                            min_keep_internal_node += 1;
                        }
                    }
                }
            }
            // update the pushed_up_entries
            pushed_up_entries = temp_pushed_up_entries;
            self.min_keep_left_nodes.push(min_keep_internal_node);
        }

        // if there is only one element in pushed up entries, it means the last commit node should be the root node
        self.root_h = last_commit_h;
    }

    pub fn prune_tree_with_latest_version(&mut self) {
        // a set to collect the kept node's hash values
        let mut node_hashes = BTreeSet::<H256>::new();
        // first add left most nodes to a set
        let left_path_ids = self.get_left_path_id().unwrap();
        for path in left_path_ids {
            for h in path {
                node_hashes.insert(h);
            }
        }
        // then add the right-most nodes
        let right_path_ids = self.get_right_most_path_id();
        for path in right_path_ids {
            for h in path {
                node_hashes.insert(h);
            }
        }
        for node_h in self.nodes.keys().copied().collect::<BTreeSet<H256>>().difference(&node_hashes) {
            self.nodes.remove(node_h);
        }
    }

    pub fn prune_tree_before_version_bound(&mut self, before_ver: u32) {
        // a set to collect the kept node's hash values
        let mut node_hashes = BTreeSet::<H256>::new();
        // first add left most nodes to a set
        let left_path_ids = self.get_left_path_id().unwrap();
        for path in left_path_ids {
            for h in path {
                node_hashes.insert(h);
            }
        }
        // traverse each nodes in the tree and keep the child with index >= lower_bound(node.keys, before_ver)
        let root = self.nodes.get(&self.root_h).unwrap();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            if let CDCNode::NonLeaf(internal) = node {
                let arr: &Vec<InnerVer> = &internal.keys.iter().map(|elem| InnerVer(elem)).collect();
                let low_idx = general_lower_bound(arr, before_ver); // find the lower_bound index
                for child_h in &internal.child_hashes[low_idx..] {
                    node_hashes.insert(*child_h);
                    let child_node = self.nodes.get(child_h).unwrap();
                    queue.push_back(child_node);
                }
            }
        }
        for node_h in self.nodes.keys().copied().collect::<BTreeSet<H256>>().difference(&node_hashes) {
            self.nodes.remove(node_h);
        }
    }

    pub fn print_tree(&self) {
        println!("min keep: {:?}", self.min_keep_left_nodes);
        let root = self.nodes.get(&self.root_h).unwrap();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            println!("h: {:?}, node: {:?}, num of keys: {}, node_size: {}", node.to_digest(), node, node.num_of_keys(), node.get_size());
            match node {
                CDCNode::Leaf(_) => {
                },
                CDCNode::NonLeaf(node) => {
                    for child_hash in &node.child_hashes {
                        let child_node_h = self.nodes.get(child_hash);
                        if let Some(child_node_inner_h) = child_node_h {
                            queue.push_back(child_node_inner_h);
                        }
                    }
                }
            }
        }
    }

    // return the hash values of each level of the right-most tree path
    pub fn get_right_most_path_id(&self) -> Vec<Vec<H256>> {
        let mut path: Vec<Vec<H256>> = Vec::new();
        let mut cur_node = self.load_node(&self.get_root_hash()).unwrap();
        while !cur_node.is_leaf() {
            path.push(vec![cur_node.to_digest()]);
            let internal_node = cur_node.get_internal().unwrap();
            cur_node = self.load_node(internal_node.child_hashes.last().unwrap()).unwrap();
        }
        // handle leaf
        path.push(vec![cur_node.to_digest()]);
        return path;
    }

    // return min_keep_left nodes number of hash values for each level of the left tree path
    pub fn get_left_path_id(&self) -> Result<Vec<Vec<H256>>> {
        let mut path: Vec<Vec<H256>> = Vec::new();
        path.push(vec![self.get_root_hash()]);
        let height = self.min_keep_left_nodes.len();
        for (idx, n) in self.min_keep_left_nodes.iter().rev().enumerate() {
            if idx == height - 1 {
                continue;
            }
            let mut child_hashes: Vec<H256> = Vec::new();
            for node_hash in path.last().unwrap() { // assume that the nodes in this level can always be pointed by some nodes in the upper level (path.last())
                let node = self.load_node(node_hash).unwrap();
                let internal = node.get_internal().unwrap();
                child_hashes.extend(&internal.child_hashes);
                if child_hashes.len() >= *n as usize {
                    // enough child hashes, no need to iterate the rest of the node in this level
                    break;
                }
            }
            if child_hashes.len() < *n as usize {
                return Err(anyhow!("Error, this level's nodes cannot be pointed by some nodes in the upper level"));
            }
            // keep the first n child hash values
            child_hashes.truncate(*n as usize);
            path.push(child_hashes);
        }
        return Ok(path);
    }

    pub fn load_node(&self, node_hash: &H256) -> Option<&CDCNode> {
        self.nodes.get(node_hash)
    }

    pub fn move_to_persist_node(&mut self, node_hash: H256) -> Option<PersistCDCNode> {
        let r = self.nodes.remove(&node_hash);
        if let Some(node) = r {
            let persist_node = PersistCDCNode::from_cdc_node(node);
            return Some(persist_node);
        }
        return None;
    }

    pub fn get_root_hash(&self) -> H256 {
        self.root_h
    }

    // given the left version l and right version r, search the results and get the proof
    pub fn search_range(&self, l: u32, r: u32) -> (Option<Vec<VerObject>>, CDCRangeProof) {
        // init a range proof
        let mut proof = CDCRangeProof::default();
        if self.nodes.len() == 0 {
            // empty tree
            return (None, proof);
        } else {
            // load root node
            let root_hash = self.get_root_hash();
            let node = self.load_node(&root_hash).unwrap();
            // init a result value vector
            let mut value_vec = Vec::<VerObject>::new();
            // create a queue to help traverse the tree
            let mut queue = VecDeque::<&CDCNode>::new();
            // push the root node to the queue
            queue.push_back(node);
            // some counter to help determine the number of nodes in the level
            let mut prev_cnt = 1;
            let mut cur_cnt = 0;
            // a temporary proof for the current level
            let mut cur_level_proof = Vec::<((usize, usize), CDCNode)>::new();
            // traverse the tree in a while loop until the queue is empty
            while !queue.is_empty() {
                let cur_node = queue.pop_front().unwrap();
                prev_cnt -= 1; // decrease the node counter of the previous level
                if !cur_node.is_leaf() {
                    // the node is an internal node, retrieve the reference of the internal node
                    let internal = cur_node.get_internal().unwrap();
                    // given the lb and ub, get the position range of the child nodes
                    let (start_idx, end_idx) = cur_node.search_node_idx_range(l, r);
                    // update the node counter for the level
                    cur_cnt += end_idx - start_idx + 1;
                    // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                    cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                    // add the corresponding child nodes to the queue
                    for idx in start_idx ..= end_idx {
                        let child_hash = internal.child_hashes[idx];
                        let child_node = self.load_node(&child_hash).unwrap();
                        queue.push_back(child_node);
                    }
                } else {
                    // the node is a leaf node, retrieve the reference of the leaf node
                    let leaf = cur_node.get_leaf().unwrap();
                    // get the position range of the leaf node
                    let (start_idx, end_idx) = cur_node.search_node_idx_range(l, r);
                    // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                    cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                    // add the corresponding searched entries to the value_vec
                    for id in start_idx ..= end_idx {
                        let data = leaf.objs[id].clone();
                        value_vec.push(data);
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

    pub fn serialize_params(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        // write exp_fanout
        bytes.extend((self.exp_fanout as u8).to_be_bytes()); // change exp_fanout as u8, maximum exp_fanout should be 255
        // write gear_hash_level
        bytes.extend((self.gear_hash_level as u8).to_be_bytes()); // change gear_hash_level as u8
        // write max_node_capacity
        bytes.extend((self.max_node_capacity as u8).to_be_bytes()); // change max_node_capacity as u8
        // record len of min_keep_left_nodes
        let n = self.min_keep_left_nodes.len();
        bytes.extend((n as u8).to_be_bytes());  // assume maximum height should be 255
        // write min_keep_left_nodes
        for keep_left in &self.min_keep_left_nodes {
            bytes.extend(keep_left.to_be_bytes());
        }
        return bytes;
    }

    pub fn height(&self) -> usize {
        let mut tree_height = 1;
        let mut node = self.nodes.get(&self.root_h).unwrap();
        while !node.is_leaf() {
            let internal = node.get_internal().unwrap();
            let left_most_child = internal.child_hashes.first().unwrap();
            node = self.nodes.get(left_most_child).unwrap();
            tree_height += 1;
        }
        return tree_height;
    }
}

fn combine_min_keep_left_nodes(left_tree: &CDCTree, right_tree: &CDCTree) -> Vec<u8> {
    let mut combined = Vec::<u8>::new();
    let left_tree_height = left_tree.min_keep_left_nodes.len();
    let right_tree_height = right_tree.min_keep_left_nodes.len();
    // handle min_keep_left_nodes, if there is a new root generated during the merge, handle the root node's min_keep_left_nodes later
    combined.extend(&left_tree.min_keep_left_nodes);
    if right_tree_height > left_tree_height {
        // right tree is higher than left tree, copy the rest of the elements in the right tree's min_keep_left_nodes
        combined.extend(&right_tree.min_keep_left_nodes[left_tree_height..]);
    }
    return combined;
}

fn collect_merged_path_ids(left_tree: &CDCTree, right_tree: &CDCTree) -> Result<Vec<Vec<H256>>> {
    // init the merged path ids
    let mut merged_path_ids: Vec<Vec<H256>> = Vec::new();
    // get the left tree's right-most path and the right tree's min_keep_left node path
    let mut left_tree_path_ids = left_tree.get_right_most_path_id();
    let mut right_tree_path_ids = right_tree.get_left_path_id()?;
    let left_tree_height = left_tree.min_keep_left_nodes.len();
    let right_tree_height = right_tree.min_keep_left_nodes.len();
    // padding empty vector for lower tree
    if left_tree_height < right_tree_height {
        let diff = right_tree_height - left_tree_height;
        let mut temp_path_ids: Vec<Vec<H256>> = vec![vec![]; diff];
        temp_path_ids.append(&mut left_tree_path_ids);
        left_tree_path_ids = temp_path_ids;
    } else if left_tree_height > right_tree_height {
        let diff = left_tree_height - right_tree_height;
        let mut temp_path_ids: Vec<Vec<H256>> = vec![vec![]; diff];
        temp_path_ids.append(&mut right_tree_path_ids);
        right_tree_path_ids = temp_path_ids;
    }

    let iter = left_tree_path_ids.into_iter().zip(right_tree_path_ids.into_iter());
    for (left_level, right_level) in iter.rev() {
        // reverse the iterator to change the direction from leaf to root
        let mut level_node_ids: Vec<H256> = Vec::new();
        level_node_ids.extend(left_level);
        level_node_ids.extend(right_level);
        merged_path_ids.push(level_node_ids);
    }

    Ok(merged_path_ids)
}

fn find_cut_points(cdc_hash: &mut CDCHash, input_hashes: Vec<H256>) -> Vec<i32> {
    let mut cut_points: Vec<i32> = vec![-1]; //add first split point -1 as the starting point
    input_hashes.iter().enumerate().for_each(|(i, h)| {
        let r = cdc_hash.generate_cut_point(h.as_bytes());
        if let CDCResult::PatternFound | CDCResult::ReachCapacity = r {
            if i != 0 && i != (input_hashes.len() - 1) {
                cut_points.push(i as i32); // should cut at i-th element if it is not the first or last index
            }
        }
    });
    cut_points.push(input_hashes.len() as i32 -1); // add last split point len - 1
    return cut_points;
}

fn merge_leaf_level_nodes(merged_tree: &mut CDCTree, merged_path: &Vec<Vec<H256>>, cdc_hash: &mut CDCHash) -> Vec<(u32, H256)> {
    // first collect the objs in this level and remove the old leaf nodes
    let mut collect_obj: Vec<VerObject> = Vec::new();
    for leaf_h in &merged_path[0] {
        let node = merged_tree.load_node(leaf_h).unwrap();
        let leaf = node.get_leaf().unwrap();
        collect_obj.extend(leaf.objs.clone());
        merged_tree.nodes.remove(leaf_h);
    }
    // find pattern in collect_obj
    let input_hashes: Vec<H256> = collect_obj.iter().map(|obj| obj.to_digest()).collect();
    let cut_points: Vec<i32> = find_cut_points(cdc_hash, input_hashes);
    // generate leaf nodes
    let mut pushed_up_entries: Vec<(u32, H256)> = vec![];
    let cut_point_num = cut_points.len();
    for i in 1..=cut_point_num-1 {
        let mut new_leaf = CDCNode::Leaf(CDCLeafNode::default());
        let start_idx = (cut_points[i-1] + 1) as usize;
        let end_idx = cut_points[i] as usize;
        new_leaf.batch_load_to_leaf(collect_obj[start_idx..=end_idx].to_vec());
        let node_hash = new_leaf.to_digest();
        merged_tree.root_h = node_hash;
        pushed_up_entries.push((new_leaf.get_node_last_key(), node_hash));
        merged_tree.nodes.insert(node_hash, new_leaf);
    }
    return pushed_up_entries;
}

fn find_pattern_and_split_nodes(merged_tree: &mut CDCTree, input_entries: Vec<(u32, H256)>, cdc_hash: &mut CDCHash) -> Vec<(u32, H256)> {
    // find pattern in collect_entries
    let input_hashes: Vec<H256> = input_entries.iter().map(|(_, h)| *h).collect();
    cdc_hash.reset_hasher();
    let cut_points = find_cut_points(cdc_hash, input_hashes);
    let mut temp_pushed_up_entries: Vec<(u32, H256)> = Vec::new();
    let cut_point_num = cut_points.len();
    for i in 1..=cut_point_num-1 {
        let mut new_node = CDCNode::NonLeaf(CDCInternalNode::default());
        let start_idx = (cut_points[i-1] + 1) as usize;
        let end_idx = cut_points[i] as usize;
        new_node.batch_load_to_internal(input_entries[start_idx..=end_idx].to_vec());
        let node_hash = new_node.to_digest();
        merged_tree.root_h = node_hash;
        temp_pushed_up_entries.push((new_node.get_node_last_key(), node_hash));
        merged_tree.nodes.insert(node_hash, new_node);
    }
    return temp_pushed_up_entries;
}

fn merge_internal_level_nodes(merged_tree: &mut CDCTree, merged_path: &Vec<Vec<H256>>, leaf_pushed_up: Vec<(u32, H256)>, cdc_hash: &mut CDCHash) {
    let mut pushed_up_entries = leaf_pushed_up;
    let mut num_of_levels = 1;
    for (level_id, path) in merged_path.iter().enumerate() {
        // iterate the merged_path from the first internal node
        if level_id == 0 {
            continue;
        }
        // collect this level's entries
        let mut collect_entries: Vec<(u32, H256)> = Vec::new();
        for node_h in path {
            let node = merged_tree.load_node(node_h).unwrap();
            let internal = node.get_internal().unwrap();
            let key_hash_iter = internal.keys.iter().zip(internal.child_hashes.iter());
            let key_hash_vec: Vec<(u32, H256)> = key_hash_iter.map(|(k, h)| (*k, *h)).collect();
            collect_entries.extend(key_hash_vec);
            merged_tree.nodes.remove(node_h); // remove the old internal node
        }
        // find the split hash merged_path[level_id-1][0]'s position from collect_entries -> left boundary_idx
        let left_searched_h = merged_path[level_id-1].first().unwrap();
        let right_searched_h = merged_path[level_id-1].last().unwrap();
        let mut left_boundary_idx = 0;
        let mut right_boundary_idx = 0;
        for (idx, (_, h)) in collect_entries.iter().enumerate() {
            if h == left_searched_h {
                left_boundary_idx = idx;
            }
            if h == right_searched_h {
                right_boundary_idx = idx + 1;
                break;
            }
        }
        if right_boundary_idx <= left_boundary_idx {
            // no found of right_searched h
            right_boundary_idx = left_boundary_idx + 1;
        }
        let mut temp_collect_entries = collect_entries[0..left_boundary_idx].to_vec();
        // insert pushed_up_entries to temp_collect_entries
        temp_collect_entries.extend(pushed_up_entries);
        // insert collect_entries[right_boundary_idx..] to temp_collect_entries
        temp_collect_entries.extend(collect_entries[right_boundary_idx..].to_vec());
        // find pattern in temp_collect_entries and split nodes
        let temp_pushed_up_entries = find_pattern_and_split_nodes(merged_tree, temp_collect_entries, cdc_hash);
        pushed_up_entries = temp_pushed_up_entries;
        num_of_levels += 1;
    }
    // handle the case when pushed_up_entries has more than 1 element, should form a new root node
    while pushed_up_entries.len() > 1 {
        // find pattern in pushed_up_entries and create nodes
        let temp_pushed_up_entries = find_pattern_and_split_nodes(merged_tree, pushed_up_entries, cdc_hash);
        pushed_up_entries = temp_pushed_up_entries;
        num_of_levels += 1;
    }
    let original_tree_height = merged_tree.min_keep_left_nodes.len();
    if num_of_levels > original_tree_height {
        // higher level is created, padding 1 to min_keep_left_nodes
        let diff = num_of_levels - original_tree_height;
        for _ in 0..diff {
            merged_tree.min_keep_left_nodes.push(1);
        }
    }
}

pub fn merge_two_cdc_trees(left_tree: CDCTree, right_tree: CDCTree) -> Result<CDCTree> {
    // check consistency of the left and right trees
    assert_eq!(left_tree.exp_fanout, right_tree.exp_fanout);
    assert_eq!(left_tree.gear_hash_level, right_tree.gear_hash_level);
    assert_eq!(left_tree.max_node_capacity, right_tree.max_node_capacity);
    let exp_fanout = left_tree.exp_fanout;
    let gear_hash_level = left_tree.gear_hash_level;
    let max_node_capacity = left_tree.max_node_capacity;
    let mut merged_tree = CDCTree::new(exp_fanout, gear_hash_level, max_node_capacity);
    // handle the min_keep_left_nodes for the merged tree
    merged_tree.min_keep_left_nodes =  combine_min_keep_left_nodes(&left_tree, &right_tree);
    // collect merged node ids from ***leaf level to the root level***
    let merged_path = collect_merged_path_ids(&left_tree, &right_tree)?;
    // move left tree's nodes and right tree's nodes to the merged tree
    merged_tree.nodes.extend(left_tree.nodes);
    merged_tree.nodes.extend(right_tree.nodes);

    let mut cdc_hash = CDCHash::new(merged_tree.exp_fanout, merged_tree.gear_hash_level, merged_tree.max_node_capacity);
    // handle leaf level node merging
    // pushed_up_entries is the possible pushed up (key, h) pairs from leaf level to the upper level
    let pushed_up_entries = merge_leaf_level_nodes(&mut merged_tree, &merged_path, &mut cdc_hash);
    merge_internal_level_nodes(&mut merged_tree, &merged_path, pushed_up_entries, &mut cdc_hash);
    Ok(merged_tree)
}

// reconstruct the range proof to the root digest 
pub fn reconstruct_cdc_range_proof(l: u32, r: u32, result: &Option<Vec<VerObject>>, proof: &CDCRangeProof) -> H256 {
    if result.is_none() && proof == &CDCRangeProof::default() {
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
                    println!("not match hash");
                    validate = false;
                    break;
                }
                // start another level by clearing the hashes
                next_level_hashes.clear();
            }
            // id of the result in the vector of the proof
            let mut leaf_id: usize = 0;
            for inner_proof in level_proof {
                let ((start_idx, end_idx), node) = &inner_proof;
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
                    for id in *start_idx..= *end_idx {
                        let ver_obj = &leaf.objs[id];
                        if &result[leaf_id] != ver_obj {
                            validate = false;
                            break;
                        }
                        leaf_id += 1;
                    }
                    // check the left-most verion
                    if leaf.objs[*start_idx].ver > l && *start_idx != 0 {
                        println!("left most check error");
                        validate = false;
                    }
                    // check the right-most version
                    if leaf.objs[*end_idx].ver < r && *end_idx != leaf.objs.len() - 1 {
                        println!("right most check error");
                        validate = false;
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

pub trait OutputOrderElement<K: PartialEq + PartialOrd + Debug>{
    fn get_element(&self,) -> &K;
}

pub fn general_lower_bound<E: OutputOrderElement<K>, K: PartialEq + PartialOrd + Debug>(arr: &[E], target_key: K) -> usize {
    if &target_key < arr[0].get_element() {
        // searched key is smaller than the first version
        return 0;
    } else if &target_key > arr.last().unwrap().get_element() {
        // searched key is larger than the last version
        return arr.len() - 1;
    } else {
        let mut low = 0usize;
        let mut high = arr.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if arr[mid].get_element() < &target_key {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }
}

impl OutputOrderElement<u32> for VerObject {
    fn get_element(&self,) -> &u32 {
        &self.ver
    }
}

struct InnerVer<'a>(&'a u32);
impl<'a> OutputOrderElement<u32> for InnerVer<'a> {
    fn get_element(&self,) -> &u32 {
        &self.0
    }
}

/* Proof of a range query, each level consist of a vector of nodes
 */
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct CDCRangeProof {
    pub levels: Vec<Vec<((usize, usize), CDCNode)>>, // the first two usize store the start_idx and end_idx of the searched entries in the node
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    #[test]
    fn test_cdc_hash() {
        let n = 300;
        let fanout = 4;
        let level = 1;

        let mut rng = StdRng::seed_from_u64(2);
        let hashes: Vec<H256> = (0..n).into_iter().map(|_| H256::random_using(&mut rng)).collect();
        let mut streaming_index: Vec<usize> = vec![];
        let max_capacity = 2usize.pow(level as u32) * fanout;
        let mut cdchash = CDCHash::new(fanout, level, max_capacity);
        for (i, h) in hashes.iter().enumerate() {
            let r = cdchash.generate_cut_point(h.as_bytes());
            match r
            {
                CDCResult::PatternFound => {
                    streaming_index.push(i);
                },
                CDCResult::ReachCapacity => {
                    streaming_index.push(i);
                },
                _ => {},
            }
        }

        for index in &mut streaming_index {
            *index = (*index+1) * 32;
        }
        println!("streaming: {:?}", streaming_index);
    }

    #[test]
    fn test_build_tree() {
        let n = 320;
        let fanout = 8;
        let level = 1;
        let max_capacity = 100;
        let mut rng = StdRng::seed_from_u64(2);
        let objs: Vec<VerObject> = (0..n).into_iter().map(|i| {
            VerObject {
                ver: 2*i as u32,
                value: StateValue(H256::random_using(&mut rng))
            }
        }).collect();

        let mut tree = CDCTree::new(fanout, level, max_capacity);
        tree.bulk_load(objs);
        tree.print_tree();
    }

    #[test]
    fn test_prune_cdc_tree() {
        let fanout = 8;
        let level = 1;
        let max_capacity = 8;
        let mut rng = StdRng::seed_from_u64(2);
        let tree_objs: Vec<VerObject> = (1..80).into_iter().map(|i| {
            VerObject {
                ver: i as u32,
                value: StateValue(H256::random_using(&mut rng))
            }
        }).collect();
        let mut tree = CDCTree::new(fanout, level, max_capacity);
        tree.bulk_load(tree_objs);
        tree.print_tree();
        println!("============================");
        tree.prune_tree_before_version_bound(60);
        println!("pruned tree");
        for (h, node) in &tree.nodes {
            println!("h: {:?}, node: {:?}", h, node);
        }
        let mut writer = CDCTreeWriter::new("tree.dat");
        let tree_addr = writer.write_tree(&mut tree.clone());
        writer.finalize();
        let mut reader = CDCTreeReader::new("tree.dat");
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        let load_tree = reader.read_tree_at(tree_addr, run_id, &mut cache_manager).unwrap();
        assert_eq!(load_tree, tree);
    }
    #[test]
    fn test_merge_cdc_tree() {
        let fanout = 8;
        let level = 1;
        let max_capacity = 8;
        let mut rng = StdRng::seed_from_u64(2);
        let left_tree_objs: Vec<VerObject> = (1..567).into_iter().map(|i| {
            VerObject {
                ver: i as u32,
                value: StateValue(H256::random_using(&mut rng))
            }
        }).collect();
        let mut left_tree = CDCTree::new(fanout, level, max_capacity);
        left_tree.bulk_load(left_tree_objs.clone());
        println!("============================");
        let right_tree_objs: Vec<VerObject> = (567..1000).into_iter().map(|i| {
            VerObject {
                ver: i as u32,
                value: StateValue(H256::random_using(&mut rng))
            }
        }).collect();
        let mut right_tree = CDCTree::new(fanout, level, max_capacity);
        right_tree.bulk_load(right_tree_objs.clone());
        println!("============================");
        println!("merge tree");
        let merged_tree = merge_two_cdc_trees(left_tree, right_tree).unwrap();
        for i in 1..1000 {
            let (r, _) = merged_tree.search_range(i, i);
            let results = r.unwrap();
            let mut flag = false;
            for obj in results {
                if obj.ver == i {
                    flag = true;
                    break;
                }
            }
            assert!(flag);
        }
        println!("merged tree root: {:?}", merged_tree.root_h);
        println!("============================");
        let mut total_obj: Vec<VerObject> = Vec::new();
        total_obj.extend(left_tree_objs);
        total_obj.extend(right_tree_objs);
        let mut total_tree = CDCTree::new(fanout, level, max_capacity);
        total_tree.bulk_load(total_obj);
        println!("total tree root: {:?}", total_tree.root_h);
    }
    
    #[test]
    fn test_ser_deser_tree() {
        let fanout = 4;
        let level = 1;
        let max_capacity = 120;
        let mut rng = StdRng::seed_from_u64(1);
        let left_tree_objs: Vec<VerObject> = (1..20).into_iter().map(|i| {
            VerObject {
                ver: i as u32,
                value: StateValue(H256::random_using(&mut rng))
            }
        }).collect();
        let mut left_tree = CDCTree::new(fanout, level, max_capacity);
        left_tree.bulk_load(left_tree_objs);
        // left_tree.print_tree();

        let mut writer = CDCTreeWriter::new("tree.dat");
        let mut addr_vec = Vec::<u64>::new();
        let duplicate_tree_num = 30;
        for _ in 0..duplicate_tree_num {
            let tree_addr = writer.write_tree(&mut left_tree.clone());
            addr_vec.push(tree_addr);
        }

        writer.finalize();
        println!("writer offset: {}", writer.new_tree_start_offset);
        let mut reader = CDCTreeReader::new("tree.dat");
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..duplicate_tree_num {
            let tree_addr = addr_vec[i];
            let load_tree = reader.read_tree_at(tree_addr, run_id, &mut cache_manager).unwrap();
            assert_eq!(load_tree, left_tree);
        }
        // assert_eq!(load_tree, left_tree);
        // let l = 35;
        // let r = 80;
        // let (result, proof) = reader.search_range_at_tree_addr(tree_addr3, run_id, l, r, &mut cache_manager);
        // let root_h = load_tree.get_root_hash();
        // let h = reconstruct_range_proof(l, r, &result, &proof);
        // assert_eq!(root_h, h);
    }

    #[test]
    fn test_ser_deser_ver_obj() {
        let ver_obj = VerObject::new(1, StateValue(H256::from_low_u64_be(1)));
        let bytes = ver_obj.to_bytes();
        assert_eq!(bytes.len(), 36);
        let deser_obj = VerObject::from_bytes(&bytes);
        assert_eq!(ver_obj, deser_obj);
    }

    #[test]
    fn test_lower_bound() {
        let v = vec![10, 20, 30, 40, 50];
        let inner_v: &Vec<InnerVer> = &v.iter().map(|elem| InnerVer(elem)).collect();
        // let search_key = 5;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 0);
        let search_key = 10;
        let r = general_lower_bound(&inner_v, search_key);
        assert_eq!(r, 0);
        // let search_key = 15;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 1);
        // let search_key = 20;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 1);
        // let search_key = 25;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 2);
        // let search_key = 30;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 2);
        // let search_key = 35;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 3);
        // let search_key = 40;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 3);
        // let search_key = 45;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 3);

        // let v = vec![10, 20, 30, 40, 50];
        // let inner_v: &Vec<InnerVer> = &v.iter().map(|elem| InnerVer(elem)).collect();
        // let search_key = 5;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 0);
        // let search_key = 10;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 0);
        // let search_key = 15;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 1);
        // let search_key = 20;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 1);
        // let search_key = 25;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 2);
        // let search_key = 30;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 2);
        // let search_key = 35;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 3);
        // let search_key = 40;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 3);
        // let search_key = 45;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 4);
        // let search_key = 50;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 4);
        // let search_key = 55;
        // let r = general_lower_bound(&inner_v, search_key);
        // assert_eq!(r, 4);

        // let mut leaf = CDCNode::Leaf(CDCLeafNode::default());
        // let vers = vec![10, 20, 30, 40, 50];
        // for v in vers {
        //     let ver_obj = VerObject::new(v, StateValue(H256::from_low_u64_be(v as u64)));
        //     leaf.insert_to_leaf(ver_obj);
        // }

        // let (start, end) = leaf.search_node_idx_range(0, 2);
        // assert_eq!(start, 0);
        // assert_eq!(end, 0);

        // let (start, end) = leaf.search_node_idx_range(0, 10);
        // assert_eq!(start, 0);
        // assert_eq!(end, 1);

        // let (start, end) = leaf.search_node_idx_range(0, 35);
        // assert_eq!(start, 0);
        // assert_eq!(end, 3);

        // let (start, end) = leaf.search_node_idx_range(0, 55);
        // assert_eq!(start, 0);
        // assert_eq!(end, 4);

        // let (start, end) = leaf.search_node_idx_range(15, 35);
        // assert_eq!(start, 0);
        // assert_eq!(end, 3);

        // let (start, end) = leaf.search_node_idx_range(25, 55);
        // assert_eq!(start, 1);
        // assert_eq!(end, 4);
    }
}