pub(crate) mod hash;
pub mod nibbles;
pub mod proof;
pub mod read;
pub mod storage;
pub mod traits;
pub mod u4;
pub mod write;

pub mod prelude;
use anyhow::Context;
pub use prelude::*;
use serde::{Deserialize, Serialize};
use utils::types::{StateValue, AddrKey};
use std::collections::{BTreeMap, HashMap, VecDeque};
use rocksdb::{OptimisticTransactionDB, SingleThreaded, WriteBatchWithTransaction, WriteOptions};

#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Key(pub NibbleBuf);

impl AsNibbles for Key {
    fn as_nibbles(&self) -> Nibbles<'_> {
        self.0.as_nibbles()
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct TestTrie {
    roots: Vec<H256>,
    nodes: HashMap<Vec<u8>, Vec<u8>>,
}

impl NodeLoader<StateValue> for TestTrie {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let id_bytes = id.as_bytes().to_vec();
        let node_bytes = self.nodes.get(&id_bytes).cloned().context("Unknown node")?;
        let node: TrieNode<StateValue> = bincode::deserialize(&node_bytes).unwrap();
        Ok(node)
    }
}

impl NodeLoader<StateValue> for &'_ TestTrie {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let id_bytes = id.as_bytes().to_vec();
        let node_bytes = self.nodes.get(&id_bytes).cloned().context("Unknown node")?;
        let node: TrieNode<StateValue> = bincode::deserialize(&node_bytes).unwrap();
        Ok(node)
    }
}

impl TestTrie {
    pub fn new() ->  Self {
        Self {
            roots: vec![H256::default()],
            nodes: HashMap::new(),
        }
    }
    #[allow(dead_code)]
    pub fn apply(&mut self, apply: Apply<StateValue>) {
        self.roots.push(apply.root);
        let mut ser_nodes = HashMap::new();
        for (h, node) in apply.nodes.into_iter() {
            let h_bytes = h.as_bytes().to_vec();
            let node_bytes = bincode::serialize(&node).unwrap();
            ser_nodes.insert(h_bytes, node_bytes);
        }
        self.nodes.extend(ser_nodes);
    }

    pub fn get_latest_root(&self) -> H256 {
        let l = self.roots.len() - 1;
        self.roots[l]
    }

    pub fn get_roots_len(&self) -> usize {
        self.roots.len()
    }

    pub fn get_root_with_version(&self, version: u32) -> H256 {
        // version starts with 1
        match self.roots.get(version as usize) {
            Some(r) => *r,
            None => H256::default(),
        }
    }

    pub fn search(&self, addr_key: AddrKey) -> Option<StateValue> {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let read_v = read_trie_without_proof(&self, self.get_latest_root(), &k).unwrap();
        return read_v;
    }

    pub fn search_with_proof(&self, addr_key: AddrKey, version: u32) -> (Option<StateValue>, Proof) {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let (read_v, proof) = read_trie(&self, self.get_root_with_version(version), &k).unwrap();
        return (read_v, proof);
    }

    pub fn insert(&mut self, addr_key: AddrKey, value: StateValue) {
        let immut_ref = unsafe {
            (self as *const TestTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        ctx.insert(&k, value).unwrap();
        let changes = ctx.changes();
        self.apply(changes);
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, StateValue>) {
        let immut_ref = unsafe {
            (self as *const TestTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        for (addr_key, value) in inputs {
            let k = Key(NibbleBuf::from_addr_key(addr_key));
            ctx.insert(&k, value).unwrap();
        }
        let changes = ctx.changes();
        self.apply(changes);
    }
}

pub struct PersistTrie<'a> {
    pub db: &'a OptimisticTransactionDB<SingleThreaded>,
    pub block_cnt: u32,
    pub keep_latest: bool,
}

impl<'a> Drop for PersistTrie<'a> {
    // persist the meta data including the len of roots and hashes in roots
    fn drop(&mut self) {
        let meta_key = "meta".as_bytes();
        let roots_len = self.block_cnt as u32;
        let meta_bytes = roots_len.to_be_bytes().to_vec();
        let tx = self.db.transaction();
        tx.put(meta_key, &meta_bytes).unwrap();
        tx.commit().unwrap();
    }
}

impl<'a> PersistTrie<'a> {
    // init a persist trie with the db reference
    pub fn new(db: &'a OptimisticTransactionDB<SingleThreaded>, keep_latest: bool) -> Self {
        Self {
            db,
            block_cnt: 0,
            keep_latest,
        }
    }
    // load the trie from the db
    pub fn open(db: &'a OptimisticTransactionDB<SingleThreaded>, keep_latest: bool) -> Self {
        let meta_key = "meta".as_bytes();
        let mut block_cnt = 0u32;
        match db.get(meta_key).unwrap() {
            Some(r) => {
                // load the len of hash vec
                let len = u32::from_be_bytes(r[0..4].try_into().unwrap());
                block_cnt = len;
            },
            None => {}
        }
        Self {
            db,
            block_cnt,
            keep_latest,
        }
    }

    pub fn apply(&mut self, apply: Apply<StateValue>) {
        self.block_cnt += 1;
        // self.roots.insert(block_id, apply.root);
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(false);
        write_opts.disable_wal(true);
        let mut batch = WriteBatchWithTransaction::default();
        // let tx = self.db.transaction();
        // before persist all nodes in apply to the db, first collect node hashes of the updated reference counter to a hashset
        let child_node_hash = self.collect_node_hash_of_reference_counter(&apply);
        // update node's reference counter
        for (node_hash, cnt) in child_node_hash {
            let mut reference_counter = self.get_node_reference_counter(node_hash);
            reference_counter += cnt;
            let reference_counter_key = Self::reference_counter_key(&node_hash);
            batch.put(&reference_counter_key, &reference_counter.to_be_bytes());
            // tx.put(&reference_counter_key, &reference_counter.to_be_bytes()).unwrap();
        }

        for (k, v) in apply.nodes {
            let v_bytes = bincode::serialize(&v).unwrap();
            batch.put(&k.as_bytes(), &v_bytes);
            // tx.put(&k.as_bytes(), &v_bytes).unwrap();
        }
        batch.put(&self.block_id_key(), apply.root.as_bytes()); // write (block_id, root)
        self.db.write_opt(batch, &write_opts).unwrap();
        // tx.commit().unwrap();
        if self.keep_latest {
            // should handle the pruning case
            let prune_block_id = self.block_cnt - 1;
            // println!("prune block id: {}", prune_block_id);
            self.prune(prune_block_id).unwrap();
        }
    }

    fn collect_node_hash_of_reference_counter(&self, apply: &Apply<StateValue>) -> HashMap<H256, u32> {
        let mut child_node_hash = HashMap::<H256, u32>::new();
        // add root node's hash
        child_node_hash.insert(apply.root, 1);
        // add all other child nodes
        for (_, node) in &apply.nodes {
            match node {
                TrieNode::Extension(extension_node) => {
                    let child_hash = extension_node.child;
                    match child_node_hash.get_mut(&child_hash) {
                        Some(cnt) => { *cnt += 1; },
                        None => {
                            child_node_hash.insert(child_hash, 1);
                        },
                    }
                },
                TrieNode::Branch(branch_node) => {
                    for c_h in branch_node.children {
                        if let Some(child_hash) = c_h {
                            match child_node_hash.get_mut(&child_hash) {
                                Some(cnt) => { *cnt += 1; },
                                None => {
                                    child_node_hash.insert(child_hash, 1);
                                },
                            }
                        }
                    }
                },
                _ => {},
            }
        }
        return child_node_hash;
    }

    fn reference_counter_key(node_hash: &H256) -> Vec<u8> {
        let mut r = Vec::<u8>::new();
        let prefix = b"cnt";
        r.extend(prefix);
        r.extend_from_slice(node_hash.as_bytes());
        return  r;
    }

    fn block_id_key(&self) -> Vec<u8> {
        let mut r = Vec::<u8>::new();
        let prefix: &[u8; 2] = b"id";
        r.extend(prefix);
        r.extend_from_slice(&self.block_cnt.to_be_bytes());
        return r;
    }

    fn block_ver_key(version: u32) -> Vec<u8> {
        let mut r = Vec::<u8>::new();
        let prefix: &[u8; 2] = b"id";
        r.extend(prefix);
        r.extend_from_slice(&version.to_be_bytes());
        return r;
    }

    fn get_node_reference_counter(&self, node_hash: H256) -> u32 {
        match self.db.get( Self::reference_counter_key(&node_hash)).unwrap() {
            Some(result_bytes) => {
                let reference_counter: u32 = u32::from_be_bytes(result_bytes.try_into().unwrap());
                return reference_counter;
            },
            None => {
                return 0;
            }
        }
    }

    pub fn get_latest_root(&self) -> (u32, H256) {
        let block_id = self.block_cnt;
        match self.db.get(&self.block_id_key()).unwrap() {
            Some(hash_bytes) => {
                let h = H256::from_slice(&hash_bytes);
                return (block_id, h);
            },
            None => {
                return (block_id, H256::default());
            }
        };
    }

    pub fn get_roots_len(&self) -> usize {
        self.block_cnt as usize
    }

    pub fn get_root_with_version(&self, version: u32) -> (u32, H256) {
        // version starts with 1
        match self.db.get(&Self::block_ver_key(version)).unwrap() {
            Some(hash_bytes) => {
                let h = H256::from_slice(&hash_bytes);
                return (version, h);
            },
            None => {
                return (0, H256::default());
            }
        }
    }

    pub fn search(&self, addr_key: AddrKey) -> Option<StateValue> {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let read_v = read_trie_without_proof(&self, self.get_latest_root().1, &k).unwrap();
        return read_v;
    }

    pub fn search_with_proof(&self, addr_key: AddrKey, version: u32) -> (Option<StateValue>, Proof) {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let (read_v, proof) = read_trie(&self, self.get_root_with_version(version).1, &k).unwrap();
        return (read_v, proof);
    }

    pub fn insert(&mut self, addr_key: AddrKey, value: StateValue) {
        let immut_ref = unsafe {
            (self as *const PersistTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root().1);
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        ctx.insert(&k, value).unwrap();
        let changes = ctx.changes();
        self.apply(changes);

    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, StateValue>) {
        let immut_ref = unsafe {
            (self as *const PersistTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root().1);
        for (addr_key, value) in inputs {
            let k = Key(NibbleBuf::from_addr_key(addr_key));
            ctx.insert(&k, value).unwrap();
        }
        let changes = ctx.changes();
        self.apply(changes);
    }

    pub fn print_latest_tree(&self, ) {
        let (_, h) = self.get_latest_root();
        let mut queue = VecDeque::<H256>::new();
        queue.push_back(h);
        while !queue.is_empty() {
            let node_hash = queue.pop_front().unwrap();
            let reference_counter = self.get_node_reference_counter(node_hash);
            let node = self.load_node(node_hash).unwrap();
            println!("node_hash: {:?}, ref: {}, node: {:?}", node_hash, reference_counter, node);
            match node {
                TrieNode::Extension(extension_node) => {
                    let child_hash = extension_node.child;
                    queue.push_back(child_hash);
                },
                TrieNode::Branch(branch_node) => {
                    for c_h in branch_node.children {
                        if let Some(child_hash) = c_h {
                            queue.push_back(child_hash);
                        }
                    }
                },
                TrieNode::Leaf(_) => {},
            }
        }
    }
 
    pub fn prune(&mut self, block_id: u32) -> Result<()> {
        let (_, root_h) = self.get_root_with_version(block_id);
        self.prune_trie_path(root_h);
        // compact lsm-tree levels
        // self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        return Ok(());
    }

    pub fn flush(&self) {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
    }

    fn prune_trie_path(&self, trie_root: H256) {
        if trie_root != H256::default() {
            // init a queue for collection removed node's hash
            let mut queue = VecDeque::<H256>::new();
            queue.push_back(trie_root);
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(false);
            write_opts.disable_wal(true);
            let mut batch = WriteBatchWithTransaction::default();
            // create a db transaction for node deletion
            // let tx = self.db.transaction();
            while !queue.is_empty() {
                let node_hash = queue.pop_front().unwrap();
                let mut reference_counter = self.get_node_reference_counter(node_hash);
                // deduct reference counter
                if reference_counter >= 1 {
                    reference_counter -= 1;
                    if reference_counter == 0 {
                        let node = self.load_node(node_hash).unwrap();
                        match node {
                            TrieNode::Extension(extension_node) => {
                                let child_hash = extension_node.child;
                                queue.push_back(child_hash);
                            },
                            TrieNode::Branch(branch_node) => {
                                for c_h in branch_node.children {
                                    if let Some(child_hash) = c_h {
                                        queue.push_back(child_hash);
                                    }
                                }
                            },
                            TrieNode::Leaf(_) => {},
                        }
                        // remove node and hash's reference counter
                        let reference_counter_key = Self::reference_counter_key(&node_hash);
                        batch.delete(&reference_counter_key);
                        batch.delete(&node_hash.as_bytes());
                        // tx.delete(&reference_counter_key).unwrap();
                        // tx.delete(&node_hash.as_bytes()).unwrap();
                    } else {
                        // update the node's reference counter
                        let reference_counter_key = Self::reference_counter_key(&node_hash);
                        // tx.put(&reference_counter_key, &reference_counter.to_be_bytes()).unwrap();
                        batch.put(&reference_counter_key, &reference_counter.to_be_bytes());
                    }
                }
            }
            // tx.commit().unwrap();
            self.db.write_opt(batch, &write_opts).unwrap();
        }
    }
}

impl NodeLoader<StateValue> for PersistTrie<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let node_byte = self.db.get(&id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<StateValue> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

impl NodeLoader<StateValue> for &'_ PersistTrie<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let node_byte = self.db.get(&id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<StateValue> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

pub fn verify_trie_proof(key: &Key, value: Option<StateValue>, root_h: H256, proof: &Proof) -> bool {
    let mut error_flag = true;
    if root_h != proof.root_hash() {
        error_flag = false;
    }

    match value {
        Some(v) => {
            if proof.value_hash(key) != Some(v.to_digest()) {
                error_flag = false;
            }
        },
        None => {
            if proof.value_hash(key) != Some(H256::default()) {
                error_flag = false;
            }
        }
    }
    return error_flag;
}

pub fn verify_with_addr_key(addr_key: &AddrKey, value: Option<StateValue>, root_h: H256, proof: &Proof) -> bool {
    let key = Key(NibbleBuf::from_addr_key(*addr_key));
    let mut error_flag = true;
    if root_h != proof.root_hash() {
        error_flag = false;
    }

    match value {
        Some(v) => {
            if proof.value_hash(&key) != Some(v.to_digest()) {
                error_flag = false;
            }
        },
        None => {
            if proof.value_hash(&key) != Some(H256::default()) {
                error_flag = false;
            }
        }
    }

    return error_flag;
} 

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use utils::types::{AddrKey, Address};
    use primitive_types::H160;
    use std::path::Path;
    use super::*;
    use rocksdb::{ColumnFamilyDescriptor, OptimisticTransactionDB, Options, SingleThreaded};

    #[test]
    fn test_rocksdb_lib() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let cf1 = ColumnFamilyDescriptor::new("cf1", Options::default());
        let cf2 = ColumnFamilyDescriptor::new("cf2", Options::default());
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(64 * 1024 * 1024);
        db_opts.create_missing_column_families(true);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open_cf_descriptors(&db_opts, path, vec![cf1, cf2]).unwrap();
        let start = std::time::Instant::now();
        let tx = db.transaction();
        tx.put_cf(&db.cf_handle("cf1").unwrap(), b"key1_cf1", b"value1_cf1").unwrap();
        tx.put_cf(&db.cf_handle("cf2").unwrap(), b"key1_cf2", b"value1_cf2").unwrap();
        tx.commit().unwrap();
        db.flush().unwrap();
        let elapse = start.elapsed().as_nanos();
        println!("insert: {:?}", elapse);
        let start = std::time::Instant::now();
        let r1 = db.get_cf(&db.cf_handle("cf1").unwrap(), b"key1_cf1").unwrap().unwrap();
        assert_eq!(r1, b"value1_cf1");
        let r2 = db.get_cf(&db.cf_handle("cf2").unwrap(), b"key1_cf2").unwrap().unwrap();
        assert_eq!(r2, b"value1_cf2");
        let r3 = db.get_cf(&db.cf_handle("cf1").unwrap(), b"key1_cf2").unwrap();
        assert!(r3.is_none());
        let r4 = db.get_cf(&db.cf_handle("cf2").unwrap(), b"key1_cf1").unwrap();
        assert!(r4.is_none());
        let elapse = start.elapsed().as_nanos();
        println!("search: {:?}", elapse);
    }

    #[test]
    fn test_trie_in_memory() {
        let num_of_contract = 100;
        let num_of_address = 100;
        let num_of_versions = 200;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }

        let mut trie = TestTrie::new();
        for i in 1..=num_of_versions {
            let v = StateValue(H256::from_low_u64_be(i));
            let mut map = BTreeMap::new();
            for key in &keys {
                map.insert(*key, v);
            }
            trie.batch_insert(map);
        }
        println!("finish insert");
        let latest_v = StateValue(H256::from_low_u64_be(num_of_versions));
        for key in &keys {
            let v = trie.search(*key).unwrap();
            assert_eq!(v, latest_v);
            for i in 1..=num_of_versions {
                let (v, p) = trie.search_with_proof(*key, i as u32);
                let current_v = StateValue(H256::from_low_u64_be(i));
                assert_eq!(current_v, v.unwrap());
                let b = verify_with_addr_key(key, v, trie.get_root_with_version(i as u32), &p);
                assert!(b);
            }
        }
    }

    #[test]
    fn test_mpt_reference_counter() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(32 * 1024 * 1024);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open(&db_opts, path).unwrap();
        {
            let num_of_contract = 2;
            let num_of_address = 5;
            let num_of_versions = 5;
            let mut rng = StdRng::seed_from_u64(1);
            let mut keys = Vec::<AddrKey>::new();
            for _ in 1..=num_of_contract {
                let addr = Address(H160::random_using(&mut rng));
                for _ in 1..=num_of_address {
                    let addr_key = AddrKey::new(addr, H256::random_using(&mut rng).into());
                    keys.push(addr_key);
                }
            }
            let mut trie = PersistTrie::new(&db, false);
            for i in 1..=num_of_versions {
                println!("block {}", i);
                let v = StateValue(H256::from_low_u64_be(i));
                for key in &keys {
                    trie.insert(*key, v);
                }
            }
            println!("block cnt: {}", trie.block_cnt);
            for i in 1..=trie.block_cnt {
                let (block_id, h) = trie.get_root_with_version(i);
                println!("{:?} {:?}", block_id, h);
            }
            // println!("{:?}", trie.roots);
            // trie.prune((0, 5)).unwrap();
            // println!("after: {:?}", trie.roots);
        }
        let trie = PersistTrie::open(&db, false);
        println!("block cnt: {}", trie.block_cnt);
        for i in 1..=trie.block_cnt {
            let (block_id, h) = trie.get_root_with_version(i);
            println!("{:?} {:?}", block_id, h);
        }
        // println!("after persist");
        // println!("roots: {:?}", trie.roots);
    }

    #[test]
    fn test_prune_mpt_disk() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(32 * 1024 * 1024);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open(&db_opts, path).unwrap();
        let num_of_contract = 10;
        let num_of_address = 2;
        let num_of_versions = 10;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }
        let mut trie = PersistTrie::new(&db, true);
        let mut cnt = 0;
        for i in 1..=num_of_versions {
            let v = StateValue(H256::from_low_u64_be(i));
            for key in &keys {
                cnt += 1;
                println!("block: {}", cnt);
                trie.insert(*key, v);
            }
        }
    }

    #[test]
    fn test_disk_trie() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(32 * 1024 * 1024);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open(&db_opts, path).unwrap();
        let num_of_contract = 100;
        let num_of_address = 100;
        let num_of_versions = 100;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }

        {
            let mut trie = PersistTrie::new(&db, false);
            let start = std::time::Instant::now();
            for i in 1..=num_of_versions {
                let v = StateValue(H256::from_low_u64_be(i));
                let mut map = BTreeMap::new();
                for key in &keys {
                    map.insert(*key, v);
                }
                trie.batch_insert(map);
            }
            let elapse = start.elapsed().as_nanos();
            println!("avg insert time: {}", elapse / (num_of_address * num_of_contract * num_of_versions) as u128);
            println!("finish insert");
        }
        
        let trie = PersistTrie::open(&db, false);
        let mut search_latest = 0;
        let mut search_prove = 0;
        let latest_v = StateValue(H256::from_low_u64_be(num_of_versions));
        for key in &keys {
            let start = std::time::Instant::now();
            let v = trie.search(*key).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(v, latest_v);
            for i in 1..=num_of_versions {
                let start = std::time::Instant::now();
                let (v, p) = trie.search_with_proof(*key, i as u32);
                let b = verify_with_addr_key(key, v, trie.get_root_with_version(i as u32).1, &p);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                let current_v = StateValue(H256::from_low_u64_be(i));
                let read_v = v.unwrap();
                if current_v != read_v {
                    println!("key: {:?}, true v: {:?}, read_v: {:?}", key, current_v, read_v);
                }
                // assert_eq!(current_v, v.unwrap());
                assert!(b);
            }
        }
        println!("search latest: {}", search_latest / (num_of_address * num_of_contract) as u128);
        println!("search prove: {}", search_prove / (num_of_address * num_of_contract * num_of_versions) as u128);
    }
}