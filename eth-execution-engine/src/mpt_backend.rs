use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use utils::{disk_usage_check_directory_storage, MemCost};
use utils::types::{Address, StateKey, StateValue, AddrKey};
use super::tx_executor::Backend;
use std::collections::BTreeMap;
use anyhow::Result;
use rocksdb::{OptimisticTransactionDB, SingleThreaded};
use patricia_trie::PersistTrie;

pub struct MPTExecutorBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: PersistTrie<'a>,
    pub path: &'a str,
}

impl<'a> MPTExecutorBackend<'a> {
    pub fn new(db: &'a OptimisticTransactionDB<SingleThreaded>, keep_latest: bool, path: &'a str) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: PersistTrie::new(&db, keep_latest),
            path,
        }
    }
}

impl<'a> Backend for MPTExecutorBackend<'a> {
    fn get_code(&self, acc_address: Address) -> Result<Code> {
        if self.code_map.contains_key(&acc_address) {
            Ok(self.code_map.get(&acc_address).unwrap().clone())
        } else {
            Ok(Code::default())
        }
    }

    fn get_nonce(&self, acc_address: Address) -> Result<Nonce> {
        if self.nonce_map.contains_key(&acc_address) {
            return Ok(self.nonce_map.get(&acc_address).unwrap().clone());
        } else {
            return Ok(Nonce::default());
        }
    }

    fn get_value(&self, acc_address: Address, key: StateKey) -> Result<StateValue> {
        let addr_key = AddrKey::new(acc_address, key);
        let v = self.states.search(addr_key);
        match v {
            Some(v) => {
                Ok(v)
            },
            None => {
                return Ok(StateValue::default());
            }
        }
    }
}

impl<'a> BackendWriteTrait for MPTExecutorBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, _: u32) {
        self.states.insert(addr_key, v);
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, _: u32) {
        self.states.batch_insert(states);
    }

    fn set_acc_nonce(&mut self, contract_addr: &Address, contract_nonce: Nonce) {
        self.nonce_map.insert(*contract_addr, contract_nonce);
    }

    fn get_acc_nonce(&self, contract_addr: &Address) -> Nonce {
        match self.nonce_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Nonce::default()
            }
        }
    }

    fn set_acc_code(&mut self, contract_addr: &Address, contract_code: Code) {
        self.code_map.insert(*contract_addr, contract_code);
    }

    fn get_acc_code(&self, contract_addr: &Address) -> Code {
        match self.code_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Code::default()
            }
        }
    }

    fn memory_cost(&self,) -> MemCost {
        MemCost::new(0, 0, 0)
    }

    fn index_stucture_output(&self,) -> String {
        todo!()
    }

    fn flush(&mut self) {
        // self.states.flush();
    }

    fn commit(&mut self) {
        
    }

    fn print_in_mem_tree(&self) {
        
    }

    fn index_size(&self) -> usize {
        let disk_size = disk_usage_check_directory_storage(&self.path);
        let mem_size = self.memory_cost().size();
        return disk_size + mem_size;
    }
}

#[cfg(test)]
mod tests {
    use crate::send_tx::{create_deploy_tx, create_call_tx, ContractArg};
    use super::super::tx_executor::{exec_tx, test_batch_exec_tx};
    use super::super::common::tx_req::TxRequest;
    use super::*;
    use patricia_trie::verify_with_addr_key;
    // use patricia_trie::verify_with_addr_key;
    use rand::prelude::*;
    use primitive_types::H160;
    use rocksdb::Options;
    use utils::{compute_cole_size_breakdown, disk_usage_check_directory_storage};
    use std::path::Path;

    #[test]
    fn test_mpt_prune_backend() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(64 * 1024 * 1024);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open(&db_opts, path).unwrap();

        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = MPTExecutorBackend::new(&db, false, path);
        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let n = 100000;
        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = i % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
            requests.push(call_tx_req);
        }

        let block_size = 100;
        let blocks: Vec<Vec<TxRequest>> = requests.chunks(block_size).into_iter().map(|v| v.to_owned()).collect();
        let mut i = 1;
        let mut states = BTreeMap::<AddrKey, StateValue>::new();
        let start = std::time::Instant::now();
        for block in blocks {
            println!("block {}", i);
            let s = test_batch_exec_tx(block, caller_address, i, &mut backend);
            states.extend(s);
            i += 1;
        }
        let elapse = start.elapsed().as_nanos();
        println!("time: {}", elapse / n as u128);
        let iter = backend.states.db.iterator(rocksdb::IteratorMode::Start);
        println!("cnt: {:?}", iter.count());

        // println!("states size: {}", states.len());
        // println!("roots len: {}", backend.states.get_roots_len());
        // let mut search_latest = 0;
        // for (k, v) in states {
        //     let start = std::time::Instant::now();
        //     let read_v = backend.states.search(k).unwrap();
        //     let elapse = start.elapsed().as_nanos();
        //     search_latest += elapse;
        //     assert_eq!(read_v, v);
        // }
        // println!("search latest: {}", search_latest / n as u128);
        let before_flush_storage = disk_usage_check_directory_storage(path);
        println!("before flush storage: {}", before_flush_storage);
        drop(backend);
        let storage_size = compute_cole_size_breakdown(path);
        println!("storage size: {:?}", storage_size);
    }

    #[test]
    fn test_mpt_backend_in_disk() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_opts = Options::default();
        db_opts.set_db_write_buffer_size(64 * 1024 * 1024);
        db_opts.create_if_missing(true);
        let db = OptimisticTransactionDB::<SingleThreaded>::open(&db_opts, path).unwrap();

        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = MPTExecutorBackend::new(&db, false, path);
        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let n = 100000;
        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = i % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
            requests.push(call_tx_req);
        }
        
        let block_size = 100;
        let blocks: Vec<Vec<TxRequest>> = requests.chunks(block_size).into_iter().map(|v| v.to_owned()).collect();
        let mut i = 1;
        let mut states = BTreeMap::<AddrKey, StateValue>::new();
        let start = std::time::Instant::now();
        for block in blocks {
            let s = test_batch_exec_tx(block, caller_address, i, &mut backend);
            states.extend(s);
            i += 1;
        }
        let elapse = start.elapsed().as_nanos();
        println!("time: {}", elapse / n as u128);

        let latest_version = n as u32 / block_size as u32;
        println!("states size: {}", states.len());
        println!("roots len: {}", backend.states.get_roots_len());
        let mut search_latest = 0;
        let mut search_prove = 0;
        for (k, v) in states {
            let start = std::time::Instant::now();
            let read_v = backend.states.search(k).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(read_v, v);
            for version in 1..=latest_version {
                let start = std::time::Instant::now();
                let (read_v, p) = backend.states.search_with_proof(k, version);
                let b = verify_with_addr_key(&k, read_v, backend.states.get_root_with_version(version).1, &p);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                if b == false {
                    println!("false");
                }
                // println!("k: {:?}, version: {:?}, read_v: {:?}, b: {:?}", k, version, read_v, b);
                // assert!(b);
            }
        }
        println!("search latest: {}", search_latest / n as u128);
        println!("search prove: {}", search_prove / (n * latest_version) as u128);

        drop(backend);
        let storage_size = compute_cole_size_breakdown(path);
        println!("storage size: {:?}", storage_size);
    }
}