use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use cxx::UniquePtr;
use utils::{disk_usage_check_directory_storage, MemCost};
use utils::types::{Address, StateKey, StateValue, AddrKey};
use super::tx_executor::Backend;
use std::collections::BTreeMap;
use anyhow::Result;
use letus_new::ffi::{Letus, new_trie};

pub struct LetusBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: UniquePtr<Letus>,
    pub path: &'a str,
    pub block_id: u32,
}

impl<'a> LetusBackend<'a> {
    pub fn new(path: &'a str) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: new_trie(String::from(path)),
            path,
            block_id: 0,
        }
    }
}

impl<'a> Backend for LetusBackend<'a> {
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
        let key_str = bytes_to_hex_string(&addr_key.state_key.to_bytes()[..10]);
        let v = self.states.LetusGet(self.block_id as u64, key_str);
        if v.len() == 0 {
            let res = StateValue::default();
            return Ok(res);
        } else {
            let v_bytes = string_to_bytes(&v);
            let res = StateValue::from_bytes(&v_bytes[..32]);
            return Ok(res);
        }
    }
}

fn bytes_to_hex_string(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

fn string_to_bytes(hex_string: &str) -> Vec<u8> {
    hex::decode(hex_string).unwrap()
}

impl<'a> BackendWriteTrait for LetusBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, _: u32) {
        self.block_id += 1;
        let key_str = bytes_to_hex_string(&addr_key.state_key.to_bytes()[..10]);
        let value_str = bytes_to_hex_string(&v.0.as_bytes().to_vec());
        println!("key: {}, value: {}", key_str, value_str);
        self.states.LetusPut(self.block_id as u64, key_str, value_str);
        self.commit();
        self.flush();
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, _: u32) {
        self.block_id += 1;
        for (addr_key, v) in states {
            let key_str = bytes_to_hex_string(&addr_key.state_key.to_bytes()[..10]);
            let value_str = bytes_to_hex_string(&v.0.as_bytes().to_vec());
            println!("key: {}, value: {}", key_str, value_str);
            self.states.LetusPut(self.block_id as u64, key_str, value_str);
        }
        self.commit();
        self.flush();
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
        String::new()
    }

    fn commit(&mut self) {
        self.states.LetusCommit(self.block_id as u64);
    }

    fn flush(&mut self) {
        self.states.LetusFlush(self.block_id as u64);
    }

    fn print_in_mem_tree(&self) {
        
    }

    fn index_size(&self) -> usize {
        let disk_size = disk_usage_check_directory_storage(&self.path);
        let mem_size = self.memory_cost().size();
        return disk_size + mem_size;
    }
}