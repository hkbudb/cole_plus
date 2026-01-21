use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use utils::{config::Configs, disk_usage_check_directory_storage, types::{AddrKey, Address, StateKey, StateValue}};
use super::tx_executor::Backend;
use std::{cell::UnsafeCell, collections::BTreeMap};
use anyhow::Result;
use cole_plus_ablation_mbtree::ColePlusAblationMBTree;
use utils::MemCost;

pub struct ColePlusAblationMBTreeBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: ColePlusAblationMBTree<'a>,
    pub path: &'a str,
}

impl<'a> ColePlusAblationMBTreeBackend<'a> {
    pub fn new(configs: &'a Configs, path: &'a str) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: ColePlusAblationMBTree::new(configs),
            path,
        }
    }

    pub fn get_mut_total_tree(&self) -> &'a mut ColePlusAblationMBTree<'a> {
        unsafe {
            let const_ptr = &self.states as *const ColePlusAblationMBTree;
            let mut_ptr = UnsafeCell::new(const_ptr as *mut ColePlusAblationMBTree);
            &mut **mut_ptr.get()
        }
    }
}

impl<'a> Backend for ColePlusAblationMBTreeBackend<'a> {
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
        let v = self.get_mut_total_tree().search_latest_state_value(addr_key);
        match v {
            Some((_, _, value)) => {
                Ok(value)
            },
            None => {
                return Ok(StateValue::default());
            }
        }
    }
}

impl<'a> BackendWriteTrait for ColePlusAblationMBTreeBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, block_id: u32) {
        self.states.insert((addr_key, block_id, v));
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, block_id: u32) {
        for (addr_key, value) in states {
            self.states.insert((addr_key, block_id, value));
        }
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
        format!("")
    }
    fn flush(&mut self) {

    }

    fn commit(&mut self) {
        
    }

    fn print_in_mem_tree(&self) {
        println!("{:?}", self.states.in_mem_group[0]);
        println!("{:?}", self.states.in_mem_group[0]);
    }

    fn index_size(&self) -> usize {
        let disk_size = disk_usage_check_directory_storage(&self.path);
        let mem_size = self.memory_cost().size();
        return disk_size + mem_size;
    }
}
