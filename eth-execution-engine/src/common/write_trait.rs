use std::collections::BTreeMap;
use utils::MemCost;
use utils::types::{StateValue, AddrKey, Address};
use super::{nonce::Nonce, code::Code};

pub trait BackendWriteTrait {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, block_id: u32);
    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, block_id: u32);
    fn set_acc_nonce(&mut self, contract_addr: &Address, contract_nonce: Nonce);
    fn get_acc_nonce(&self, contract_addr: &Address) -> Nonce;
    fn set_acc_code(&mut self, contract_addr: &Address, contract_code: Code);
    fn get_acc_code(&self, contract_addr: &Address) -> Code;
    fn memory_cost(&self,) -> MemCost;
    fn index_stucture_output(&self,) -> String;
    fn flush(&mut self);
    fn commit(&mut self);
    fn print_in_mem_tree(&self);
    fn index_size(&self) -> usize;
}