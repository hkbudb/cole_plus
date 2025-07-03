pub mod state_pager;
pub mod model_pager;
pub mod cdc_mht;
pub mod upper_mht;
use primitive_types::H256;
use crate::{models::{CompoundKeyModel, MODEL_SIZE}, types::{AddrKey, CompoundKey, StateValue, ADDRKEY_SIZE, COMPOUND_KEY_SIZE, MHT_INDEX_SIZE, NUM_OF_VER_SIZE, STATEVALUE_SIZE, VALUE_SIZE, VERSION_SIZE}};
use self::model_pager::ModelCollections;
pub const PAGE_SIZE: usize = 4096;
pub const MAX_NUM_MODEL_IN_PAGE: usize = PAGE_SIZE / MODEL_SIZE;
pub const MAX_NUM_HASH_IN_PAGE: usize = PAGE_SIZE / 32 - 1; // dedeuction of one is because we need to store some meta-data (i.e., num_of_hash) in the page
pub const MAX_NUM_MERKLE_OFFSET_IN_PAGE: usize = PAGE_SIZE / 8 - 1;

pub mod old_state_pager;
pub mod old_mht_pager;
pub const OLD_STATE_SIZE: usize = COMPOUND_KEY_SIZE + VALUE_SIZE;
pub const MAX_NUM_OLD_STATE_IN_PAGE: usize = PAGE_SIZE / OLD_STATE_SIZE;
/* Structure of a page with default 4096 bytes
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Page {
    pub block: [u8; PAGE_SIZE], // 4096 byte page size
}

impl Page {
    /* Initialize a page with 4096 bytes
     */
    pub fn new() -> Self {
        Self {
            block: [0u8; PAGE_SIZE],
        }
    }

    /* Create a page with the given data block
     */
    pub fn from_array(block: [u8; PAGE_SIZE]) -> Self {
        Self {
            block,
        }
    }

    /* Write model vector to the page
    4 bytes num of model | 4 bytes model level | model_0, model_1, ...
     */
    pub fn from_model_vec(v: &Vec<CompoundKeyModel>, model_level: u32) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_MODEL_IN_PAGE);
        let mut page = Self::new();
        let num_of_model = v.len() as u32;
        let num_of_model_bytes = num_of_model.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of models in the front of the page
        offset.copy_from_slice(&num_of_model_bytes);
        let model_level_bytes = model_level.to_be_bytes();
        let offset = &mut page.block[4..8];
        // write model level
        offset.copy_from_slice(&model_level_bytes);
        // iteratively write each model to the block
        for (i, model) in v.iter().enumerate() {
            let bytes = model.to_bytes();
            let start_idx = i * MODEL_SIZE + 8;
            let end_idx = start_idx + MODEL_SIZE;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(&bytes);
        }
        return page;
    }

    /*
    Read the models from a block
     */
    pub fn to_model_vec(&self) -> ModelCollections {
        let mut v = Vec::<CompoundKeyModel>::new();
        // deserialize the number of models in the page
        let num_of_model = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize the model level
        let model_level = u32::from_be_bytes(self.block[4..8].try_into().expect("error"));
        // deserialize each of the model from the page
        for i in 0..num_of_model {
            let start_idx = i * MODEL_SIZE + 8;
            let end_idx = start_idx + MODEL_SIZE;
            let model = CompoundKeyModel::from_bytes(&self.block[start_idx .. end_idx]);
            v.push(model);
        }
        return ModelCollections {
            v,
            model_level,
        };
    }

    pub fn to_compound_key_value(&self) -> Vec<(CompoundKey, StateValue)> {
        let mut r = Vec::<(CompoundKey, StateValue)>::new();
        // first read page_n (first 4 bytes)
        let page_n = u32::from_be_bytes(self.block[0..4].try_into().unwrap());
        // iteratively read each address's versions and states
        let mut state_cnt = 0;
        let mut cur_read_offset = 4;
        while state_cnt < page_n {
            // first read the addr_key
            let addr_key = AddrKey::from_bytes(&self.block[cur_read_offset .. cur_read_offset + ADDRKEY_SIZE]);
            cur_read_offset += ADDRKEY_SIZE;
            // skip MHT_INDEX
            cur_read_offset += MHT_INDEX_SIZE;
            // read num_of_ver
            let num_of_ver = u32::from_be_bytes(self.block[cur_read_offset .. cur_read_offset + NUM_OF_VER_SIZE].try_into().unwrap());
            cur_read_offset += NUM_OF_VER_SIZE;
            for _ in 0..num_of_ver {
                let version = u32::from_be_bytes(self.block[cur_read_offset .. cur_read_offset + VERSION_SIZE].try_into().unwrap());
                cur_read_offset += VERSION_SIZE;
                let state = StateValue::from_bytes(&self.block[cur_read_offset .. cur_read_offset + STATEVALUE_SIZE]);
                cur_read_offset += STATEVALUE_SIZE;
                // println!("ver: {}, state: {:?}", version, state);
                r.push((CompoundKey::new(addr_key, version), state));
            }
            state_cnt += num_of_ver;
        }
        return r;
    }
    
    pub fn to_state(&self) -> (Vec<(AddrKey, u32)>, Vec<(AddrKey, Vec<(u32, StateValue)>)>) {
        let mut mht_index_vec = Vec::<(AddrKey, u32)>::new();
        let mut state_version_vec = Vec::<(AddrKey, Vec::<(u32, StateValue)>)>::new();
        // first read page_n (first 4 bytes)
        let page_n = u32::from_be_bytes(self.block[0..4].try_into().unwrap());
        // println!("page_n: {}", page_n);
        // iteratively read each address's versions and states
        let mut state_cnt = 0;
        let mut cur_read_offset = 4;
        while state_cnt < page_n {
            // first read the addr_key
            let addr_key = AddrKey::from_bytes(&self.block[cur_read_offset .. cur_read_offset + ADDRKEY_SIZE]);
            // println!("addr_key: {:?}", addr_key);
            let mut addr_verion_vec = vec![];
            cur_read_offset += ADDRKEY_SIZE;
            // read MHT_INDEX
            let mht_index = u32::from_be_bytes(self.block[cur_read_offset .. cur_read_offset + MHT_INDEX_SIZE].try_into().unwrap());
            // println!("mht_index: {}", mht_index);
            mht_index_vec.push((addr_key, mht_index));
            cur_read_offset += MHT_INDEX_SIZE;
            // read num_of_ver
            let num_of_ver = u32::from_be_bytes(self.block[cur_read_offset .. cur_read_offset + NUM_OF_VER_SIZE].try_into().unwrap());
            // println!("num of ver: {}", num_of_ver);
            cur_read_offset += NUM_OF_VER_SIZE;
            for _ in 0..num_of_ver {
                let version = u32::from_be_bytes(self.block[cur_read_offset .. cur_read_offset + VERSION_SIZE].try_into().unwrap());
                cur_read_offset += VERSION_SIZE;
                let state = StateValue::from_bytes(&self.block[cur_read_offset .. cur_read_offset + STATEVALUE_SIZE]);
                cur_read_offset += STATEVALUE_SIZE;
                // println!("ver: {}, state: {:?}", version, state);
                addr_verion_vec.push((version, state));
            }
            state_version_vec.push((addr_key, addr_verion_vec));
            state_cnt += num_of_ver;
        }
        return (mht_index_vec, state_version_vec);
    }
    
    /* Write hash vector to the page
    4 bytes num of hash | hash_0, hash_1, ...
     */
    pub fn from_hash_vec(v: &Vec<H256>) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_HASH_IN_PAGE);
        let mut page = Self::new();
        let num_of_hash = v.len() as u32;
        let num_of_hash_bytes = num_of_hash.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of hash in the front of the page
        offset.copy_from_slice(&num_of_hash_bytes);
        for (i, hash) in v.iter().enumerate() {
            let bytes = hash.as_bytes();
            let start_idx = i * 32 + 4;
            let end_idx = start_idx + 32;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(bytes);
        }
        return page;
    }

    /*
    Read the hashes from a block
     */
    pub fn to_hash_vec(&self) -> Vec<H256> {
        let mut v = Vec::<H256>::new();
        // deserialize the number of hashes in the page
        let num_of_hash = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize each of the hash from the page
        for i in 0..num_of_hash {
            let start_idx = i * 32 + 4;
            let end_idx = start_idx + 32;
            let hash = H256::from_slice(&self.block[start_idx .. end_idx]);
            v.push(hash);
        }
        return v;
    }

    /* Write merkle offset vector to the page
    4 bytes total num of offset for all pages (only for the first page) | 4 bytes num of merkle offset in this page | offset_0, offset_1, ...
     */
    pub fn from_merkle_offset_vec(v: &Vec<u64>) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_MERKLE_OFFSET_IN_PAGE);
        let mut page = Self::new();
        let num_of_merkle_offset = v.len() as u32;
        let num_of_merkle_offset_bytes = num_of_merkle_offset.to_be_bytes();
        let offset = &mut page.block[4..8];
        // write the number of merkle index in the front of the page
        offset.copy_from_slice(&num_of_merkle_offset_bytes);
        for (i, merkle_offset) in v.iter().enumerate() {
            let start_idx = i * 8 + 8;
            let end_idx = start_idx + 8;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(&merkle_offset.to_be_bytes());
        }
        return page;
    }

    pub fn to_merkle_offset_vec(&self) -> Vec<u64> {
        let mut v = Vec::<u64>::new();
        // deserialize the number of merkle offset in the page
        let num_of_merkle_offset = u32::from_be_bytes(self.block[4..8].try_into().expect("error")) as usize;
        // deserialize each of the hash from the page
        for i in 0..num_of_merkle_offset {
            let start_idx = i * 8 + 8;
            let end_idx = start_idx + 8;
            let merkle_offset: u64 = u64::from_be_bytes(self.block[start_idx .. end_idx].try_into().unwrap());
            v.push(merkle_offset);
        }
        return v;
    }

    // old cole design
    // write state vector to the page with the old design 
    /*
    4 bytes num of state | state_0, state_1, ...
     */
    pub fn from_state_vec_old_design(v: &Vec<(CompoundKey, StateValue)>) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_OLD_STATE_IN_PAGE);
        let mut page = Self::new();
        let num_of_state = v.len() as u32;
        let num_of_state_bytes = num_of_state.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of states in the front of the page
        offset.copy_from_slice(&num_of_state_bytes);
        // iteratively write each state to the block
        for (i, (key, value)) in v.iter().enumerate() {
            let mut bytes = vec![];
            bytes.extend(key.to_bytes());
            bytes.extend(value.to_bytes());
            let start_idx = i * OLD_STATE_SIZE + 4;
            let end_idx = start_idx + OLD_STATE_SIZE;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(&bytes);
        }
        return page;
    }

    pub fn to_state_vec_old_design(&self) -> Vec<(CompoundKey, StateValue)> {
        let mut v = Vec::<(CompoundKey, StateValue)>::new();
        // deserialize the number of states in the page
        let num_of_state = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize each of the state from the page
        for i in 0..num_of_state {
            let start_idx = i * OLD_STATE_SIZE + 4;
            let key = CompoundKey::from_bytes(&self.block[start_idx .. start_idx + COMPOUND_KEY_SIZE]);
            let value = StateValue::from_bytes(&self.block[start_idx + COMPOUND_KEY_SIZE .. start_idx + COMPOUND_KEY_SIZE + VALUE_SIZE]);
            v.push((key, value));
        }
        return v;
    }
}

#[cfg(test)]
mod tests {
    use primitive_types::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use super::*;
    #[test]
    fn test_state_old_design_ser_deser() {
        let mut rng = StdRng::seed_from_u64(1);
        for num_in_page in 0..MAX_NUM_OLD_STATE_IN_PAGE {
            let mut v = Vec::<(CompoundKey, StateValue)>::new();
            for i in 0..num_in_page {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let version = i as u32;
                let value: StateValue = H256::random_using(&mut rng).into();
                let key = CompoundKey::new(AddrKey::new(acc_addr.into(), state_addr.into()), version);
                v.push((key, value));
            }
            let page = Page::from_state_vec_old_design(&v);
            let deser_v = page.to_state_vec_old_design();
            assert_eq!(deser_v, v); 
        }
    }
}