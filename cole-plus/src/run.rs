use growable_bloom_filter::GrowableBloom;
use primitive_types::{H160, H256};
use utils::{cacher::CacheManager, merge_sort::{in_memory_collect, merge_state}, pager::{cdc_mht::{reconstruct_cdc_range_proof, CDCRangeProof, CDCTreeReader, VerObject}, model_pager::ModelPageReader, state_pager::{StateIterator, StatePageReader}, upper_mht::{reconstruct_upper_range_proof, UpperMHTRangeProof, UpperMHTReader}}, types::{bytes_hash, AddrKey, Address, CompoundKey, StateKey, StateValue}, OpenOptions, Read, Write};
use utils::config::Configs;
use std::fmt::{Debug, Formatter, Error};
use serde::{Serialize, Deserialize};
const FILTER_FP_RATE: f64 = 0.1; // false positive rate of a filter
const MAX_FILTER_SIZE: usize = 1024 * 1024; // 1M
// define a run in a level
pub struct LevelRun {
    pub run_id: u32, // id of this run
    pub state_reader: StatePageReader, // state's reader
    pub model_reader: ModelPageReader, // model's reader
    pub upper_mht_reader: UpperMHTReader, // upper mht's reader (read each addr's tree offset)
    pub lower_cdc_tree_reader: CDCTreeReader, // lower cdc tree's reader (read each addr's versions)
    pub filter: Option<GrowableBloom>, // bloom filter
    pub filter_hash: Option<H256>, // filter's hash
    pub digest: H256, // a digest of mht_root and filter's hash if filter exists
}

impl LevelRun {
    // load a run using the run's id, level's id and the configuration reference
    pub fn load(run_id: u32, level_id: u32, configs: &Configs) -> Self {
        // first define the file names of the state, model, mht, and filter
        let state_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "lh");
        let filer_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "f");

        // load the readers using their file names
        let state_reader = StatePageReader::load(&state_file_name);
        let model_reader = ModelPageReader::load(&model_file_name);
        let upper_mht_reader = UpperMHTReader::new(&upper_offset_file_name, &upper_hash_file_name);
        let lower_cdc_tree_reader = CDCTreeReader::new(&lower_cdc_file_name);

        // initiate the filter using None object
        let mut filter = None;
        // if the filter's file exists, read the filter from the file
        match OpenOptions::new().read(true).open(&filer_file_name) {
            Ok(mut file) => {
                let mut len_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut len_bytes).unwrap();
                let len = u32::from_be_bytes(len_bytes);
                let mut v = vec![0u8; len as usize];
                file.read_exact(&mut v).unwrap();
                let filter_obj: GrowableBloom = bincode::deserialize(&v).unwrap();
                filter = Some(filter_obj);
            },
            Err(_) => {
            }
        }

        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    fn estimate_all_filter_size(level_id: u32, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> usize {
        let mut total_size = 0;
        let mut cur_level = level_id as i32;
        let mut cur_num_of_state = max_num_of_state;
        let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * level_num_of_run;
        total_size += cur_level_filter_size;
        while cur_level >= 0 {
            cur_level -= 1;
            cur_num_of_state /= size_ratio;
            let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * size_ratio;
            total_size += cur_level_filter_size;
        }
        return total_size;
    }

    // use the in-memory iterator to process the merge operation
    pub fn construct_run_by_in_memory_collection(state_vec: Vec<(CompoundKey, StateValue)>, run_id: u32, level_id: u32, dir_name: &str, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize, is_pruned: bool) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id,  &dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id,  &dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id,  &dir_name, "lh");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_reader, model_reader, upper_mht_reader, lower_cdc_tree_reader,  filter) = in_memory_collect(state_vec, &state_file_name, &model_file_name, &upper_offset_file_name, &upper_hash_file_name, &lower_cdc_file_name, fanout, filter, is_pruned);
        
        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);

        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    // use the state iterator to process the merge operation
    pub fn construct_run_by_merge(input_states: Vec<StateIterator>, lower_cdc_tree_readers: Vec::<CDCTreeReader>, upper_mht_readers: Vec::<(u32, UpperMHTReader)>, run_id: u32, level_id: u32, dir_name: &str, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize, is_pruned: bool) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let upper_offset_file_name = Self::file_name(run_id, level_id,  &dir_name, "uo");
        let upper_hash_file_name = Self::file_name(run_id, level_id,  &dir_name, "uh");
        let lower_cdc_file_name = Self::file_name(run_id, level_id,  &dir_name, "lh");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_reader, model_reader, upper_mht_reader, lower_cdc_tree_reader, filter) = merge_state(input_states, lower_cdc_tree_readers, upper_mht_readers, &state_file_name, &model_file_name, &upper_offset_file_name, &upper_hash_file_name, &lower_cdc_file_name, fanout, filter, is_pruned);
        
        let mht_root = upper_mht_reader.mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);

        Self {
            run_id,
            state_reader,
            model_reader,
            upper_mht_reader,
            lower_cdc_tree_reader,
            filter,
            filter_hash,
            digest,
        }
    }

    // helper function to generate the file name of different file types: "s", "m", "h"
    pub fn file_name(run_id: u32, level_id: u32, dir_name: &str, file_type: &str) -> String {
        format!("{}/{}_{}_{}.dat", dir_name, file_type, level_id, run_id)
    }

    // persist the filter if it exists
    pub fn persist_filter(&self, level_id: u32, configs: &Configs) {
        if self.filter.is_some() {
            // init the filter's file name
            let filer_file_name = Self::file_name(self.run_id, level_id, &configs.dir_name, "f");
            // serialize the filter using bincode
            let bytes = bincode::serialize(self.filter.as_ref().unwrap()).unwrap();
            // get the length of the serialized bytes
            let bytes_len = bytes.len() as u32;
            // v is a vector that will be persisted to the filter's file
            let mut v = bytes_len.to_be_bytes().to_vec();
            v.extend(&bytes);
            // write v to the file
            let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&filer_file_name).unwrap();
            file.write_all(&mut v).unwrap();
        }
    }

    pub fn search_run(&mut self, compound_key: &CompoundKey, cache_manager: &mut CacheManager) -> Option<(AddrKey, u32, StateValue)> {
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&compound_key.addr) {
                return None;
            }
        }
        // use the model file to predict the position in the state file
        // use model file to predict the page_id
        let pred_page_id = self.model_reader.get_pred_state_page_id(self.run_id, compound_key, false, cache_manager);
        let (compound_key, state_value) = self.state_reader.query_page_state(compound_key, self.run_id, pred_page_id, cache_manager);
        return Some((compound_key.addr, compound_key.version, state_value));
    }

    pub fn prove_range(&mut self, addr_key: AddrKey, lb: u32, ub: u32, configs: &Configs, cache_manager: &mut CacheManager) -> (Option<Vec<VerObject>>, RunProof) {
        // init the proof
        let mut proof = RunProof::new();
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&addr_key) {
                // the addr_key must not exist in the run, so include the filter to the proof
                let filter = self.filter.clone().unwrap();
                let mht_root = self.upper_mht_reader.mht_reader.root.unwrap();
                proof.include_filter_and_mht_root(filter, mht_root);
                return (None, proof);
            }
        }
        // use the model file to predict the position in the state file
        // use model file to predict the page_id
        let compound_key = CompoundKey::new(addr_key, lb);
        let pred_page_id = self.model_reader.get_pred_state_page_id(self.run_id, &compound_key, false, cache_manager);
        let (addr_key_exist, merkle_index_vec) = self.state_reader.query_merkle_index(&addr_key, self.run_id, pred_page_id, cache_manager);
        if addr_key_exist {
            // addr_key exists, should search the lower cdc tree
            let (addr_key, lower_tree_index) = merkle_index_vec[0];
            let lower_tree_index = lower_tree_index - 1; // since the first addr in each page is 0x00000...000
            // get the tree addr
            let lower_tree_addr = self.upper_mht_reader.merkle_offset_reader.read_merkle_offset(self.run_id, lower_tree_index as usize, cache_manager);
            // generate cdc proof
            let (result, lower_cdc_proof) = self.lower_cdc_tree_reader.search_range_at_tree_addr(lower_tree_addr, self.run_id, lb, ub, cache_manager);
            let lower_proof = LowerProofOrRootHash::LowerProof((addr_key, lower_cdc_proof));
            let (l, r) = (lower_tree_index as usize, lower_tree_index as usize);
            let num_of_leaf_hash = self.upper_mht_reader.merkle_offset_reader.num_of_merkle_offset as usize;
            let upper_proof = self.upper_mht_reader.mht_reader.prove_upper_mht_range(self.run_id, l, r, num_of_leaf_hash, configs.fanout, cache_manager);
            let upper_and_lower_proof = UpperAndLowerProof::new(upper_proof, lower_proof);
            // compute filter's hash and add it to the Run's proof
            let filter_hash = self.filter_hash.clone();
            proof.include_range_proof_and_filer_hash(upper_and_lower_proof, filter_hash);
            return (result, proof);
        } else {
            let mut v: Vec::<(AddrKey, H256)> = Vec::new();
            // addr_key does not exist, should include the roots of the boundary addr keys' lower trees
            let mut index_vec = Vec::new();
            for (addr_key, lower_tree_index) in &merkle_index_vec {
                // read the root hash at lower_tree_addr
                let min_addr_key = AddrKey::default();
                let max_addr_key = AddrKey::new(Address(H160::from_slice(&vec![255u8;20])), StateKey(H256::from_slice(&vec![255u8;32])));
                if *addr_key != min_addr_key && *addr_key != max_addr_key {
                    // not the default addr 0x00000...000 and not the max addr 0xffff...fff
                    let index = (*lower_tree_index - 1) as usize; // since the first addr in each page is 0x00000...000
                    index_vec.push(index);
                    let lower_tree_addr = self.upper_mht_reader.merkle_offset_reader.read_merkle_offset(self.run_id, index, cache_manager);
                    let lower_tree_root = self.lower_cdc_tree_reader.read_tree_root_at(lower_tree_addr, self.run_id, cache_manager);
                    v.push((*addr_key, lower_tree_root));
                }
            }
            let lower_proof = LowerProofOrRootHash::RootHash(v);
            let (l, r) = if index_vec.len() == 1 {
                (index_vec[0], index_vec[0])
            } else {
                (index_vec[0], index_vec[index_vec.len()-1])
            };
            let num_of_leaf_hash = self.upper_mht_reader.merkle_offset_reader.num_of_merkle_offset as usize;
            let upper_proof = self.upper_mht_reader.mht_reader.prove_upper_mht_range(self.run_id, l, r, num_of_leaf_hash, configs.fanout, cache_manager);
            let upper_and_lower_proof = UpperAndLowerProof::new(upper_proof, lower_proof);
            // compute filter's hash and add it to the Run's proof
            let filter_hash = self.filter_hash.clone();
            proof.include_range_proof_and_filer_hash(upper_and_lower_proof, filter_hash);
            return (None, proof);
        }
    }
    
    // derive the digest of the LevelRun
    pub fn compute_digest(&self) -> H256 {
        let mht_root = self.upper_mht_reader.mht_reader.root.unwrap();
        let mut bytes = mht_root.as_bytes().to_vec();
        if self.filter_hash.is_some() {
            bytes.extend(self.filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    // compute the digest of the run according to the MHT root and the filter if it exists
    pub fn load_digest(mht_root: H256, filter_hash: &Option<H256>) -> H256 {
        let mut bytes = mht_root.as_bytes().to_vec();
        if filter_hash.is_some() {
            bytes.extend(filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    pub fn filter_cost(&self) -> RunFilterSize {
        // filter cost
        let filter_ref = self.filter.as_ref();
        let filter_size;
        if filter_ref.is_some() {
            filter_size = filter_ref.unwrap().memory_size();
        } else {
            filter_size = 0;
        }
        return RunFilterSize::new(filter_size);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RunFilterSize {
    pub filter_size: usize,
}

impl RunFilterSize {
    pub fn new(filter_size: usize) -> Self {
        Self {
            filter_size,
        }
    }

    pub fn add(&mut self, other: &RunFilterSize) {
        self.filter_size += other.filter_size;
    }
}

impl Debug for LevelRun {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "<Level Run Info> run_id: {}, filter is some: {:?}, page len: {}", self.run_id, self.filter.is_some(), self.state_reader.num_stored_pages)
    }
}

/*
Compute a recommended bitmap size for items_count items
and a fp_p rate of false positives.
fp_p obviously has to be within the ]0.0, 1.0[ range.
 */
pub fn compute_bitmap_size_in_bytes(items_count: usize, fp_p: f64) -> usize {
    assert!(items_count > 0);
    assert!(fp_p > 0.0 && fp_p < 1.0);
    // We're using ceil instead of round in order to get an error rate <= the desired.
    // Using round can result in significantly higher error rates.
    let num_slices = ((1.0 / fp_p).log2()).ceil() as u64;
    let slice_len_bits = (items_count as f64 / 2f64.ln()).ceil() as u64;
    let total_bits = num_slices * slice_len_bits;
    // round up to the next byte
    let buffer_bytes = ((total_bits + 7) / 8) as usize;
    buffer_bytes
}

/* Either a filter or the digest of the filter
   used to be the part of the proof to prove the non-existence of the addr
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOrHash {
    Filter(GrowableBloom),
    Hash(H256),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpperAndLowerProof {
    upper_proof: UpperMHTRangeProof,
    lower_proof: LowerProofOrRootHash,
}

impl UpperAndLowerProof {
    pub fn new(upper_proof: UpperMHTRangeProof, lower_proof: LowerProofOrRootHash) -> Self {
        Self {
            upper_proof,
            lower_proof,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LowerProofOrRootHash {
    LowerProof((AddrKey, CDCRangeProof)),
    RootHash(Vec<(AddrKey, H256)>)
}

/* Either a range proof or the MHT root hash
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeProofOrHash {
    RangeProof(UpperAndLowerProof),
    Hash(H256),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunProof {
    range_proof_or_hash: RangeProofOrHash,
    filter_or_hash: Option<FilterOrHash>,
}

impl RunProof {
    // init the run proof with the empty range proof and empty filter or hash
    pub fn new() -> Self {
        let range_proof_or_hash = RangeProofOrHash::Hash(H256::default());
        let filter_or_hash = None;
        Self {
            range_proof_or_hash,
            filter_or_hash,
        }
    }

    pub fn include_filter_and_mht_root(&mut self, filter: GrowableBloom, root: H256) {
        let range_proof_or_hash = RangeProofOrHash::Hash(root);
        let filter_or_hash = Some(FilterOrHash::Filter(filter));
        self.filter_or_hash = filter_or_hash;
        self.range_proof_or_hash = range_proof_or_hash;
    }

    pub fn include_range_proof_and_filer_hash(&mut self, range_proof: UpperAndLowerProof, filter_hash: Option<H256>) {
        let range_proof_or_hash = RangeProofOrHash::RangeProof(range_proof);
        let mut filter_or_hash = None;
        if filter_hash.is_some() {
            filter_or_hash = Some(FilterOrHash::Hash(filter_hash.unwrap()));
        }
        self.filter_or_hash = filter_or_hash;
        self.range_proof_or_hash = range_proof_or_hash;
    }

    pub fn get_filter_ref(&self) -> Option<&GrowableBloom> {
        match &self.filter_or_hash {
            Some(filter_or_hash) => {
                match filter_or_hash {
                    FilterOrHash::Filter(f) => Some(f),
                    FilterOrHash::Hash(_) => None,
                }
            },
            None => return None,
        }
    }

    pub fn get_range_proof_ref(&self) -> Option<&UpperAndLowerProof> {
        match &self.range_proof_or_hash {
            RangeProofOrHash::RangeProof(p) => Some(p),
            RangeProofOrHash::Hash(_) => None,
        }
    }

    pub fn get_filter_hash(&self) -> Option<H256> {
        match &self.filter_or_hash {
            Some(filter_or_hash) => {
                match filter_or_hash {
                    FilterOrHash::Filter(_) => None,
                    FilterOrHash::Hash(h) => Some(*h),
                }
            },
            None => return None,
        }
    }

    pub fn get_merkle_hash(&self) -> Option<H256> {
        match &self.range_proof_or_hash {
            RangeProofOrHash::Hash(h) => Some(*h),
            RangeProofOrHash::RangeProof(_) => None,
        }
    }
}

pub fn reconstruct_run_proof(addr_key: &AddrKey, lb: u32, ub: u32, results: &Option<Vec<VerObject>>, proof: &RunProof, fanout: usize) -> H256 {
    if results.is_none() {
        let filter_ref = proof.get_filter_ref();
        let merkle_root = proof.get_merkle_hash();
        if filter_ref.is_some() {
            let filter = filter_ref.unwrap();
            let root = merkle_root.unwrap();
            if !filter.contains(addr_key) {
                // filter does not contain addr_key, it successfully prove the non-existence, combine the filter's hash and merkle root hash
                let filter_bytes = bincode::serialize(filter).unwrap();
                let filter_hash = bytes_hash(&filter_bytes);
                let mut bytes = root.as_bytes().to_vec();
                bytes.extend(filter_hash.as_bytes());
                let recomputed_h = bytes_hash(&bytes);
                return recomputed_h;
            }
        }
        // no addr key
        let range_proof = proof.get_range_proof_ref().unwrap();
        let upper_proof = &range_proof.upper_proof;
        let lower_proof = &range_proof.lower_proof;
        match lower_proof {
            LowerProofOrRootHash::LowerProof(_) => {  
                // error
                return H256::default(); 
            },
            LowerProofOrRootHash::RootHash(addr_and_root) => {
                let mut obj_hashes: Vec<H256> = Vec::new();
                for (boundary_addr_key, root) in addr_and_root {
                    let mut bytes = Vec::<u8>::new(); // help compute the lower tree root
                    bytes.extend(boundary_addr_key.to_bytes());
                    bytes.extend(root.as_bytes());
                    let lower_tree_root = bytes_hash(&mut bytes); // lower tree root = H(addr_key | cdc_tree_root)
                    obj_hashes.push(lower_tree_root);
                }
                let reconstruct_merkle_root  = reconstruct_upper_range_proof(upper_proof, fanout, obj_hashes);
                let filter_hash = proof.get_filter_hash();
                let mut bytes = reconstruct_merkle_root.as_bytes().to_vec();
                if filter_hash.is_some() {
                    bytes.extend(filter_hash.unwrap().as_bytes());
                }
                let recomputed_h = bytes_hash(&bytes);
                return recomputed_h;
            },
        }
    } else {
        // exist addr_key
        let range_proof = proof.get_range_proof_ref().unwrap();
        let upper_proof = &range_proof.upper_proof;
        let lower_proof = &range_proof.lower_proof;
        match lower_proof {
            LowerProofOrRootHash::LowerProof((boundary_addr_key, cdc_range_proof)) => {
                if boundary_addr_key != addr_key {
                    // error, mismatch addr_key
                    return H256::default();
                }
                let cdc_root = reconstruct_cdc_range_proof(lb, ub, results, cdc_range_proof);
                let mut bytes = Vec::<u8>::new(); // help compute the lower tree root
                bytes.extend(boundary_addr_key.to_bytes());
                bytes.extend(cdc_root.as_bytes());
                let lower_tree_root = bytes_hash(&mut bytes); // lower tree root = H(addr_key | cdc_tree_root)
                let obj_hashes: Vec<H256> = vec![lower_tree_root];
                let reconstruct_merkle_root  = reconstruct_upper_range_proof(upper_proof, fanout, obj_hashes);
                let filter_hash = proof.get_filter_hash();
                let mut bytes = reconstruct_merkle_root.as_bytes().to_vec();
                if filter_hash.is_some() {
                    bytes.extend(filter_hash.unwrap().as_bytes());
                }
                let recomputed_h = bytes_hash(&bytes);
                return recomputed_h;
            },
            LowerProofOrRootHash::RootHash(_) => {
                // error
                return H256::default(); 
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::in_memory_mbtree::InMemoryMBTree;
    use merkle_btree_storage::traits::BPlusTreeNodeIO;
    use primitive_types::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use utils::{pager::upper_mht::reconstruct_upper_range_proof, types::{bytes_hash, Address, StateKey}};
    use super::*;

    fn generate_addr_keys(num_of_contract: usize, num_of_state: usize, rng: &mut StdRng) -> Vec<AddrKey> {
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            let acc_addr = H160::random_using(rng);
            for _ in 1..=num_of_state {
                let state_addr = H256::random_using(rng);
                let addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
                addr_key_vec.push(addr_key);
            }
        }
        return addr_key_vec;
    }

    #[test]
    fn test_in_memory_collection() {
        let fanout = 4;
        let size_ratio = 10;
        let num_of_contract = 20;
        let num_of_state = 18;
        let num_of_version = 10;
        let n = num_of_contract * num_of_state * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let configs = Configs::new(fanout, 0, dir_name.to_string(), n, size_ratio, false);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let mut addr_key_vec = generate_addr_keys(num_of_contract, num_of_state, &mut rng);
        addr_key_vec.sort();
        for k in 1..=num_of_version {
            for (_, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k as u32), StateValue(H256::from_low_u64_be( k as u64))));
            }
        }

        let mut mb_tree = InMemoryMBTree::new(fanout);
        for (compound_key, value) in &state_vec {
            merkle_btree_storage::insert(&mut mb_tree, *compound_key, *value);
        }

        let loaded = mb_tree.load_all_key_values();
        let run_id = 0;
        let level_id = 0;
        let mut run = LevelRun::construct_run_by_in_memory_collection(loaded.clone(), run_id, level_id, dir_name, fanout, configs.max_num_of_states_in_a_run(level_id), n, size_ratio, false);
        
        let offset_reader = &mut run.upper_mht_reader.merkle_offset_reader;
        let merkle_offsets = offset_reader.load_all_offsets();
        println!("offset len: {}", merkle_offsets.len());
        let lower_cdc_tree_reader = &mut run.lower_cdc_tree_reader;
        let mut cache_manager = CacheManager::new();

        // for page_id in 0..state_reader.num_stored_pages {
        //     let (merkle_index, states) = state_reader.read_deser_states_at(run_id, page_id, &mut cache_manager);
        //     println!("page: {}, merkle index: {:?}", page_id, merkle_index);
        //     println!("page: {}, states: {:?}", page_id, states);
        // }
        
        let num_of_addr_key = merkle_offsets.len();
        let mut lower_tree_root_h = Vec::<H256>::new();
        for i in 0..num_of_addr_key {
            let tree_addr = merkle_offsets[i];
            let cdc_tree = lower_cdc_tree_reader.read_tree_at(tree_addr, run_id, &mut cache_manager).unwrap();
            let cdc_tree_h = cdc_tree.get_root_hash();
            let addr_key = addr_key_vec[i];
            let mut bytes = Vec::<u8>::new(); // help compute the lower tree root
            bytes.extend(addr_key.to_bytes());
            bytes.extend(cdc_tree_h.as_bytes());
            let lower_tree_root = bytes_hash(&mut bytes); // lower tree root = H(addr_key | cdc_tree_root)
            lower_tree_root_h.push(lower_tree_root);
        }

        let upper_mht_reader = &mut run.upper_mht_reader.mht_reader;
        let mut loaded_h = Vec::<H256>::new();
        let num_of_hash_pages = upper_mht_reader.num_stored_pages;
        for i in 0..num_of_hash_pages {
            let v = upper_mht_reader.read_deser_page_at(run_id, i, &mut cache_manager);
            loaded_h.extend(v);
        }
        assert_eq!(loaded_h[0..num_of_addr_key], lower_tree_root_h);
        let upper_mht_root = upper_mht_reader.root.unwrap();
        for i in 0..num_of_addr_key {
            let result = upper_mht_reader.read_hashes_with_index_range(run_id, i, i, &mut cache_manager);
            let proof = upper_mht_reader.prove_upper_mht_range(run_id, i, i, num_of_addr_key, fanout, &mut cache_manager);
            let reconstruct_h = reconstruct_upper_range_proof(&proof, fanout, result);
            if upper_mht_root != reconstruct_h {
                println!("false");
                break;
            }
        }

        let digest = run.digest;
        for addr in &addr_key_vec {
            let lb = 1;
            let ub = num_of_version as u32;
            let (results, proof) = run.prove_range(*addr, lb, ub, &configs, &mut cache_manager);
            let reconstruct_h = reconstruct_run_proof(addr, lb, ub, &results, &proof, fanout);
            if reconstruct_h != digest {
                println!("false");
            }
        }
        let acc_addr = H160::random_using(&mut rng);
        let state_addr = H256::random_using(&mut rng);
        let random_addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
        let (results, proof) = run.prove_range(random_addr_key, 1, num_of_version as u32, &configs, &mut cache_manager);
        println!("results: {:?}", results);
        let reconstruct_h = reconstruct_run_proof(&random_addr_key, 1, num_of_version as u32, &results, &proof, fanout);
        if reconstruct_h != digest {
            println!("false");
        }
    }

    #[test]
    fn test_merge_construct() {
        let fanout = 4;
        let size_ratio = 10;
        let num_of_contract = 12;
        let num_of_state = 12;
        let num_of_version = 12;
        let n = num_of_contract * num_of_state * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let configs = Configs::new(fanout, 0, dir_name.to_string(), n, size_ratio, false);

        let mut addr_key_vec = generate_addr_keys(num_of_contract, num_of_state, &mut rng);
        addr_key_vec.sort();
        println!("addr keys: {:?}", addr_key_vec);
        let level_id = 0;
        let num_of_runs = 7;
        let mut state_iters: Vec<StateIterator> = Vec::new();
        let mut lower_tree_readers: Vec<CDCTreeReader> = Vec::new();
        let mut upper_mht_readers: Vec::<(u32, UpperMHTReader)> = Vec::new();
        let mut ver_cnt = 1;
        for run_id in 0..num_of_runs {
            let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
            for _ in 1..=(run_id+1) {
                for addr_key in addr_key_vec.iter() {
                    let ver = ver_cnt;
                    println!("ver: {}", ver);
                    ver_cnt += 1;
                    let compound_key = CompoundKey::new(*addr_key, ver as u32);
                    let value = StateValue(H256::from_low_u64_be(ver as u64));
                    state_vec.push((compound_key, value));
                }
            }

            let mut mb_tree = InMemoryMBTree::new(fanout);
            for (compound_key, value) in &state_vec {
                merkle_btree_storage::insert(&mut mb_tree, *compound_key, *value);
            }

            let loaded = mb_tree.load_all_key_values();
            let mut run = LevelRun::construct_run_by_in_memory_collection(loaded, run_id as u32, level_id, dir_name, fanout, configs.max_num_of_states_in_a_run(level_id), n, size_ratio, false);
            let offset_reader = &mut run.upper_mht_reader.merkle_offset_reader;
            let merkle_offsets = offset_reader.load_all_offsets();
            println!("run_id: {}, offsets: {:?}", run_id, merkle_offsets);
            // for offset in merkle_offsets {
            //     let lower_tree = run.lower_cdc_tree_reader.read_tree_at(offset, run_id as u32, &mut cache_manager).unwrap();
            //     println!("lower tree at offset: {}", offset);
            //     lower_tree.print_tree();
            //     assert_eq!(lower_tree.min_keep_left_nodes.len(), lower_tree.height());
            // }
            state_iters.push(run.state_reader.to_state_iter());
            lower_tree_readers.push(run.lower_cdc_tree_reader);
            upper_mht_readers.push((run_id as u32, run.upper_mht_reader));
        }

        let next_run_id = num_of_runs as u32;
        let next_level_id = level_id + 1;
        let mut disk_run = LevelRun::construct_run_by_merge(state_iters, lower_tree_readers, upper_mht_readers, next_run_id, next_level_id, dir_name, fanout, configs.max_num_of_states_in_a_run(next_level_id), 0, size_ratio, false);
        let offset_reader = &mut disk_run.upper_mht_reader.merkle_offset_reader;
        let merkle_offsets = offset_reader.load_all_offsets();
        println!("offset len: {}", merkle_offsets.len());
        assert_eq!(merkle_offsets.len(), offset_reader.num_of_merkle_offset as usize);
        println!("offsets: {:?}", merkle_offsets);
        let lower_cdc_tree_reader = &mut disk_run.lower_cdc_tree_reader;
        let mut cache_manager = CacheManager::new();
        
        let num_of_addr_key = merkle_offsets.len();
        let mut lower_tree_root_h = Vec::<H256>::new();
        for i in 0..num_of_addr_key {
            let tree_addr = merkle_offsets[i];
            let cdc_tree = lower_cdc_tree_reader.read_tree_at(tree_addr, next_run_id, &mut cache_manager).unwrap();
            let cdc_tree_h = cdc_tree.get_root_hash();
            let addr_key = addr_key_vec[i];
            let mut bytes = Vec::<u8>::new(); // help compute the lower tree root
            bytes.extend(addr_key.to_bytes());
            bytes.extend(cdc_tree_h.as_bytes());
            let lower_tree_root = bytes_hash(&mut bytes); // lower tree root = H(addr_key | cdc_tree_root)
            lower_tree_root_h.push(lower_tree_root);
        }

        let upper_mht_reader = &mut disk_run.upper_mht_reader.mht_reader;
        let mut loaded_h = Vec::<H256>::new();
        let num_of_hash_pages = upper_mht_reader.num_stored_pages;
        for i in 0..num_of_hash_pages {
            let v = upper_mht_reader.read_deser_page_at(next_run_id, i, &mut cache_manager);
            loaded_h.extend(v);
        }
        assert_eq!(loaded_h[0..num_of_addr_key], lower_tree_root_h);

        let upper_mht_root = upper_mht_reader.root.unwrap();
        for i in 0..num_of_addr_key {
            let result = upper_mht_reader.read_hashes_with_index_range(next_run_id, i, i, &mut cache_manager);
            let proof = upper_mht_reader.prove_upper_mht_range(next_run_id, i, i, num_of_addr_key, fanout, &mut cache_manager);
            let reconstruct_h = reconstruct_upper_range_proof(&proof, fanout, result);
            if upper_mht_root != reconstruct_h {
                println!("false");
                break;
            }
        }
    }
}