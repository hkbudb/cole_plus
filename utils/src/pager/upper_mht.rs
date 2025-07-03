use primitive_types::H256;
use std::{fs::{File, OpenOptions}, io::{Read, Seek, SeekFrom, Write}};
use crate::{cacher::CacheManager, pager::{Page, MAX_NUM_HASH_IN_PAGE, PAGE_SIZE, MAX_NUM_MERKLE_OFFSET_IN_PAGE}, types::compute_concatenate_hash};
use serde::{Serialize, Deserialize};
pub struct UpperMHTWriter{
    pub merkle_offset_writer: MerkleOffsetPageWriter,
    pub output_mht_writer: HashPageWriter,
    pub fanout: usize,
    pub cache_vec: Vec<H256>, // a cache that keeps a vector of at most FANOUT hash values of the level to compute the upper level's hash
    pub num_of_hash: usize, // record the total number of hash values in the file
    pub cnt_in_level: usize, // a counter for the focused level
}

impl UpperMHTWriter {
    pub fn new(merkle_offset_file_name: &str, merkle_hash_file_name: &str, fanout: usize) -> Self {
        Self {
            merkle_offset_writer: MerkleOffsetPageWriter::new(merkle_offset_file_name),
            output_mht_writer: HashPageWriter::new(merkle_hash_file_name),
            fanout,
            cache_vec: Vec::<H256>::new(),
            num_of_hash: 0,
            cnt_in_level: 0,
        }
    }

    pub fn append_upper_mht_offset_and_hash(&mut self, merkle_offset: u64, hash: H256) {
        self.merkle_offset_writer.append(merkle_offset);
        self.output_mht_writer.append(hash);
        self.num_of_hash += 1;
    }

    pub fn finalize_write_mht_offset(&mut self) {
        self.merkle_offset_writer.finalize();
        self.build_upper_mht();
    }

    fn build_upper_mht(&mut self) {
        /*
        recursively construct MHT 
        */
        // n is the number of hash values of the current input MHT level
        let mut n = self.num_of_hash;
        while n != 1 {
            // reset the cnt in the level
            self.reset_cnt_in_level();
            // start_hash_pos is the position of the starting input hash value of the current level
            let start_hash_pos = self.num_of_hash - n;
            // end_hash_pos is the position of the ending input hash value of the current level
            let end_hash_pos = self.num_of_hash - 1;
            let start_page_id = start_hash_pos / MAX_NUM_HASH_IN_PAGE;
            let end_page_id = end_hash_pos / MAX_NUM_HASH_IN_PAGE;
            for page_id in start_page_id ..= end_page_id {
                let mut page_vec = self.output_mht_writer.load_hashes(page_id);
                if page_id == start_page_id {
                    page_vec = page_vec[start_hash_pos % MAX_NUM_HASH_IN_PAGE ..].to_vec();
                } else if page_id == end_page_id {
                    page_vec = page_vec[0..= end_hash_pos % MAX_NUM_HASH_IN_PAGE].to_vec();
                }
                for hash in page_vec {
                    self.add_hash_to_upper_mht(hash);
                }
            }
            self.finalize_mht_level();
            n = self.cnt_in_level;
        }
        self.output_mht_writer.flush();
    }

    fn finalize_mht_level(&mut self) {
        // finalize the hashes in the cache
        if self.cache_vec.len() != 0 {
            let h = compute_concatenate_hash(&self.cache_vec);
            self.output_mht_writer.append(h);
            self.num_of_hash += 1;
            self.cnt_in_level += 1;
            self.cache_vec.clear();
        }
    }

    fn reset_cnt_in_level(&mut self) {
        self.cnt_in_level = 0;
    }

    fn add_hash_to_upper_mht(&mut self, hash: H256) {
        self.cache_vec.push(hash);
        if self.cache_vec.len() == self.fanout {
            // cache is full
            let h = compute_concatenate_hash(&self.cache_vec);
            self.output_mht_writer.append(h);
            self.num_of_hash += 1;
            self.cnt_in_level += 1;
            self.cache_vec.clear();
        }
    }

    pub fn to_upper_mht_reader(self) -> UpperMHTReader {
        UpperMHTReader {
            merkle_offset_reader: self.merkle_offset_writer.to_merkle_offset_page_reader(),
            mht_reader: self.output_mht_writer.to_hash_page_reader(),
        }
    }
}

pub struct UpperMHTReader {
    pub merkle_offset_reader: MerkleOffsetPageReader,
    pub mht_reader: HashPageReader,
}

impl UpperMHTReader {
    pub fn new(merkle_offset_file_name: &str, merkle_hash_file_name: &str) -> Self {
        Self {
            merkle_offset_reader: MerkleOffsetPageReader::new(merkle_offset_file_name),
            mht_reader: HashPageReader::new(merkle_hash_file_name),
        }
    }
}

pub struct MerkleOffsetPageWriter {
    pub file: File, // file object of the corresponding Merkle offset
    pub vec_in_latest_update_page: Vec<u64>, // a preparation vector to obsorb the streaming state values which are not persisted in the file yet
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
    pub num_of_merkle_offset: u32,
}

impl MerkleOffsetPageWriter {
    /* Initialize the writer using a given file name
     */
    pub fn new(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            vec_in_latest_update_page: vec![],
            num_stored_pages: 0,
            num_of_merkle_offset: 0,
        }
    }

    pub fn append(&mut self, merkle_offset: u64) {
        // add the hash
        self.vec_in_latest_update_page.push(merkle_offset);
        self.num_of_merkle_offset += 1;
        if self.vec_in_latest_update_page.len() == MAX_NUM_MERKLE_OFFSET_IN_PAGE {
            // vector is full, should be added to a page and flushed the page to the file
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.vec_in_latest_update_page.len() != 0 {
            // first put the vector into a page
            let page = Page::from_merkle_offset_vec(&self.vec_in_latest_update_page);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.write_all(&page.block).unwrap();
            // clear the vector
            self.vec_in_latest_update_page.clear();
            self.num_stored_pages += 1;
        }
    }

    pub fn finalize(&mut self) {
        self.flush();
        // record the num_of_merkle_offset to the first 4 bytes of the offset file
        self.file.seek(SeekFrom::Start(0)).unwrap();
        self.file.write_all(&self.num_of_merkle_offset.to_be_bytes()).unwrap();
    }

    pub fn to_merkle_offset_page_reader(self) -> MerkleOffsetPageReader {
        MerkleOffsetPageReader {
            file: self.file,
            num_of_merkle_offset: self.num_of_merkle_offset,
        }
    }
}

pub struct MerkleOffsetPageReader {
    pub file: File, // file object of the corresponding Merkle offset
    pub num_of_merkle_offset: u32,
}

impl MerkleOffsetPageReader {
    pub fn new(file_name: &str) -> Self {
        let mut file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        // read num_of_merkle_offset from the first 4 bytes of the file
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes).unwrap();
        let num_of_merkle_offset: u32 = u32::from_be_bytes(bytes);
        Self {
            file,
            num_of_merkle_offset,
        }
    }

    pub fn read_merkle_offset(&mut self, run_id: u32, index: usize, cache_manager: &mut CacheManager) -> u64 {
        let page_id = index / MAX_NUM_MERKLE_OFFSET_IN_PAGE;
        let r = cache_manager.read_offset_cache(run_id, page_id);
        if r.is_some() {
            let page = r.unwrap();
            let offset_vec = page.to_merkle_offset_vec();
            let inner_page_index = index % MAX_NUM_MERKLE_OFFSET_IN_PAGE;
            return offset_vec[inner_page_index];
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let offset_vec = page.to_merkle_offset_vec();
            let inner_page_index = index % MAX_NUM_MERKLE_OFFSET_IN_PAGE;
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_offset_cache(run_id, page_id, page);
            return offset_vec[inner_page_index];
        }
    }

    pub fn load_all_offsets(&mut self, ) -> Vec<u64> {
        let mut offsets = Vec::<u64>::new();
        let max_page_id = self.num_of_merkle_offset as usize / MAX_NUM_MERKLE_OFFSET_IN_PAGE;
        for page_id in 0..=max_page_id {
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let offset_vec = page.to_merkle_offset_vec();
            offsets.extend(offset_vec);
        }
        return offsets;
    }
}

/* A helper that writes the hash into a file with a sequence of pages
   According to the disk-optimization objective, the state writing should be in a streaming fashion.
 */
pub struct HashPageWriter {
    pub file: File, // file object of the corresponding Merkle file
    pub vec_in_latest_update_page: Vec<H256>, // a preparation vector to obsorb the streaming state values which are not persisted in the file yet
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
}

impl HashPageWriter {
    /* Initialize the writer using a given file name
     */
    pub fn new(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            vec_in_latest_update_page: vec![],
            num_stored_pages: 0,
        }
    }

    /* Streamingly add the hash to the latest_update_page
       Flush the latest_update_page to the file once it is full, and clear it.
     */
    pub fn append(&mut self, hash: H256) {
        // add the hash
        self.vec_in_latest_update_page.push(hash);
        if self.vec_in_latest_update_page.len() == MAX_NUM_HASH_IN_PAGE {
            // vector is full, should be added to a page and flushed the page to the file
            self.flush();
        }
    }

    /* Flush the vector in latest update page to the last page in the value file
     */
    pub fn flush(&mut self) {
        if self.vec_in_latest_update_page.len() != 0 {
            // first put the vector into a page
            let page = Page::from_hash_vec(&self.vec_in_latest_update_page);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.write_all(&page.block).unwrap();
            // clear the vector
            self.vec_in_latest_update_page.clear();
            self.num_stored_pages += 1;
        }
    }

    pub fn load_hashes(&mut self, page_id: usize) -> Vec<H256> {
        let mut hash_vec = Vec::<H256>::new();
        if page_id == self.num_stored_pages {
            // page should be read from the in-memory cache vector
            for hash in &self.vec_in_latest_update_page {
                hash_vec.push(*hash);
            }
        } else if page_id < self.num_stored_pages {
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            // deserialize the hashes from the page
            let v = page.to_hash_vec();
            hash_vec.extend(v);
        }
        return hash_vec;
    }

    pub fn to_hash_page_reader(mut self) -> HashPageReader {
        let num_stored_pages = self.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut root = None;
        if num_stored_pages != 0 {
            let last_page_id = num_stored_pages - 1;
            let last_page_offset = last_page_id * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(last_page_offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let page_vec = page.to_hash_vec();
            root = Some(*page_vec.last().unwrap());
        }
        HashPageReader {
            file: self.file,
            root,
            num_stored_pages,
        }
    }
}

pub struct HashPageReader {
    pub file: File, // file object of the corresponding Merkle file
    pub root: Option<H256>, // cache of the root hash
    pub num_stored_pages: usize,
}

impl HashPageReader {
    pub fn new(file_name: &str) -> Self {
        let mut file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut root = None;
        if num_stored_pages != 0 {
            let last_page_id = num_stored_pages - 1;
            let last_page_offset = last_page_id * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            file.seek(SeekFrom::Start(last_page_offset as u64)).unwrap();
            file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let page_vec = page.to_hash_vec();
            root = Some(*page_vec.last().unwrap());
        }

        Self {
            file,
            root,
            num_stored_pages,
        }
    }

    pub fn read_hashes_with_index_range(&mut self, run_id: u32, l: usize, r: usize, cache_manager: &mut CacheManager) -> Vec<H256> {
        let page_id_l = l / MAX_NUM_HASH_IN_PAGE;
        let page_id_r = r / MAX_NUM_HASH_IN_PAGE;
        let mut v = Vec::<H256>::new();
        for page_id in page_id_l ..= page_id_r {
            let hashes = self.read_deser_page_at(run_id, page_id, cache_manager);
            v.extend(&hashes);
        }
        // keep the hashes from proof_pos_l % MAX_NUM_HASH_IN_PAGE to 
        let left_slice_pos = l % MAX_NUM_HASH_IN_PAGE;
        let right_slice_pos = (page_id_r - page_id_l) * MAX_NUM_HASH_IN_PAGE + (r % MAX_NUM_HASH_IN_PAGE);
        v = v[left_slice_pos ..= right_slice_pos].to_vec();
        return v;
    }

    /* Load the deserialized vector of the page from the file at given page_id
     */
    pub fn read_deser_page_at(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> Vec<H256> {
        // first check whether the cache contains the page
        let r = cache_manager.read_mht_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let page = r.unwrap();
            page.to_hash_vec()
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut v = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut v).unwrap();
            let page = Page::from_array(v);
            let v = page.to_hash_vec();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_mht_cache(run_id, page_id, page);
            return v;
        }
    }

    /* Generate the non-leaf range proof given the left position l, right position r, and the total number of hash in the leaf of the MHT
    */
    pub fn prove_upper_mht_range(&mut self, run_id: u32, l: usize, r: usize, num_of_leaf_hash: usize, fanout: usize, cache_manager: &mut CacheManager) -> UpperMHTRangeProof {
        let mut proof = UpperMHTRangeProof::default();
        proof.index_list = [l, r];
        if num_of_leaf_hash == 1 {
            // only one data, just return the empty proof since the data's hash equals the root hash
            return proof;
        } else {
            // compute the size of the current level
            let mut cur_level_size = num_of_leaf_hash;
            // a position that is used to determine the first position of the current level
            let mut start_idx = 0;
            // compute the level's left and right position
            let mut level_l = l;
            let mut level_r = r;
            // iteratively add the hash values from the bottom to the top
            while cur_level_size != 1 {
                // compute the boundary of the two positions (i.e. used to generate the left and right hashes of the proved Merkle node to reconstruct the Merkle root)
                let proof_pos_l = level_l - level_l % fanout;
                let proof_pos_r = if level_r - level_r % fanout + fanout > cur_level_size {
                    cur_level_size
                } else {
                    level_r - level_r % fanout + fanout
                } - 1;
                let proof_pos_l = proof_pos_l + start_idx;
                let proof_pos_r = proof_pos_r + start_idx;
                // next, retrieve the hash values from the position (proof_pos_l + start_idx) to (proof_pos_r + start_idx)
                // compute the corresponding page id
                let page_id_l = proof_pos_l / MAX_NUM_HASH_IN_PAGE;
                let page_id_r = proof_pos_r / MAX_NUM_HASH_IN_PAGE;
                let mut v = Vec::<H256>::new();
                for page_id in page_id_l ..= page_id_r {
                    let hashes = self.read_deser_page_at(run_id, page_id, cache_manager);
                    v.extend(&hashes);
                }
                // keep the hashes from proof_pos_l % MAX_NUM_HASH_IN_PAGE to 
                let left_slice_pos = proof_pos_l % MAX_NUM_HASH_IN_PAGE;
                let right_slice_pos = (page_id_r - page_id_l) * MAX_NUM_HASH_IN_PAGE + (proof_pos_r % MAX_NUM_HASH_IN_PAGE);
                v = v[left_slice_pos ..= right_slice_pos].to_vec();
                // remove the proving hashes from index level_l - proof_pos_l to level_r - proof_pos_l
                for _ in 0..(level_r - level_l + 1) {
                    v.remove(level_l - (proof_pos_l - start_idx));
                }

                proof.p.push(v);
                level_l /= fanout;
                level_r /= fanout;
                start_idx += cur_level_size;
                cur_level_size = ((cur_level_size as f64) / fanout as f64).ceil() as usize;
            }
            return proof;
        }
    }
}

pub fn reconstruct_upper_range_proof(proof: &UpperMHTRangeProof, fanout: usize, obj_hashes: Vec<H256>) -> H256 {
    let l = proof.index_list[0];
    let r = proof.index_list[1];
    let mut index_list: Vec<usize> = (l..=r).collect();
    if index_list.len() == 1 && index_list[0] == 0 && proof.p.len() == 0 {
        return obj_hashes[0].clone();
    } else {
        let mut inserted_hashes = obj_hashes;
        for elem in &proof.p {
            let mut v = elem.clone();
            let offset = index_list[0] % fanout;
            for i in 0..index_list.len() {
                v.insert(i + offset, inserted_hashes[i]);
            }
            inserted_hashes.clear();
            // hash_seg: recomputed hash number for the current level
            let hash_seg = (v.len() as f64 / fanout as f64).ceil() as usize;
            
            for j in 0..hash_seg {
                let start_idx = j * fanout;
                let end_idx = if start_idx + fanout > v.len() {
                    v.len() - 1
                } else {
                    start_idx + fanout - 1
                };
                let sub_hash_vec = &v[start_idx ..= end_idx];
                let h = compute_concatenate_hash(sub_hash_vec);
                inserted_hashes.push(h);
            }
            for index in index_list.iter_mut() {
                (*index) /= fanout;
            }
            index_list.dedup();
        }
        assert_eq!(inserted_hashes.len(), 1);
        return inserted_hashes[0];
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpperMHTRangeProof {
    pub index_list: [usize; 2], // include left and right position
    pub p: Vec<Vec<H256>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    #[test]
    fn test_split_string() {
        let file_name = "run3-level2.dat";
        let parts: Vec<&str> = file_name.split(".").collect();
        let s = format!("{}-tmp_hash_file.dat", parts[0]);
        println!("{}", s);
    }

    #[test]
    fn test_write_read_merkle_offset() {
        let n = 12345u64;
        let offset_file_name = "offset.dat";
        let mut merkle_offset_writer = MerkleOffsetPageWriter::new(offset_file_name);
        for i in 1..=n {
            merkle_offset_writer.append(2*i);
        }
        merkle_offset_writer.finalize();

        let mut merkle_offset_reader = MerkleOffsetPageReader::new(offset_file_name);
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..n {
            let offset = merkle_offset_reader.read_merkle_offset(run_id, i as usize, &mut cache_manager);
            assert_eq!(offset, (i+1)*2);
        }
        assert_eq!(n as u32, merkle_offset_reader.num_of_merkle_offset);
    }

    #[test]
    fn test_write_read_upper_merkle_tree() {
        let n = 12345u64;
        let fanout = 4;
        let offset_file_name = "offset.dat";
        let upper_mht_file_name = "upper_mht.dat";
        let mut rng = StdRng::seed_from_u64(1);
        let hashes: Vec<H256> = (0..n).into_iter().map(|_| H256::random_using(&mut rng)).collect();
        let mut upper_mht_writer = UpperMHTWriter::new(offset_file_name, upper_mht_file_name, fanout);
        for (i, h) in hashes.iter().enumerate() {
            upper_mht_writer.append_upper_mht_offset_and_hash(i as u64, *h);
        }
        upper_mht_writer.finalize_write_mht_offset();


        let mut merkle_offset_reader = MerkleOffsetPageReader::new(offset_file_name);
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..n {
            let offset = merkle_offset_reader.read_merkle_offset(run_id, i as usize, &mut cache_manager);
            assert_eq!(offset, i);
        }
        assert_eq!(n as u32, merkle_offset_reader.num_of_merkle_offset);
    }
}