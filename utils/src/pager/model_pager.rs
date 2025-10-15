use anyhow::{anyhow, Ok, Result};
use crate::{cacher::{CacheManager, CacheManagerOld}, models::CompoundKeyModel, types::CompoundKey, File, OpenOptions};
use std::{io::{Read, Seek, SeekFrom, Write}, os::unix::fs::FileExt};
use super::{Page, PAGE_SIZE, MAX_NUM_MODEL_IN_PAGE};
use crate::models::{ModelGenerator, fetch_model_and_predict};
pub const TMP_MODEL_FILE_NAME: &str = "tmp_model_file.dat";
use std::cmp::{max, min};
/* A helper structure to keep a collection of models and their sharing model_level
 */
#[derive(Debug, Clone)]
pub struct ModelCollections {
    pub v: Vec<CompoundKeyModel>, // vector of models
    pub model_level: u32, 
}

impl ModelCollections {
    pub fn new() -> Self {
        Self {
            v: vec![],
            model_level: 0,
        }
    }

    pub fn init_with_model_level(model_level: u32) -> Self {
        Self {
            v: vec![],
            model_level,
        }
    }
}

/* A helper that writes a level of models into a file with a sequence of pages
 */
pub struct ModelPageWriter {
    pub file: File, // file object of the corresponding index file
    pub latest_model_collection: ModelCollections, // a preparation vector to obsorb the streaming models which are not persisted in the file yet
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
}

impl ModelPageWriter {
    /* Initialize the writer using a given file_name
     */
    pub fn create(file_name: &str, model_level: u32) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            latest_model_collection: ModelCollections::init_with_model_level(model_level),
            num_stored_pages: 0,
        }
    }

    /* Streamingly add the model to the latest collection, if the collection is full, flush it to the file
     */
    pub fn append(&mut self, model: CompoundKeyModel) {
        // add the model
        self.latest_model_collection.v.push(model);
        if self.latest_model_collection.v.len() == MAX_NUM_MODEL_IN_PAGE {
            // vector is full, should be added to a page and flushed the page to the file
            self.flush();
        }
    }

    /* Flush the vector in latest update page to the last page in the value file
     */
    pub fn flush(&mut self) {
        if self.latest_model_collection.v.len() != 0 {
            // first put the vector into a page
            let page = Page::from_model_vec(&self.latest_model_collection.v, self.latest_model_collection.model_level);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.write_all(&page.block).unwrap();
            // clear the vector
            self.latest_model_collection.v.clear();
            self.num_stored_pages += 1;
        }
    }

    pub fn to_model_reader(self) -> ModelPageReader {
        let num_stored_pages = self.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        ModelPageReader {
            file: self.file,
            num_stored_pages,
        }
    }
}

/* A helper to read the models from the file
 */
pub struct ModelPageReader {
    pub file: File, // file object of the corresponding index file
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
}

impl ModelPageReader {
    /* Load the reader from a given file
     num_stored_pages are derived from the file
     */
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        Self {
            file,
            num_stored_pages,
        }
    }

    /* Load the deserialized vector of the page from the file at given page_id
     */
    pub fn read_deser_page_at(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> ModelCollections {
        // first check whether the cache contains the page
        let r = cache_manager.read_model_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let page = r.unwrap();
            let v = page.to_model_vec();
            return v;
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let v = page.to_model_vec();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_model_cache(run_id, page_id, page);
            return v;
        }
    }
    // given the search key, return the predicted page id in the state file
    pub fn get_pred_state_page_id(&mut self, run_id: u32, search_key: &CompoundKey, verbose: bool, cache_manager: &mut CacheManager) -> usize {
        // first load the last page and find the model that covers the search key
        let last_page_id = self.num_stored_pages - 1;
        let top_model_collection = self.read_deser_page_at(run_id, last_page_id, cache_manager);
        let (model_v, mut model_level) = (top_model_collection.v, top_model_collection.model_level);
        let mut pred_pos = fetch_model_and_predict(&model_v, search_key);
        if verbose {
            println!("search key: {:?}", search_key);
            println!("last page model pred pos: {}", pred_pos);
        }
        if model_level == 0 {
            // the last page stores the lowest level
            return pred_pos;
        } else {
            // first search the model in the model_v and then determine the predicted page id
            model_level -= 1;
            pred_pos = self.query_page_model(run_id, pred_pos, search_key, model_level, verbose, cache_manager).unwrap();
            if verbose {
                println!("model level: {}, pred_pos: {}", model_level, pred_pos);
            }
            while model_level != 0 {
                model_level -= 1;
                pred_pos = self.query_page_model(run_id, pred_pos, search_key, model_level, verbose, cache_manager).unwrap();
                if verbose {
                    println!("model level: {}, pred_pos: {}", model_level, pred_pos);
                }
            }
            return pred_pos;
        }
    }
    // query the model page: given the page_id and search key, ouput the predicted page id using the "right model" in the page
    fn query_page_model(&mut self, run_id: u32, page_id: usize, search_key: &CompoundKey, model_level: u32, verbose: bool, cache_manager: &mut CacheManager) -> Result<usize> {
        let mut model_v = Vec::<CompoundKeyModel>::new();
        // first load page_id's models to model_v
        let collection = self.read_deser_page_at(run_id, page_id, cache_manager);
        // get first and last model's key
        let first_compound_key = collection.v.first().unwrap().start;
        let last_compound_key = collection.v.last().unwrap().start;
        if verbose {
            println!("first compound key: {:?}, last comopund key: {:?}", first_compound_key, last_compound_key);
        }
        if search_key < &first_compound_key {
            if verbose {
                println!("load previous page");
            }
            // load the previous page if exist, the previous page should have the same model level
            if page_id >= 1 {
                let prev_page_id = page_id - 1;
                let pre_collection = self.read_deser_page_at(run_id, prev_page_id, cache_manager);
                if pre_collection.model_level == model_level {
                    model_v.extend(&pre_collection.v);
                }
                if collection.model_level == model_level {
                    model_v.extend(&collection.v);
                }
                if verbose {
                    println!("prev page exist");
                    println!("model_v: {:?}", model_v);
                }
            } else {
                if collection.model_level == model_level {
                    model_v.extend(&collection.v);
                }
            }
        } else if search_key > &last_compound_key {
            if verbose {
                println!("load next page");
            }
            // load the next page if exist, the next page should have the same model level
            if page_id + 1 < self.num_stored_pages {
                let next_page_id = page_id + 1;
                let next_collection = self.read_deser_page_at(run_id, next_page_id, cache_manager);
                if collection.model_level == model_level {
                    model_v.extend(&collection.v);
                }
                if next_collection.model_level == model_level {
                    model_v.extend(&next_collection.v);
                }
                if verbose {
                    println!("next page exist");
                    println!("model_v: {:?}", model_v);
                }
            } else {
                if collection.model_level == model_level {
                    model_v.extend(&collection.v);
                }
            }
        } else {
            if collection.model_level == model_level {
                model_v.extend(&collection.v);
            }
        }
        if model_v.is_empty() {
            return Err(anyhow!("wrong model prediction: model level error"));
        }
        let pred_pos = fetch_model_and_predict(&model_v, search_key);
        return Ok(pred_pos);
    }

    // the remaining two functions are for old cole design
    /* Load the deserialized vector of the page from the file at given page_id
     */
    pub fn read_deser_page_at_old_version(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManagerOld) -> ModelCollections {
        // first check whether the cache contains the page
        let r = cache_manager.read_model_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let page = r.unwrap();
            let v = page.to_model_vec();
            return v;
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut bytes = [0u8; PAGE_SIZE];
            self.file.seek(SeekFrom::Start(offset as u64)).unwrap();
            self.file.read_exact(&mut bytes).unwrap();
            let page = Page::from_array(bytes);
            let v = page.to_model_vec();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_model_cache(run_id, page_id, page);
            return v;
        }
    }
    /* Query Models in the model file
     */
    pub fn get_pred_state_pos(&mut self, run_id: u32, search_key: &CompoundKey, epsilon: i64, cache_manager: &mut CacheManagerOld) -> usize {
        /* First load the last page and find the model that covers the search key
         */
        let last_page_id = self.num_stored_pages - 1;
        let top_model_collection = self.read_deser_page_at_old_version(run_id, last_page_id, cache_manager);
        let (model_v, mut model_level) = (top_model_collection.v, top_model_collection.model_level);
        let mut pred_pos = fetch_model_and_predict(&model_v, search_key);
        if model_level == 0 {
            // the last page stores the lowest level
            return pred_pos;
        } else {
            // first search the model in the model_v and then determine the predicted page id range
            model_level -= 1;
            pred_pos = self.query_model_old_design(run_id, pred_pos, epsilon, search_key, model_level, cache_manager);
            while model_level != 0 {
                model_level -= 1;
                pred_pos = self.query_model_old_design(run_id, pred_pos, epsilon, search_key, model_level, cache_manager);
            }
            return pred_pos;
        }
    }

    fn query_model_old_design(&mut self, run_id: u32, pred_pos: usize, epsilon: i64, search_key: &CompoundKey, model_level: u32, cache_manager: &mut CacheManagerOld) -> usize {
        let pred_page_id_lb = max(0, (pred_pos as i64 - epsilon - 1) / MAX_NUM_MODEL_IN_PAGE as i64) as usize;
        let pred_page_id_ub = min(self.num_stored_pages - 1, (pred_pos + epsilon as usize + 1) / MAX_NUM_MODEL_IN_PAGE);
        let mut model_v = Vec::<CompoundKeyModel>::new();
        for page_id in pred_page_id_lb ..= pred_page_id_ub {
            let collection = self.read_deser_page_at_old_version(run_id, page_id, cache_manager);
            if collection.model_level == model_level {
                model_v.extend(&collection.v);
            }
        }
        let pred_pos = fetch_model_and_predict(&model_v, search_key);
        return pred_pos;
    }
}

/* A model constructor that generates and appends models to the file in a streaming fashion
 */
pub struct ModelConstructor {
    pub output_model_writer: ModelPageWriter, // a writer of the output model file
    pub lowest_level_model_generator: ModelGenerator, // a model generator of the lowest level (learn input is the states)
    pub key_pos: usize, // the position of the input state
    pub temp_file_name: String,
}

impl ModelConstructor {
    /* Initiate the constructor with the output model file name 
     */
    pub fn new(output_file_name: &str) -> Self {
        // create the output model writer
        let output_model_writer = ModelPageWriter::create(output_file_name, 0);
        // initiate the model generator for the lowest level
        let lowest_level_model_generator = ModelGenerator::new(1);
        let split_string_vec: Vec<&str> = output_file_name.split(".").collect();
        let base_path = split_string_vec[1].split("/").last().unwrap();
        let temp_file_name = format!("{}-{}", base_path, TMP_MODEL_FILE_NAME);
        Self {
            output_model_writer,
            lowest_level_model_generator,
            key_pos: 0,
            temp_file_name,
        }
    }

    /* Streaminly append the key to the model generator for the lowest level
     */
    pub fn append_state_key(&mut self, key: &CompoundKey) {
        let pos = self.key_pos;
        let r = self.lowest_level_model_generator.append(key, pos);
        if r == false {
            // finalize the model since the new coming key cannot fit the model within the prediction error bound
            let model = self.lowest_level_model_generator.finalize_model();
            // write the model to the output model writer (can be kept in the latest page cache in memory or be flushed to the file)
            self.output_model_writer.append(model);
            // re-insert the key position to the model generator since the previous insertion fails
            self.lowest_level_model_generator.append(key, pos);
        }
        self.key_pos += 1;
    }

    /* Finalize the append of the key-pos
       End the insertion of the lowest_level_model_generator: finalize the model, append it to the output_model_writer, and flush it to the file
       Recursively build the models upon the previous level and append them to the file in a streaming fashion.
     */
    pub fn finalize_append(&mut self) {
        /* First finalize the lowest level models
         */
        if !self.lowest_level_model_generator.is_hull_empty() {
            let model = self.lowest_level_model_generator.finalize_model();
            self.output_model_writer.append(model);
        }
        self.output_model_writer.flush();
        /*
        recursively construct models in the upper levels
        */
        let output_model_writer = &mut self.output_model_writer;
        // n is the number of page in the previous model level
        let mut n = output_model_writer.num_stored_pages;
        let mut model_level = 0;
        while n > 1 {
            // n > 1 means we should build an upper level models since the top level models should be kept in a single page
            // increment the model_level
            model_level += 1;
            // start_page_id is the id of the starting page that the upper level is learned from
            let start_page_id = output_model_writer.num_stored_pages - n;
            // initiate a model generator for the upper level models
            let mut model_generator = ModelGenerator::new(1);
            // create a temporary model writer for keeping the upper level models
            let mut tmp_model_writer = ModelPageWriter::create(&self.temp_file_name, model_level);
            // pos is the position of the learned input of the upper level models
            let mut pos = start_page_id;
            for page_id in start_page_id .. output_model_writer.num_stored_pages {
                // read the model page from the file in the output_model_writer at the corresponding offset
                let offset = page_id * PAGE_SIZE;
                let mut bytes = [0u8; PAGE_SIZE];
                output_model_writer.file.seek(SeekFrom::Start(offset as u64)).unwrap();
                output_model_writer.file.read_exact(&mut bytes).unwrap();
                let page = Page::from_array(bytes);
                // deserialize the models from the page, these are seen as the learned input of the upper level models
                let first_key_in_page = page.to_model_vec().v.first().unwrap().start;
                let r = model_generator.append(&first_key_in_page, pos);
                if r == false {
                    let output_model = model_generator.finalize_model();
                    // write the output model to the temporary model write of the upper models
                    tmp_model_writer.append(output_model);
                    model_generator.append(&first_key_in_page, pos);
                }
                pos += 1;
            }

            // handle the rest of the points in the hull
            if !model_generator.is_hull_empty() {
                let output_model = model_generator.finalize_model();
                tmp_model_writer.append(output_model);
            }
            // flush the temporary model writer to the temporary file
            tmp_model_writer.flush();
            // update n as the number of page of the temporary model writer
            n = tmp_model_writer.num_stored_pages;
            // concatenate the content of temporary model file to the output model file
            concatenate_file_a_to_file_b(&mut tmp_model_writer.file, &mut output_model_writer.file);
            // update the number of pages in the output model file
            output_model_writer.num_stored_pages += tmp_model_writer.num_stored_pages;

            // drop the tmp_model_writer and remove the temporary file
            drop(tmp_model_writer);
            std::fs::remove_file(&self.temp_file_name).unwrap();
        }
    }
}

// old design of stream_model_constructor in cole
pub struct StreamModelConstructor {
    pub output_model_writer: ModelPageWriter, // a writer of the output model file
    pub lowest_level_model_generator: ModelGenerator, // a model generator of the lowest level (learn input is the states)
    pub epsilon: i64, // an upper-bound model prediction error
    pub state_pos: usize, // the position of the input state
    pub temp_file_name: String,
}

impl StreamModelConstructor {
    /* Initiate the constructor with the output model file name and the upper error bound
     */
    pub fn new(output_file_name: &str, epsilon: i64) -> Self {
        // create the output model writer
        let output_model_writer = ModelPageWriter::create(output_file_name, 0);
        // initiate the model generator for the lowest level
        let lowest_level_model_generator = ModelGenerator::new(epsilon);
        let split_string_vec: Vec<&str> = output_file_name.split(".").collect();
        let base_path = split_string_vec[1].split("/").last().unwrap();
        let temp_file_name = format!("{}-{}", base_path, TMP_MODEL_FILE_NAME);
        Self {
            output_model_writer,
            lowest_level_model_generator,
            epsilon,
            state_pos: 0,
            temp_file_name
        }
    }

    /* Streaminly append the key to the model generator for the lowest level
     */
    pub fn append_state_key(&mut self, key: &CompoundKey) {
        let pos = self.state_pos;
        let r = self.lowest_level_model_generator.append(key, pos);
        if r == false {
            // finalize the model since the new coming key cannot fit the model within the prediction error bound
            let model = self.lowest_level_model_generator.finalize_model();
            // write the model to the output model writer (can be kept in the latest page cache in memory or be flushed to the file)
            self.output_model_writer.append(model);
            // re-insert the key position to the model generator since the previous insertion fails
            self.lowest_level_model_generator.append(key, pos);
        }
        self.state_pos += 1;
    }

    /* Finalize the append of the key-pos
       End the insertion of the lowest_level_model_generator: finalize the model, append it to the output_model_writer, and flush it to the file
       Recursively build the models upon the previous level and append them to the file in a streaming fashion.
     */
    pub fn finalize_append(&mut self) {
        /* First finalize the lowest level models
         */
        if !self.lowest_level_model_generator.is_hull_empty() {
            let model = self.lowest_level_model_generator.finalize_model();
            self.output_model_writer.append(model);
        }
        self.output_model_writer.flush();

        /*
        recursively construct models in the upper levels
        */
        
        let output_model_writer = &mut self.output_model_writer;
        // n is the number of page in the previous model level
        let mut n = output_model_writer.num_stored_pages;
        let mut model_level = 0;
        while n > 1 {
            // n > 1 means we should build an upper level models since the top level models should be kept in a single page
            // increment the model_level
            model_level += 1;
            // start_page_id is the id of the starting page that the upper level is learned from
            let start_page_id = output_model_writer.num_stored_pages - n;
            // initiate a model generator for the upper level models
            let mut model_generator = ModelGenerator::new(self.epsilon);
            // create a temporary model writer for keeping the upper level models
            let mut tmp_model_writer = ModelPageWriter::create(&self.temp_file_name, model_level);
            // pos is the position of the learned input of the upper level models
            let mut pos = start_page_id * MAX_NUM_MODEL_IN_PAGE;
            
            for page_id in start_page_id .. output_model_writer.num_stored_pages {
                // read the model page from the file in the output_model_writer at the corresponding offset
                let offset = page_id * PAGE_SIZE;
                let mut bytes = [0u8; PAGE_SIZE];
                output_model_writer.file.read_exact_at(&mut bytes, offset as u64).unwrap();
                let page = Page::from_array(bytes);
                // deserialize the models from the page, these are seen as the learned input of the upper level models
                let input_models = page.to_model_vec().v;
                for input_model in input_models {
                    let r = model_generator.append(&input_model.start, pos);
                    if r == false {
                        let output_model = model_generator.finalize_model();
                        // write the output model to the temporary model write of the upper models
                        tmp_model_writer.append(output_model);
                    }
                    model_generator.append(&input_model.start, pos);
                    pos += 1;
                }
            }
            
            // handle the rest of the points in the hull
            if !model_generator.is_hull_empty() {
                let output_model = model_generator.finalize_model();
                tmp_model_writer.append(output_model);
            }
            // flush the temporary model writer to the temporary file
            tmp_model_writer.flush();
            // update n as the number of page of the temporary model writer
            n = tmp_model_writer.num_stored_pages;
            // concatenate the content of temporary model file to the output model file
            concatenate_file_a_to_file_b(&mut tmp_model_writer.file, &mut output_model_writer.file);
            // update the number of pages in the output model file
            output_model_writer.num_stored_pages += tmp_model_writer.num_stored_pages;
            
            // drop the tmp_model_writer and remove the temporary file
            drop(tmp_model_writer);
            std::fs::remove_file(&self.temp_file_name).unwrap();
        }
    }
}

pub fn concatenate_file_a_to_file_b(file_a: &mut File, file_b: &mut File) {
    // rewind the cursor of file_a to the start
    file_a.rewind().unwrap();
    let l = file_b.metadata().unwrap().len();
    // seek the cursor of file_b to the end
    file_b.seek(SeekFrom::Start(l)).unwrap();
    // copy the content in file_a to the end of file_b
    std::io::copy(file_a, file_b).unwrap();
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use primitive_types::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use crate::{pager::state_pager::StatePageWriter, types::{AddrKey, Address, CompoundKey, StateKey, StateValue}};
    use super::*;
    #[test]
    fn test_model_pager() {
        let n = 10000;
        let mut writer = ModelPageWriter::create("model.dat", 0);
        let mut model_vec = Vec::<CompoundKeyModel>::new();
        let mut rng = StdRng::seed_from_u64(1);
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let key = CompoundKey::new(AddrKey::new(Address(acc_addr), StateKey(state_addr)), version);
            let model = CompoundKeyModel {
                start: key,
                slope: 1.0,
                intercept: 2.0,
                last_index: i as u32,
            };
            model_vec.push(model.clone());
            writer.append(model);
        }
        writer.flush();

        let mut reader = ModelPageReader::load("model.dat");
        let mut model_collections: Vec::<CompoundKeyModel> = Vec::new();
        let total_page_num = reader.num_stored_pages;
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..total_page_num {
            let collections = reader.read_deser_page_at(run_id, i, &mut cache_manager);
            model_collections.extend(&collections.v);
        }
        assert_eq!(model_collections, model_vec);
    }

    #[test]
    fn test_streaming_model() {
        let n = 1000000;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<CompoundKey>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let key = CompoundKey::new(AddrKey::new(Address(acc_addr), StateKey(state_addr)), version);
            keys.push(key);
        }
        keys.sort();
        let start = std::time::Instant::now();
        let mut stream_model_constructor = ModelConstructor::new("model.dat");
        let mut point_vec = Vec::<(CompoundKey, usize)>::new();
        for (i, key) in keys.iter().enumerate() {
            stream_model_constructor.append_state_key(key);
            point_vec.push((*key, i));
        }
        stream_model_constructor.finalize_append();
        let elapse = start.elapsed().as_nanos();
        println!("avg construct time: {}", elapse / n as u128);
        let mut reader = ModelPageReader::load("model.dat");
        let num_of_pages = reader.num_stored_pages;
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        for i in 0..num_of_pages {
            let page_collections = reader.read_deser_page_at(run_id, i, &mut cache_manager);
            let page_collection_len = page_collections.v.len();
            // println!("page collections len: {}", page_collection_len);
            let mut model_set = BTreeSet::new();
            for model in &page_collections.v {
                model_set.insert(model.start);
            }
            let page_collection_set_len = model_set.len();
            // println!("page collection set len: {}", page_collection_set_len);
            assert_eq!(page_collection_len, page_collection_set_len);
            // println!("page {} model: {:?}", i, page_collections);
        }
        println!("check points");
        for point in point_vec {
            let (key, true_pos) = (point.0, point.1);
            let pred_pos = reader.get_pred_state_page_id(run_id, &key, false, &mut cache_manager);
            if (true_pos as f64 - pred_pos as f64).abs() > 2.0 {
                println!("true_pos: {}, pred_pos: {}, diff: {}", true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
                reader.get_pred_state_page_id(run_id, &key, true, &mut cache_manager);
            }
        }
    }

    fn generate_addr(rng: &mut StdRng) -> AddrKey {
        AddrKey { addr: Address(H160::random_using(rng)), state_key: StateKey(H256::random_using(rng)) }
    }

    #[test]
    fn test_state_and_model() {
        let min_state = (AddrKey::new(Address(H160::from_low_u64_be(0)), StateKey(H256::from_low_u64_be(0))), 0u32, StateValue(H256::from_low_u64_be(0)));
        let max_state = (AddrKey::new(Address(H160::from_slice(&vec![255u8;20])), StateKey(H256::from_slice(&vec![255u8;32]))), u32::MAX, StateValue(H256::from_low_u64_be(0)));
        let n = 1234;
        let mut rng = StdRng::seed_from_u64(1);
        let mut addr_vec: Vec<AddrKey> = (1..=n).map(|_| generate_addr(&mut rng)).collect();
        addr_vec.sort();
        let ver_num =123u32;
        let mut states = Vec::<(AddrKey, u32, StateValue)>::new();
        for addr in &addr_vec {
            for i in 1..=ver_num {
                let ver = i;
                let value = StateValue(H256::from_low_u64_be(i as u64));
                states.push((*addr, ver, value));
            }
        }

        let file_name = "state.dat";
        let model_file_name = "model.dat";
        let mut state_page_writer = StatePageWriter::create(&file_name);
        let mut model_constructor = ModelConstructor::new(model_file_name);
        let mut page_cnt: usize = 0;
        let mut key_page_map = BTreeMap::new();
        state_page_writer.append((min_state.0, min_state.1, min_state.2));
        for state in &states {
            let model_key = state_page_writer.append(*state);
            if let Some(inner_model_key) = model_key {
                model_constructor.append_state_key(&inner_model_key);
                page_cnt += 1;
            }
            key_page_map.insert(CompoundKey::new(state.0, state.1), page_cnt - 1);
        }
        let model_key = state_page_writer.append((max_state.0, max_state.1, max_state.2));
        if let Some(inner_model_key) = model_key {
            model_constructor.append_state_key(&inner_model_key);
            page_cnt += 1;
        }
        key_page_map.insert(CompoundKey::new(max_state.0, max_state.1), page_cnt - 1);
        state_page_writer.flush();
        model_constructor.finalize_append();

        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        let mut model_reader = ModelPageReader::load(model_file_name);
/*         for i in 0..model_reader.num_stored_pages {
            let model_collections = model_reader.read_deser_page_at(run_id, i, &mut cache_manager);
            println!("model level: {}", model_collections.model_level);
            println!("models: {:?}", model_collections.v);
        } */
        println!("check points");
        for state in &states {
            let search_key = CompoundKey::new(state.0, state.1);
            let pred_pos = model_reader.get_pred_state_page_id(run_id, &search_key, false, &mut cache_manager);
            let true_pos = *key_page_map.get(&search_key).unwrap();
            if (true_pos as f64 - pred_pos as f64).abs() > 2.0 {
                println!("search key: {:?}, true_pos: {}, pred_pos: {}, diff: {}", search_key, true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
                model_reader.get_pred_state_page_id(run_id, &search_key, true, &mut cache_manager);
                break;
            }
        }

        println!("check largest key");
        let max_key = CompoundKey::new(max_state.0, max_state.1);
        let pred_pos = model_reader.get_pred_state_page_id(run_id, &max_key, false, &mut cache_manager);
        println!("max key pred_pos: {}", pred_pos);
        let total_state_page = state_page_writer.num_stored_pages;
        println!("total state page: {}", total_state_page);
    }

    #[test]
    fn test_old_design_model_pager() {
        let n = 10000;
        let mut writer = ModelPageWriter::create("model.dat", 0);
        let mut model_vec = Vec::<CompoundKeyModel>::new();
        for i in 0..n {
            let acc_addr = H160::random();
            let state_addr = H256::random();
            let version = i as u32;
            let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
            let key = CompoundKey::new(addr_key, version);
            let model = CompoundKeyModel {
                start: key,
                slope: 1.0,
                intercept: 2.0,
                last_index: i as u32,
            };
            model_vec.push(model.clone());
            writer.append(model);
        }
        writer.flush();
        let mut reader = writer.to_model_reader();
        let mut cache_manager = CacheManagerOld::new();
        for j in 0..5 {
            // iteratively read the pages
            let start = std::time::Instant::now();
            for i in 0..n {
                let page_id = i / MAX_NUM_MODEL_IN_PAGE;
                let inner_page_pos = i % MAX_NUM_MODEL_IN_PAGE;
                let collections = reader.read_deser_page_at_old_version(0, page_id, &mut cache_manager);
                let state = collections.v[inner_page_pos];
                assert_eq!(state, model_vec[i]);
            }
            let elapse = start.elapsed().as_nanos() as usize / n;
            println!("round {}, read state time: {}", j, elapse);
        }
    }

    #[test]
    fn test_skip_model() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 10000;

        let mut keys = Vec::<CompoundKey>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
            let key = CompoundKey::new(addr_key, version);
            keys.push(key);
        }
        keys.sort();
        let start = std::time::Instant::now();
        let mut stream_model_constructor = ModelConstructor::new("model.dat");
        let mut point_vec = Vec::<(CompoundKey, usize)>::new();
        let skip = 46;
        for (i, key) in keys.iter().enumerate() {
            if i % skip == 0 {
                stream_model_constructor.append_state_key(key);
                point_vec.push((*key, i));
            }
        }
        stream_model_constructor.finalize_append();
        let elapse = start.elapsed().as_nanos();
        println!("avg construct time: {}", elapse / n as u128);

        let mut reader = ModelPageReader::load("model.dat");
        let run_id = 0;
        let mut cache_manager = CacheManager::new();
        println!("check points");
        let mut time_accum = 0;
        let point_vec_len = point_vec.len();
        for point in point_vec {
            let (key, true_pos) = (point.0, point.1);
            let start = std::time::Instant::now();
            let pred_pos = reader.get_pred_state_page_id(run_id, &key, false, &mut cache_manager);
            let elapse = start.elapsed().as_nanos();
            time_accum += elapse;
            if (true_pos as f64 - (pred_pos * skip) as f64).abs() > 2.0 * (skip as f64) {
                println!("true_pos: {}, pred_pos: {}, diff: {}", true_pos, (pred_pos * skip), (true_pos as f64 - (pred_pos * skip) as f64).abs());
                reader.get_pred_state_page_id(run_id, &key, true, &mut cache_manager);
            }
        }
        println!("avg pred time: {}", time_accum / point_vec_len as u128);
    }
    
    #[test]
    fn test_old_design_streaming_model() {
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 23;
        let n = 10000;
        
        let mut keys = Vec::<CompoundKey>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
            let key = CompoundKey::new(addr_key, version);
            keys.push(key);
        }
        keys.sort();
        let start = std::time::Instant::now();
        let mut stream_model_constructor = StreamModelConstructor::new("model.dat", epsilon);
        let mut point_vec = Vec::<(CompoundKey, usize)>::new();
        for (i, key) in keys.iter().enumerate() {
            stream_model_constructor.append_state_key(key);
            point_vec.push((*key, i));
        }
        stream_model_constructor.finalize_append();
        let elapse = start.elapsed().as_nanos();
        println!("avg construct time: {}", elapse / n as u128);
        let writer = stream_model_constructor.output_model_writer;
        let mut reader = writer.to_model_reader();
        let mut cache_manager = CacheManagerOld::new();
        let num_of_pages = reader.num_stored_pages;
        for i in 0..num_of_pages {
            let collection = reader.read_deser_page_at_old_version(0, i, &mut cache_manager);
            println!("collection size: {:?}, model_level: {:?}", collection.v.len(), collection.model_level);
        }

        
        let mut time_accum = 0;
        for point in point_vec {
            let key = point.0;
            let true_pos = point.1;
            let start = std::time::Instant::now();
            let pred_pos = reader.get_pred_state_pos(0, &key, epsilon, &mut cache_manager);
            let elapse = start.elapsed().as_nanos();
            time_accum += elapse;
            if (true_pos as f64 - pred_pos as f64).abs() > (epsilon + 1) as f64 {
                println!("true_pos: {}, pred_pos: {}, diff: {}", true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
            }
            // println!("true: {}, pred: {}, diff: {}", true_pos, pred_pos, (true_pos as f64 - pred_pos as f64).abs());
            // assert!((true_pos as f64 - pred_pos as f64).abs().floor() <= (epsilon + 1) as f64);
        }
        println!("avg pred time: {}", time_accum / n as u128);
    }
}