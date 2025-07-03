use chrono::Utc;
use eth_execution_engine::common::write_trait::BackendWriteTrait;
use eth_execution_engine::send_tx::{create_deploy_tx, create_call_tx, ContractArg, YCSB};
use eth_execution_engine::tx_executor::{exec_tx, test_batch_exec_tx, Backend};
use eth_execution_engine::common::{tx_req::TxRequest, nonce::Nonce};
use eth_execution_engine::{cole_index_backend::ColeIndexBackend, cole_plus_backend::ColePlusBackend};
use primitive_types::H160;
use utils::{types::{Address, AddrKey}, config::Configs};
use rand::prelude::*;
use std::collections::BTreeMap;
use std::{fs::OpenOptions, io::{Read, BufReader, Write}, path::Path, sync::Mutex, str};
use json::{self, object};
use anyhow::{anyhow, Ok, Result};

#[derive(Default, Debug)]
pub struct ReadParams {
    pub index_name: String,
    pub scale: usize,
    pub ycsb_path: String,
    pub ycsb_base_row_number: usize,
    pub tx_in_block: usize,
    pub db_path: String,
    pub mem_size: usize,
    pub size_ratio: usize,
    pub epsilon: usize,
    pub mht_fanout: usize,
    pub result_path: String,
}

impl ReadParams {
    pub fn from_json_file(path: &str) -> ReadParams {
        let mut file = OpenOptions::new().read(true).open(path).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let json_data = json::parse(data.as_str()).unwrap();
        let mut params = ReadParams::default();
        if json_data.has_key("index_name") {
            params.index_name = json_data["index_name"].to_string();
        }
        if json_data.has_key("scale") {
            params.scale = json_data["scale"].as_usize().unwrap();
        }
        if json_data.has_key("ycsb_path") {
            params.ycsb_path = json_data["ycsb_path"].to_string();
        }
        if json_data.has_key("ycsb_base_row_number") {
            params.ycsb_base_row_number = json_data["ycsb_base_row_number"].as_usize().unwrap();
        }
        if json_data.has_key("tx_in_block") {
            params.tx_in_block = json_data["tx_in_block"].as_usize().unwrap();
        }
        if json_data.has_key("db_path") {
            params.db_path = json_data["db_path"].to_string();
        }
        if json_data.has_key("mem_size") {
            params.mem_size = json_data["mem_size"].as_usize().unwrap();
        }
        if json_data.has_key("size_ratio") {
            params.size_ratio = json_data["size_ratio"].as_usize().unwrap();
        }
        if json_data.has_key("epsilon") {
            params.epsilon = json_data["epsilon"].as_usize().unwrap();
        }
        if json_data.has_key("mht_fanout") {
            params.mht_fanout = json_data["mht_fanout"].as_usize().unwrap();
        }
        if json_data.has_key("result_path") {
            params.result_path = json_data["result_path"].to_string();
        }
        return params;
    }
}

pub trait ReadTestTrait {
    fn read_latest_on_disk(&mut self, key: AddrKey) -> (usize, u32);
    fn print_struct(&mut self);
}

impl<'a> ReadTestTrait for ColeIndexBackend<'a> {
    fn read_latest_on_disk(&mut self, key: AddrKey) -> (usize, u32) {
        let start = std::time::Instant::now();
        let (num_runs, _) = self.states.search_latest_state_value_on_disk(key).unwrap();
        let elapse = start.elapsed().as_nanos();
        (elapse as usize, num_runs)
    }
    fn print_struct(&mut self) {
        self.states.print_structure_info();
    }
}

impl<'a> ReadTestTrait for ColePlusBackend<'a> {
    fn read_latest_on_disk(&mut self, key: AddrKey) -> (usize, u32) {
        let start = std::time::Instant::now();
        let (num_runs, _) = self.states.search_latest_state_value_on_disk(key).unwrap();
        let elapse = start.elapsed().as_nanos();
        (elapse as usize, num_runs)
    }
    fn print_struct(&mut self) {
        self.states.print_structure_info();
    }
}

pub fn build_db(params: &ReadParams, backend: &mut (impl BackendWriteTrait + Backend)) -> (u32, Vec<AddrKey>) {
    let caller_address = Address::from(H160::from_low_u64_be(1));
    let contract_arg = ContractArg::KVStore;
    let yscb_path = &params.ycsb_path;
    let file = OpenOptions::new().read(true).open(yscb_path).unwrap();
    YCSB.set(Mutex::new(BufReader::new(file))).map_err(|_e| anyhow!("Failed to set YCSB.")).unwrap();
    // deploy contract
    let mut block_id = 1;
    let (contract_address, tx_req) = create_deploy_tx(contract_arg, caller_address, Nonce::from(0));
    exec_tx(tx_req, caller_address, block_id, backend);
    println!("finish deploy contract");
    // run transactions to build db
    let mut rng = StdRng::seed_from_u64(1);
    let tx_in_block = params.tx_in_block;
    let n = params.scale;
    let mut build_requests_per_block = Vec::<TxRequest>::new();
    let mut state = BTreeMap::new();
    let base_row_num = params.ycsb_base_row_number;
    for i in 0..base_row_num {
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
        build_requests_per_block.push(call_tx_req);
        if build_requests_per_block.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            let s = test_batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, backend);
            state.extend(s);
            block_id += 1;
            //clear the requests for the next round
            build_requests_per_block.clear();
        }
    }
    if !build_requests_per_block.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        let s = test_batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, backend);
        state.extend(s);
        block_id += 1;
        build_requests_per_block.clear();
    }
    println!("finish build base");
    let mut requests = Vec::<TxRequest>::new();
    let result_path = &params.result_path;
    let base = format!("{}-{}-{}k-fan{}-ratio{}-mem{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.size_ratio, params.mem_size);
    let mut timestamp_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-ts.json", base)).unwrap();
    for i in 0..n {
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
        requests.push(call_tx_req);
        if requests.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            let now = Utc::now();
            let start_ts: i64 = now.timestamp_nanos_opt().unwrap();
            let s = test_batch_exec_tx(requests.clone(), caller_address, block_id, backend);
            let now = Utc::now();
            let end_ts: i64 = now.timestamp_nanos_opt().unwrap();
            let elapse = end_ts - start_ts;
            let ts_result_str = object! {
                block_id: block_id,
                start_ts: start_ts,
                end_ts: end_ts,
                elapse: elapse,
            }.dump();
            write!(timestamp_file, "{}\n", ts_result_str).unwrap();
            state.extend(s);
            block_id += 1;
            //clear the requests for the next round
            requests.clear();
        }
    }

    if !requests.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        let now = Utc::now();
        let start_ts: i64 = now.timestamp_nanos_opt().unwrap();
        let s = test_batch_exec_tx(requests.clone(), caller_address, block_id, backend);
        let now = Utc::now();
        let end_ts: i64 = now.timestamp_nanos_opt().unwrap();
        let elapse = end_ts - start_ts;
        let ts_result_str = object! {
            block_id: block_id,
            start_ts: start_ts,
            end_ts: end_ts,
            elapse: elapse,
        }.dump();
        write!(timestamp_file, "{}\n", ts_result_str).unwrap();
        state.extend(s);
        block_id += 1;
        //clear the requests for the next round
        requests.clear();
    } else {
        block_id -= 1;
    }
    timestamp_file.flush().unwrap();
    // backend.print_in_mem_tree();
    let state_key_vec: Vec<AddrKey> = state.into_keys().collect();
    return (block_id, state_key_vec);
}

pub fn cole_index_backend_read_test(params: &ReadParams) -> Result<()> {
    let result_path = &params.result_path;
    let base = format!("{}-{}-{}k-mem{}", result_path, params.index_name, params.scale/1000, params.mem_size);
    let mut read_test_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-read-test.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio, false);
    let mut backend = ColeIndexBackend::new(&configs, db_path);
    let (block_id, mut state_keys) = build_db(params, &mut backend);
    state_keys.sort();
    println!("after build db, block_id: {}", block_id);
    for addr in state_keys {
        let (read_time, num_read_runs) = backend.read_latest_on_disk(addr);
        let record = object! {
            addr: format!("{:?}", addr),
            read_time: read_time,
            num_read_runs: num_read_runs,
        }.dump();
        write!(read_test_file, "{}\n", record).unwrap();
    }
    read_test_file.flush().unwrap();
    // backend.print_struct();
    Ok(())
}

pub fn cole_plus_backend_read_test(params: &ReadParams) -> Result<()> {
    let result_path = &params.result_path;
    let base = format!("{}-{}-{}k-fan{}", result_path, params.index_name, params.scale/1000, params.mht_fanout);
    let mut read_test_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-read-test.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio, false);
    let mut backend = ColePlusBackend::new(&configs, db_path);
    let (block_id, mut state_keys) = build_db(params, &mut backend);
    state_keys.sort();
    println!("after build db, block_id: {}", block_id);
    for addr in state_keys {
        let (read_time, num_read_runs) = backend.read_latest_on_disk(addr);
        let record = object! {
            addr: format!("{:?}", addr),
            read_time: read_time,
            num_read_runs: num_read_runs,
        }.dump();
        write!(read_test_file, "{}\n", record).unwrap();
    }
    read_test_file.flush().unwrap();
    // backend.print_struct();
    Ok(())
}