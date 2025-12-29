use eth_execution_engine::common::write_trait::BackendWriteTrait;
use eth_execution_engine::send_tx::{create_call_tx, create_deploy_tx, ContractArg, YCSB};
use eth_execution_engine::tx_executor::{exec_tx, batch_exec_tx, Backend};
use eth_execution_engine::common::{tx_req::TxRequest, nonce::Nonce};
use eth_execution_engine::cole_plus_backend::ColePlusBackend;
use utils::{types::Address, config::Configs};
use rand::prelude::*;
use std::{fs::OpenOptions, io::{Read, BufReader, Write}, path::Path, sync::Mutex, str};
use json::{self, object};
use anyhow::{anyhow, Ok, Result};
use chrono::prelude::*;
use primitive_types::H160;

#[derive(Default, Debug)]
pub struct ReOrgParams {
    pub index_name: String,
    pub contract_name: String,
    pub scale: usize,
    pub ycsb_path: String,
    pub ycsb_base_row_number: usize,
    pub eth_path: String,
    pub num_of_contract: usize,
    pub tx_in_block: usize,
    pub db_path: String,
    pub mem_size: usize,
    pub size_ratio: usize,
    pub epsilon: usize,
    pub mht_fanout: usize,
    pub result_path: String,
    pub keep_latest_block: bool,
    pub rewind_blocks: usize,
}

impl ReOrgParams {
    pub fn from_json_file(path: &str) -> ReOrgParams {
        let mut file = OpenOptions::new().read(true).open(path).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let json_data = json::parse(data.as_str()).unwrap();
        let mut params = ReOrgParams::default();
        if json_data.has_key("index_name") {
            params.index_name = json_data["index_name"].to_string();
        }
        if params.index_name.contains("archive") {
            params.keep_latest_block = false;
        }
        if params.index_name.contains("pruned") {
            params.keep_latest_block = true;
        }
        if json_data.has_key("contract_name") {
            params.contract_name = json_data["contract_name"].to_string();
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
        if json_data.has_key("eth_path") {
            params.eth_path = json_data["eth_path"].to_string();
        }
        if json_data.has_key("num_of_contract") {
            params.num_of_contract = json_data["num_of_contract"].as_usize().unwrap();
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
        if json_data.has_key("rewind_blocks") {
            params.rewind_blocks = json_data["rewind_blocks"].as_usize().unwrap();
        }
        return params;
    }
}

pub fn build_base_for_reorg(params: &ReOrgParams, backend: &mut (impl BackendWriteTrait + Backend)) {
    let caller_address = Address::from(H160::from_low_u64_be(1));
    let contract_arg = ContractArg::KVStore;
    // prepare for deploying contracts
    let mut contract_address_list = Vec::<Address>::new();
    let mut block_id = 1;
    let num_of_contract = params.num_of_contract;
    for i in 0..num_of_contract {
        let (contract_address, tx_req) = create_deploy_tx(contract_arg, caller_address, Nonce::from(i));
        contract_address_list.push(contract_address);
        exec_tx(tx_req, caller_address, block_id, backend);
    }
    println!("finish deploy contract");
    // prepare for building the base db for KVSTORE
    let mut rng = StdRng::seed_from_u64(1);
    let tx_in_block = params.tx_in_block;
    let n = params.scale;

    let smallbank_acc_num = n / 100;
    // pack up the testing transactions for each block and execute the block
    let mut requests = Vec::<TxRequest>::new();

    for i in 0..n {
        let contract_id = 0 % num_of_contract;
        let contract_address = contract_address_list[contract_id as usize];
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, smallbank_acc_num as usize);
        requests.push(call_tx_req);
        if requests.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            batch_exec_tx(requests.clone(), caller_address, block_id, backend);

            // if i == n - 1 {
            //     // should sleep 30 sec in case some threads do not end
            //     let thirty_sec = time::Duration::from_secs(30);
            //     thread::sleep(thirty_sec);
            // }

            
            /* let index_name = &params.index_name;
            if index_name.contains("cole") {
                let memory_cost = backend.memory_cost();
                let mem_result_str = object! {
                    block_id: block_id,
                    state_cache_size: memory_cost.state_cache_size,
                    model_cache_size: memory_cost.model_cache_size,
                    mht_cache_size: memory_cost.mht_cache_size,
                    filter_size: memory_cost.filter_size,
                }.dump();
                write!(memory_file, "{}\n", mem_result_str).unwrap();
            } */
            
            block_id += 1;
            //clear the requests for the next round
            requests.clear();
        }
    }

    if !requests.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        batch_exec_tx(requests.clone(), caller_address, block_id, backend);

        // should check storage size, sleep 30 sec in case some threads do not end
        // let thirty_sec = time::Duration::from_secs(30);
        // thread::sleep(thirty_sec);
        
        /* let index_name = &params.index_name;
        if index_name.contains("cole") {
            let memory_cost = backend.memory_cost();
            let mem_result_str = object! {
                block_id: block_id,
                state_cache_size: memory_cost.state_cache_size,
                model_cache_size: memory_cost.model_cache_size,
                mht_cache_size: memory_cost.mht_cache_size,
                filter_size: memory_cost.filter_size,
            }.dump();
            write!(memory_file, "{}\n", mem_result_str).unwrap();
        } */
        requests.clear();
    }
}


pub fn test_cole_plus_large_reorg(params: &ReOrgParams) -> Result<()> {
    let yscb_path = &params.ycsb_path;
    let file = OpenOptions::new().read(true).open(yscb_path).unwrap();
    YCSB.set(Mutex::new(BufReader::new(file))).map_err(|_e| anyhow!("Failed to set YCSB.")).unwrap();

    let result_path = &params.result_path;
    let base = format!("{}-{}-{}k-fan{}-ratio{}-mem{}-rewind{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.size_ratio, params.mem_size, params.rewind_blocks);
    let mut long_reorg_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-long-reorg.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    let rewind_blocks = params.rewind_blocks;
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let test_in_mem_roll = true;
    let test_disk_roll = true;
    let is_pruned = false;
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio, is_pruned, test_in_mem_roll, test_disk_roll);
    let kept_num_blocks = (params.mem_size / params.tx_in_block) as u32;
    let mut backend = ColePlusBackend::new(&configs, db_path, kept_num_blocks);

    build_base_for_reorg(params, &mut backend);
    println!("start rewind");
    let now = Utc::now();
    let start_ts: i64 = now.timestamp_nanos_opt().unwrap();
    backend.rewind_disk_states(rewind_blocks as u32);
    let now = Utc::now();
    let end_ts: i64 = now.timestamp_nanos_opt().unwrap();
    let elapse = end_ts - start_ts;
    let ts_result_str = object! {
        rewind: rewind_blocks,
        elapse: elapse,
    }.dump();
    write!(long_reorg_file, "{}\n", ts_result_str).unwrap();

    long_reorg_file.flush().unwrap();
    return Ok(());
}
