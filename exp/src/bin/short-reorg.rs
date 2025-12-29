extern crate locallib;
use locallib::short_reorg_test::{ReOrgParams, test_cole_plus_short_reorg};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    let json_file_path = args.last().unwrap();
    let params = ReOrgParams::from_json_file(json_file_path);
    println!("{:?}", params);
    test_cole_plus_short_reorg(&params).unwrap();
}