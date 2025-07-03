extern crate locallib;
use locallib::read_test::{ReadParams, cole_index_backend_read_test, cole_plus_backend_read_test};
use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    let json_file_path = args.last().unwrap();
    let params = ReadParams::from_json_file(json_file_path);
    println!("{:?}", params);
    let index_name = &params.index_name;
    if index_name == "cole" {
        cole_index_backend_read_test(&params).unwrap();
    } else if index_name == "cole_plus_archive" {
        cole_plus_backend_read_test(&params).unwrap();
    }
}