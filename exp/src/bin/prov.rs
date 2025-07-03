extern crate locallib;
use locallib::prov_test::{cole_index_backend_prov_query, cole_plus_async_backend_prov_query, cole_plus_backend_prov_query, cole_star_backend_prov_query, mpt_backend_prov_query, ProvParams};
use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    let json_file_path = args.last().unwrap();
    let params = ProvParams::from_json_file(json_file_path);
    println!("{:?}", params);
    let index_name = &params.index_name;
    if index_name == "mpt_archive" {
        mpt_backend_prov_query(&params).unwrap();
    } else if index_name == "cole" {
        cole_index_backend_prov_query(&params).unwrap();
    } else if index_name == "cole_plus_archive" {
        cole_plus_backend_prov_query(&params).unwrap();
    } else if index_name == "cole_star" {
        cole_star_backend_prov_query(&params).unwrap();
    } else if index_name == "cole_plus_async_archive" {
        cole_plus_async_backend_prov_query(&params).unwrap();
    }
}