use cole_plus::in_memory_postree::InMemoryPOSTree;
use pattern_oridented_split_tree::{get_tree_height, print_tree, reconstruct_range_proof, traits::POSTreeNodeIO};
use primitive_types::{H160, H256};
use rand::{rngs::StdRng, SeedableRng};
use utils::types::{AddrKey, Address, CompoundKey, StateKey, StateValue};

fn main() {
    let num_of_contract = 1;
    let num_of_addr = 500;
    let num_of_version = 100;
    // let n = num_of_contract * num_of_addr * num_of_version;
    let mut rng = StdRng::seed_from_u64(1);
    let fanout = 4;

    let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
    let mut addr_key_vec = Vec::<AddrKey>::new();
    for _ in 1..=num_of_contract {
        for _ in 1..=num_of_addr {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let addr_key = AddrKey::new(Address(acc_addr), StateKey(state_addr));
            addr_key_vec.push(addr_key);
        }
    }
    for k in 1..=num_of_version {
        for (i, addr_key) in addr_key_vec.iter().enumerate() {
            state_vec.push((CompoundKey::new(*addr_key, k * 2), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
        }
    }

    let mut tree = InMemoryPOSTree::new(fanout, 100);
    let start = std::time::Instant::now();
    for (i, (key, value)) in state_vec.iter().enumerate() {
        pattern_oridented_split_tree::insert(&mut tree, *key, *value);
    }
    let elapse = start.elapsed().as_nanos();
    // print_tree(&tree);
    println!("insert time: {:?}", elapse);
    let root_h = tree.get_root_hash();
    println!("root: {:?}", root_h);
    let height = get_tree_height(&tree);
    println!("height: {}", height);
    println!("------------------------");
    for (i, (key, value)) in state_vec.iter().enumerate() {
        // println!("i: {}", i);
        let r = pattern_oridented_split_tree::search_without_proof(&tree, *key).unwrap();
        let read_value = r.1;
        let read_key = r.0;
        if read_key != *key || read_value != *value {
            println!("read key: {:?}, key: {:?}, read_value: {:?}, value: {:?}", read_key, key, read_value, value);
        }
    }
    // // search latest
    // for (i, addr) in addr_key_vec.iter().enumerate() {
    //     let upper_key = CompoundKey::new(*addr, u32::MAX);
    //     let r = pattern_oridented_split_tree::search_with_upper_key(&tree, upper_key).unwrap();
    //     let read_value = r.1;
    //     let read_key = r.0;
    //     let true_value = StateValue(H256::from_low_u64_be( (num_of_version as u64 + i as u64) * 2));
    //     let true_key = CompoundKey::new(*addr, num_of_version * 2);
    //     if read_key != true_key || read_value != true_value {
    //         println!("read key: {:?}, true key: {:?}, read value: {:?}, true value: {:?}", read_key, true_key, read_value, true_value);
    //     }
    // }
}