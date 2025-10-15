use cole_plus::in_memory_postree::InMemoryPOSTree;
use siri_tree::in_memory_siri::InMemorySIRI;
use pattern_oridented_split_tree::{check_pointers, print_tree};
use primitive_types::{H160, H256};
use rand::{rngs::StdRng, SeedableRng};
use utils::types::{AddrKey, Address, CompoundKey, StateKey, StateValue};
use cole_plus::run::LevelRun;
fn main() {
    let num_of_contract = 1;
    let num_of_addr = 50;
    let num_of_version = 100;
    // let n = num_of_contract * num_of_addr * num_of_version;
    let mut rng = StdRng::seed_from_u64(1);
    let fanout = 2;

    let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
    let mut addr_key_vec = Vec::<AddrKey>::new();
    for _ in 1..=num_of_contract {
        let acc_addr = H160::random_using(&mut rng);
        for _ in 1..=num_of_addr {
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

    let mut reverse_n = 1;
    let mut reverses = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
    for reverse_n in reverses.clone() {
        let reverse_states = reverse_n * num_of_version;
        {
            // build tree
            let mut tree = InMemoryPOSTree::new(fanout, 92);
            for (i, (key, value)) in state_vec.iter().enumerate() {
            pattern_oridented_split_tree::insert(&mut tree, *key, *value);
            }
            // remove
            if reverse_states < 50 * num_of_version {
                let start = std::time::Instant::now();
                for (key, _) in state_vec.iter().rev().take(reverse_states as usize) {
                    pattern_oridented_split_tree::remove(&mut tree, *key);
                }
                let elapse = start.elapsed().as_nanos();
                println!("{}\t{:?}", reverse_n, elapse);
            } else {
                let start = std::time::Instant::now();
                tree.clear();
                let clear_elapse = start.elapsed().as_nanos();
                for (i, (key, value)) in state_vec.iter().enumerate() {
                    pattern_oridented_split_tree::insert(&mut tree, *key, *value);
                }
                let real_reverse = reverse_states - 50 * num_of_version;
                let start = std::time::Instant::now();
                for (key, _) in state_vec.iter().rev().take(real_reverse as usize) {
                    pattern_oridented_split_tree::remove(&mut tree, *key);
                }
                let elapse = start.elapsed().as_nanos();
                println!("{}\t{:?}", reverse_n, elapse + clear_elapse);
            }
        }
    }
    println!("siri performance:");
    for reverse_n in reverses {
        let reverse_states = reverse_n * num_of_version;
        {
            // build tree
            let mut tree = InMemorySIRI::new(fanout, 92);
            for (i, (key, value)) in state_vec.iter().enumerate() {
            siri_tree::insert(&mut tree, *key, *value);
            }
            // remove
            if reverse_states < 50 * num_of_version {
                let start = std::time::Instant::now();
                for (key, _) in state_vec.iter().rev().take(reverse_states as usize) {
                    siri_tree::remove(&mut tree, *key);
                }
                let elapse = start.elapsed().as_nanos();
                println!("{}\t{:?}", reverse_n, elapse);
            } else {
                let start = std::time::Instant::now();
                tree.clear();
                let clear_elapse = start.elapsed().as_nanos();
                for (i, (key, value)) in state_vec.iter().enumerate() {
                    siri_tree::insert(&mut tree, *key, *value);
                }
                let real_reverse = reverse_states - 50 * num_of_version;
                let start = std::time::Instant::now();
                for (key, _) in state_vec.iter().rev().take(real_reverse as usize) {
                    siri_tree::remove(&mut tree, *key);
                }
                let elapse = start.elapsed().as_nanos();
                println!("{}\t{:?}", reverse_n, elapse + clear_elapse);
            }
        }
    }



    // build tree, 10 blocks/1000 states, 20 blocks, 30 blocks, 40 blocks, 50 blocks, 60 blocks, 70 blocks, 80 blocks, 90 blocks, 100 blocks
    let versions = vec![20, 40, 60, 80, 100, 120, 140, 160, 180, 200];
    for v in versions {
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        for k in 1..=v {
            for (i, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k * 2), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
            }
        }
        let mut tree = InMemoryPOSTree::new(fanout, 92);
        let start = std::time::Instant::now();
        for (i, (key, value)) in state_vec.iter().enumerate() {
            pattern_oridented_split_tree::insert(&mut tree, *key, *value);
        }
        let elapse = start.elapsed().as_nanos();
        println!("v: {}, insert: {:?}", v, elapse);
    }

    println!("siri performance");
    let versions = vec![20, 40, 60, 80, 100, 120, 140, 160, 180, 200];
    for v in versions {
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        for k in 1..=v {
            for (i, addr_key) in addr_key_vec.iter().enumerate() {
                state_vec.push((CompoundKey::new(*addr_key, k * 2), StateValue(H256::from_low_u64_be( (k as u64 + i as u64) * 2))));
            }
        }
        let mut tree = InMemorySIRI::new(fanout, 92);
        let start = std::time::Instant::now();
        for (i, (key, value)) in state_vec.iter().enumerate() {
            siri_tree::insert(&mut tree, *key, *value);
        }
        let elapse = start.elapsed().as_nanos();
        println!("v: {}, insert: {:?}", v, elapse);
    }

/*    let reverse_states = 64 * num_of_version;
   let mut tree = InMemoryPOSTree::new(fanout, 92);
    for (i, (key, value)) in state_vec.iter().enumerate() {
        pattern_oridented_split_tree::insert(&mut tree, *key, *value);
    }

    for (key, _) in state_vec.iter().rev().take(reverse_states as usize) {
        pattern_oridented_split_tree::remove(&mut tree, *key);
    } */
}