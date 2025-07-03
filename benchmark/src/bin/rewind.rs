use cole_plus::in_memory_postree::InMemoryPOSTree;
use pattern_oridented_split_tree::{check_pointers, print_tree};
use primitive_types::{H160, H256};
use rand::{rngs::StdRng, SeedableRng};
use utils::types::{AddrKey, Address, CompoundKey, StateKey, StateValue};
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
    for reverse_n in reverses {
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

/*    let reverse_states = 64 * num_of_version;
   let mut tree = InMemoryPOSTree::new(fanout, 92);
    for (i, (key, value)) in state_vec.iter().enumerate() {
        pattern_oridented_split_tree::insert(&mut tree, *key, *value);
    }

    for (key, _) in state_vec.iter().rev().take(reverse_states as usize) {
        pattern_oridented_split_tree::remove(&mut tree, *key);
    } */
}