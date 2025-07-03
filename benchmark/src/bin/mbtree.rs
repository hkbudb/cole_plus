use cole_plus::in_memory_mbtree::InMemoryMBTree;
use merkle_btree_storage::get_tree_height;
use primitive_types::{H160, H256};
use rand::{rngs::StdRng, SeedableRng};
use utils::types::{AddrKey, Address, CompoundKey, StateKey, StateValue};

fn main() {
    let num_of_contract = 1;
        let num_of_addr = 500;
        let num_of_version = 100;
        // let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);
        let fanout = 8;

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

        let mut tree = InMemoryMBTree::new(fanout);
        let start = std::time::Instant::now();
        for (_, (key, value)) in state_vec.iter().enumerate() {
            // println!("i: {}", i);
            merkle_btree_storage::insert(&mut tree, *key, *value);
        }
        let elapse = start.elapsed().as_nanos();
        println!("insert time: {:?}", elapse);
        let height = get_tree_height(&tree);
        println!("height: {}", height);
}