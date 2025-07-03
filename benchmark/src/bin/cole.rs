use cole_index::Cole;
use primitive_types::{H160, H256};
use rand::{rngs::StdRng, SeedableRng};
use utils::{config::Configs, types::{AddrKey, Address, CompoundKey, StateKey, StateValue}};
fn main() {
    let num_of_contract = 100;
    let num_of_addr = 1000;
    let num_of_version = 10;
    let n = num_of_contract * num_of_addr * num_of_version;
    let mut rng = StdRng::seed_from_u64(1);
    let fanout = 5;
    let dir_name = "cole_storage";
    if std::path::Path::new(dir_name).exists() {
        std::fs::remove_dir_all(dir_name).unwrap_or_default();
    }
    std::fs::create_dir(dir_name).unwrap_or_default();
    let base_state_num = 45000;
    let size_ratio = 10;
    let configs = Configs::new(fanout, 0, dir_name.to_string(), base_state_num, size_ratio, false);
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

    let mut cole = Cole::new(&configs);
    let start = std::time::Instant::now();
    for state in &state_vec {
        cole.insert((state.0, state.1));
    }
    let elapse = start.elapsed().as_nanos();
    println!("average insert: {:?}", elapse / n as u128);
}