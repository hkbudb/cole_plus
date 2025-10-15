# COLE<sup>+</sup>: Towards Practical Column-based Learned Storage for Blockchain Systems

## Components

- `cole-plus-async` and `cole-star` are COLE<sup>+</sup> and COLE with asynchronous merge functionality
- `patricia-trie` is the implementation of MPT
- `exp` is the evaluation backend of all systems including the throughput and the provenance queries
- `cole-plus-ablation-siri` is an ablation version of `cole-plus` where the CDC method is replaced with Structurally Invariant and Reusable Indexes (SIRI)
- `cole-ablation-layout` is an ablation version of `cole-star` that modifies the state file layout by separating the latest and historical states into different files

##  Install Dependencies

- Install [Rust](https://rustup.rs).
```
sudo apt install -y curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
- Run `sudo apt update` and `sudo apt -y install git make clang pkg-config libssl-dev libsqlite3-dev llvm m4 build-essential`


## Download Repository to $HOME directory
Download the repository and rename it with `cole-plus`
## Build
* Build the latency testing binary and provenance testing binary
```
cd ~/cole-plus
cargo build --release --bin latency
cargo build --release --bin prov
```
* If the program is built successfully, you should find two executable programs `latency` and `prov` in the directory `~/cole-plus/target/release/`

## Prepare YCSB Dataset
* Download the latest release of YCSB to the $HOME directory:
```
cd ~
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
```
* Install Java
```
sudo apt -y install default-jdk
sudo apt -y install default-jre
```
* Use scripts `build_ycsb_uniform.sh` and `build_ycsb_zipfian.sh` to generate `readonly`, `writeonly`, and `readwriteeven` datasets
```
cd ~/cole-plus/exp
./build_ycsb.sh
```

* After the build process finishes, three `txt` files will be generate:
    * `cole-plus/exp/readonly/readonly-zipfian-data.txt`
    * `cole-plus/exp/readonly/readonly-uniform-data.txt`
    * `cole-plus/exp/writeonly/writeonly-zipfian-data.txt`
    * `cole-plus/exp/writeonly/writeonly-uniform-data.txt`
    * `cole-plus/exp/readwriteeven/readwriteeven-zipfian-data.txt`
    * `cole-plus/exp/readwriteeven/readwriteeven-uniform-data.txt`

* Next, prepare the dataset for provenance queries:
```
cd ~/cole-plus/exp/
./build_prov_ycsb.sh
```

* After the build process finishes, a file named `cole-plus/exp/prov/prov-data.txt` will be generated.

## Run Script
```
cd ~/cole-plus/exp/
python3 run.py
```

* Use functions like `test_overall_kvstore()` and `test_prov()` in `cole-plus/exp/run.py` to evaluate the workload of `KVStore` and provenance query performance.
* You may select different scales `scale = [3000000, 6000000, 30000000, 60000000]` or different indexes `indexes = ["cole_star", "cole_plus_async_archive", "cole_plus_async_prune", "cole_plus_ablation_siri", "cole_ablation_layout"]`

## Check the Result

The result `json` files can be found in each workload directory (e.g., writeonly, prov)
* `*-storage.json` stores the storage information
* `*-ts.json` stores the block timestamp information including start timestamp, end timestamp, and block latency, which can be used to compute the system throughput and latency