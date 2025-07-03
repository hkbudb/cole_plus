import json
import os
import subprocess
from glob import glob

indexes = ["mpt_archive", "mpt_prune", "cole_star", "cole_plus_async_archive", "cole_plus_async_prune"]
# scale = [10000, 100000, 1000000, 10000000]
default_tx_in_block = 100
size_ratio_default = 10
mht_fanout_default = 4
capacity_default = 1

def latency_json_gen(workload, distribution, contract_name, index_name, mem_size, scale, num_of_contract, tx_in_block, size_ratio, epsilon, mht_fanout):
    ycsb_path = "./%s/%s-%s-data.txt" % (workload, workload, distribution)
    ycsb_base_row_number = 20000
    db_path = "./%s/default_db" % workload
    if contract_name == "kvstore":
        result_path = "./%s/%s-%s" % (workload, workload, distribution)
    elif contract_name == "smallbank":
        result_path = "./%s/%s" % (workload, workload)
    params_dict = { "index_name": index_name, "contract_name": contract_name, "scale": scale, "ycsb_path": ycsb_path, "ycsb_base_row_number": ycsb_base_row_number, "num_of_contract": num_of_contract, "tx_in_block": tx_in_block, "db_path": db_path, "mem_size": mem_size, "size_ratio": size_ratio, "epsilon": epsilon, "mht_fanout": mht_fanout, "result_path":result_path}
    params_json = json.dumps(params_dict)
    if contract_name == "kvstore":
        params_file_path = "./%s/params-%s-%s-%s-%sk-fan%s-ratio%s-mem%s.json" % (workload, workload, distribution, index_name, scale//1000, mht_fanout, size_ratio, mem_size)
    elif contract_name == "smallbank":
        params_file_path = "./%s/params-%s-%s-%sk-fan%s-ratio%s-mem%s.json" % (workload, contract_name, index_name, scale//1000, mht_fanout, size_ratio, mem_size)
    with open(params_file_path, "w") as f:
        f.write(params_json)
    return params_file_path

def prov_json_gen(workload, distribution, index_name, mem_size, scale, tx_in_block, size_ratio, epsilon, mht_fanout, fix_window_size):
    ycsb_path = "./%s/prov-%s-data.txt" % (workload, distribution)
    ycsb_base_row_number = 100
    db_path = "./%s/default_db" % workload
    result_path = "./%s/%s-%s" % (workload, workload, distribution)
    params_dict = { "index_name": index_name, "scale": scale, "ycsb_path": ycsb_path, "ycsb_base_row_number": ycsb_base_row_number, "tx_in_block": tx_in_block, "db_path": db_path, "mem_size": mem_size, "size_ratio": size_ratio, "epsilon": epsilon, "mht_fanout": mht_fanout, "result_path":result_path, "fix_window_size": fix_window_size}
    params_json = json.dumps(params_dict)
    params_file_path = "./%s/params-%s-%s-%s-%sk-fan%s-ratio%s-mem%s-fixwind%s.json" % (workload, workload, distribution, index_name, scale//1000, mht_fanout, size_ratio, mem_size, fix_window_size)
    with open(params_file_path, "w") as f:
        f.write(params_json)
    return params_file_path

# def read_test_json_gen(workload, distribution, index_name, mem_size, scale, tx_in_block, size_ratio, epsilon, mht_fanout):
#     ycsb_path = "./%s/%s-%s-data.txt" % (workload, workload, distribution)
#     ycsb_base_row_number = 10000
#     db_path = "./%s/default_db" % workload
#     result_path = "./%s/%s-%s" % (workload, workload, distribution)
#     params_dict = { "index_name": index_name, "scale": scale, "ycsb_path": ycsb_path, "ycsb_base_row_number": ycsb_base_row_number, "tx_in_block": tx_in_block, "db_path": db_path, "mem_size": mem_size, "size_ratio": size_ratio, "epsilon": epsilon, "mht_fanout": mht_fanout, "result_path":result_path }
#     params_json = json.dumps(params_dict)
#     params_file_path = "./%s/params-%s-%s-%s-%sk-fan%s-ratio%s-mem%s.json" % (workload, workload, distribution, index_name, scale//1000, mht_fanout, size_ratio, mem_size)
#     with open(params_file_path, "w") as f:
#         f.write(params_json)
#     return params_file_path

def compute_general_size(path, others):
    file_exists = glob("%s/%s" % (path, others))
    if file_exists:
        proc = subprocess.Popen("du -b -s %s/%s" % (path, others), stdout=subprocess.PIPE, shell=True, encoding="utf8")
        (out, _) = proc.communicate()
        out = str(out)
        lines = out.split("\n")[:-1]
        size = 0
        for line in lines:
            s = line.split("\t")[0]
            size += int(s)
        return size
    else:
        return 0

def compute_size_breakdown(path):
    # compute mht size
    tree_meta_size = compute_general_size(path, "mht")
    # compute lv size
    lv_size = compute_general_size(path, "*.lv")
    # compute state size
    state_size = compute_general_size(path, "s_*")
    # compute model size
    model_size = compute_general_size(path, "m_*")
    # compute hash size
    mht_size = compute_general_size(path, "h_*")
    # compute filter size
    filter_size = compute_general_size(path, "f_*")
    # compute lower h size
    mht_size += compute_general_size(path, "lh_*")
    low_mht_size = compute_general_size(path, "lh_*")
    # compute upper h size
    mht_size += compute_general_size(path, "uh_*")
    up_mht_size = compute_general_size(path, "uh_*")
    # compute upper offset size
    mht_size += compute_general_size(path, "uo_*")
    offset_size = compute_general_size(path, "uo_*")
    # compute total size
    total_size = compute_general_size(path, "")
    return (tree_meta_size, lv_size, state_size, model_size, mht_size, low_mht_size, up_mht_size, offset_size, filter_size, total_size)

def write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default):
    result_path = "./%s/%s-%s" % (cur_workload, cur_workload, distribution)
    storage_file_name = "%s-%s-%sk-fan%s-ratio%s-mem%s-storage.json" % (result_path, cur_index, cur_scale//1000, mht_fanout_default, size_ratio_default, mem_size)
    db_path = "./%s/default_db" % cur_workload
    if "cole" in cur_index:
        (tree_meta_size, lv_size, state_size, model_size, mht_size, low_mht_size, up_mht_size, offset_size, filter_size, total_size) = compute_size_breakdown(db_path)
        storage_data = {
            "tree_meta": tree_meta_size,
            "level_meta": lv_size,
            "state_size": state_size,
            "mht_size": mht_size,
            "low_mht_size": low_mht_size,
            "up_mht_size": up_mht_size,
            "offset_size": offset_size, 
            "model_size": model_size,
            "filter_size": filter_size,
            "total_size": total_size
        }
    else:
        total_size = compute_general_size(db_path, "")
        storage_data = {
            "tree_meta": 0,
            "level_meta": 0,
            "state_size": 0,
            "mht_size": 0,
            "model_size": 0,
            "filter_size": 0,
            "total_size": total_size
        }
    storage_json = json.dumps(storage_data)
    with open(storage_file_name, "w") as f:
        f.write("%s\n" % storage_json)

def test_overall_kvstore(distribution, workloads):
    # workloads = ["writeonly", "readwriteeven", "readonly"]
    # workloads = ["writeonly"]
    # indexes = ["mpt_archive", "mpt_prune", "cole_star", "cole_plus_async_archive", "cole_plus_async_prune"]
    indexes = ["cole_star", "cole_plus_async_archive", "cole_plus_async_prune"]
    # indexes = ["letus"]
    scale = [60000000]
    for cur_workload in workloads:
        if not os.path.exists(cur_workload):
            os.mkdir(cur_workload)
        for cur_scale in scale:
            for cur_index in indexes:
                if cur_index == "cole_plus_async_archive" or cur_index == "cole_plus_async_prune" or cur_index == "cole_star":
                    mem_size = 10000
                else:
                    mem_size = 32
                print("%s %s %s %s" % (cur_workload, distribution, cur_index, cur_scale))
                params_file_path = latency_json_gen(workload=cur_workload, distribution=distribution, contract_name="kvstore", index_name=cur_index, mem_size=mem_size, scale=cur_scale, num_of_contract=1, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=mht_fanout_default)
                os.system("cargo run --release --bin latency %s" % (params_file_path))
                # compute storage
                # write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default)
                os.system("rm -rf ./%s/default_db" % cur_workload)

""" def test_overall_smallbank():
    workloads = ["smallbank"]
    indexes = ["mpt_archive", "mpt_prune", "cole", "cole_plus_archive", "cole_plus_prune"]
    scale = [10000, 100000, 1000000, 10000000]
    for cur_workload in workloads:
        if not os.path.exists(cur_workload):
            os.mkdir(cur_workload)
        for cur_scale in scale:
            for cur_index in indexes:
                if cur_index == "cole_plus_archive" or cur_index == "cole_plus_prune":
                    mem_size = int(450000 * capacity_default)
                elif cur_index == "cole":
                    mem_size = 450000
                else:
                    mem_size = 64
                print("%s %s %s" % (cur_workload, cur_index, cur_scale))
                params_file_path = latency_json_gen(workload=cur_workload, contract_name="smallbank", index_name=cur_index, mem_size=mem_size, scale=cur_scale, num_of_contract=1, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=mht_fanout_default)
                os.system("cargo run --release --bin latency %s" % (params_file_path))
                # compute storage
                write_storage_json(cur_workload, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default)
                os.system("rm -rf ./%s/default_db" % cur_workload) """

def test_prov(distribution):
    cur_workload = "prov"
    if not os.path.exists(cur_workload):
        os.mkdir(cur_workload)
    cur_scale = 20000000
    test_index = ["mpt_archive"]
    for cur_index in test_index:
        print("test prov, index: %s" % (cur_index))
        if cur_index == "cole_plus_async_archive" or cur_index == "cole_plus_async_prune" or cur_index == "cole_star":
            mem_size = 10000
        else:
            mem_size = 32
        params_file_path = prov_json_gen(workload=cur_workload, distribution=distribution, index_name=cur_index, mem_size=mem_size, scale=cur_scale, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=mht_fanout_default, fix_window_size=False)
        os.system("cargo run --release --bin prov %s" % (params_file_path))
        # compute storage
        # write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default)
        os.system("rm -rf ./%s/default_db" % cur_workload)

""" def read_test(distribution):
    indexes = ["mpt_archive"]
    scale = [1000000]
    cur_workload = "writeonly"
    if not os.path.exists(cur_workload):
        os.mkdir(cur_workload)
    for cur_scale in scale:
        for cur_index in indexes:
            if cur_index == "cole_plus_archive" or cur_index == "cole_plus_prune":
                mem_size = int(450000 * capacity_default)
            elif cur_index == "cole":
                mem_size = 450000
            else:
                mem_size = 64
            print("%s %s %s" % (cur_workload, cur_index, cur_scale))
            params_file_path = read_test_json_gen(workload=cur_workload, distribution=distribution, index_name=cur_index, mem_size=mem_size, scale=cur_scale, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=mht_fanout_default)
            os.system("cargo run --release --bin read %s" % (params_file_path))
            # compute storage
            write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default)
            os.system("rm -rf ./%s/default_db" % cur_workload)
 """

def test_mht_fanout(distribution):
    cur_workload = "fanout"
    if not os.path.exists(cur_workload):
        os.mkdir(cur_workload)
    test_index = ["cole_star", "cole_plus_async_archive"]
    test_fanout = [2, 4, 8, 16, 32]
    cur_scale = 20000000
    for cur_index in test_index:
        if cur_index == "cole_plus_async_archive" or cur_index == "cole_plus_async_prune" or cur_index == "cole_star":
            mem_size = 10000
        for fanout in test_fanout:
            print("test mht fanout, index: %s, fanout: %s" % (cur_index, fanout))
            params_file_path = prov_json_gen(workload=cur_workload, distribution=distribution, index_name=cur_index, mem_size=mem_size, scale=cur_scale, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=fanout, fix_window_size=False)
            os.system("cargo run --release --bin prov %s" % (params_file_path))
            # compute storage
            # write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, fanout, size_ratio_default)
            os.system("rm -rf ./%s/default_db" % cur_workload)

def test_size_ratio(distribution):
    cur_workload = "size_ratio"
    if not os.path.exists(cur_workload):
        os.mkdir(cur_workload)
    test_index = ["cole_star", "cole_plus_async_archive"]
    test_size_ratio = [2, 4, 6, 8, 10, 12]
    cur_scale = 60000000
    for cur_index in test_index:
        if cur_index == "cole_plus_async_archive" or cur_index == "cole_plus_async_prune" or cur_index == "cole_star":
            mem_size = 10000
        for size_ratio in test_size_ratio:
            print("test size ratio, index: %s, size_ratio: %s" % (cur_index, size_ratio))
            params_file_path = latency_json_gen(workload=cur_workload, distribution=distribution, contract_name="kvstore", index_name=cur_index, mem_size=mem_size, scale=cur_scale, num_of_contract=1, tx_in_block=default_tx_in_block, size_ratio=size_ratio, epsilon=23, mht_fanout=mht_fanout_default)
            os.system("cargo run --release --bin latency %s" % (params_file_path))
            # compute storage
            # write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio)
            os.system("rm -rf ./%s/default_db" % cur_workload)

""" def test_capacity(distribution):
    cur_workload = "capacity"
    if not os.path.exists(cur_workload):
        os.mkdir(cur_workload)
    test_index = ["cole"]
    cur_scale = 30000000
    test_capacity = [1]
    for cur_index in test_index:
        for capacity in test_capacity:
            if cur_index == "cole_plus_archive":
                mem_size = int(450000 * capacity_default)
            elif cur_index == "cole":
                mem_size = 450000
            print("test capacity, index: %s, capacity: %s" % (cur_index, capacity))
            params_file_path = latency_json_gen(workload=cur_workload, distribution=distribution, contract_name="kvstore", index_name=cur_index, mem_size=mem_size, scale=cur_scale, num_of_contract=1, tx_in_block=default_tx_in_block, size_ratio=size_ratio_default, epsilon=23, mht_fanout=mht_fanout_default)
            os.system("cargo run --release --bin latency %s" % (params_file_path))
            # compute storage
            write_storage_json(cur_workload, distribution, cur_index, mem_size, cur_scale, mht_fanout_default, size_ratio_default)
            os.system("rm -rf ./%s/default_db" % cur_workload) """

if __name__ == "__main__":
    # test_mht_fanout("uniform")
    # test_size_ratio("uniform")
    # test_prov("uniform")
    # test_overall_kvstore("uniform", ["writeonly", "readwriteeven", "readonly"])
    # test_overall_kvstore("zipfian", ["readwriteeven", "readonly"])
    test_overall_kvstore("uniform", ["writeonly"])
    pass