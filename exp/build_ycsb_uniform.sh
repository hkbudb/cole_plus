~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./writeonly/writeonly-uniform > ./writeonly/writeonly-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./writeonly/writeonly-uniform > ./writeonly/writeonly-uniform-run.txt
cat ./writeonly/writeonly-uniform-load.txt ./writeonly/writeonly-uniform-run.txt > ./writeonly/writeonly-uniform-data.txt
rm ./writeonly/writeonly-uniform-load.txt ./writeonly/writeonly-uniform-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readwriteeven/readwriteeven-uniform > ./readwriteeven/readwriteeven-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readwriteeven/readwriteeven-uniform > ./readwriteeven/readwriteeven-uniform-run.txt
cat ./readwriteeven/readwriteeven-uniform-load.txt ./readwriteeven/readwriteeven-uniform-run.txt > ./readwriteeven/readwriteeven-uniform-data.txt
rm ./readwriteeven/readwriteeven-uniform-load.txt ./readwriteeven/readwriteeven-uniform-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readonly/readonly-uniform > ./readonly/readonly-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readonly/readonly-uniform > ./readonly/readonly-uniform-run.txt
cat ./readonly/readonly-uniform-load.txt ./readonly/readonly-uniform-run.txt > ./readonly/readonly-uniform-data.txt
rm ./readonly/readonly-uniform-load.txt ./readonly/readonly-uniform-run.txt