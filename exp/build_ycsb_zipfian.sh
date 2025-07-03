~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./writeonly/writeonly-zipfian > ./writeonly/writeonly-zipfian-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./writeonly/writeonly-zipfian > ./writeonly/writeonly-zipfian-run.txt
cat ./writeonly/writeonly-zipfian-load.txt ./writeonly/writeonly-zipfian-run.txt > ./writeonly/writeonly-zipfian-data.txt
rm ./writeonly/writeonly-zipfian-load.txt ./writeonly/writeonly-zipfian-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readwriteeven/readwriteeven-zipfian > ./readwriteeven/readwriteeven-zipfian-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readwriteeven/readwriteeven-zipfian > ./readwriteeven/readwriteeven-zipfian-run.txt
cat ./readwriteeven/readwriteeven-zipfian-load.txt ./readwriteeven/readwriteeven-zipfian-run.txt > ./readwriteeven/readwriteeven-zipfian-data.txt
rm ./readwriteeven/readwriteeven-zipfian-load.txt ./readwriteeven/readwriteeven-zipfian-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readonly/readonly-zipfian > ./readonly/readonly-zipfian-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readonly/readonly-zipfian > ./readonly/readonly-zipfian-run.txt
cat ./readonly/readonly-zipfian-load.txt ./readonly/readonly-zipfian-run.txt > ./readonly/readonly-zipfian-data.txt
rm ./readonly/readonly-zipfian-load.txt ./readonly/readonly-zipfian-run.txt