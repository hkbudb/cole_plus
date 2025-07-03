~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./prov/prov-data-uniform > ./prov/prov-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./prov/prov-data-uniform > ./prov/prov-uniform-run.txt
cat ./prov/prov-uniform-load.txt ./prov/prov-uniform-run.txt > ./prov/prov-uniform-data.txt
rm ./prov/prov-uniform-load.txt ./prov/prov-uniform-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./prov/prov-data-zipfian > ./prov/prov-zipfian-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./prov/prov-data-zipfian > ./prov/prov-zipfian-run.txt
cat ./prov/prov-zipfian-load.txt ./prov/prov-zipfian-run.txt > ./prov/prov-zipfian-data.txt
rm ./prov/prov-zipfian-load.txt ./prov/prov-zipfian-run.txt