~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readheavy/readheavy-uniform > ./readheavy/readheavy-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readheavy/readheavy-uniform > ./readheavy/readheavy-uniform-run.txt
cat ./readheavy/readheavy-uniform-load.txt ./readheavy/readheavy-uniform-run.txt > ./readheavy/readheavy-uniform-data.txt
rm ./readheavy/readheavy-uniform-load.txt ./readheavy/readheavy-uniform-run.txt