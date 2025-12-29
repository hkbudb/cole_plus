~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./writeheavy/writeheavy-uniform > ./writeheavy/writeheavy-uniform-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./writeheavy/writeheavy-uniform > ./writeheavy/writeheavy-uniform-run.txt
cat ./writeheavy/writeheavy-uniform-load.txt ./writeheavy/writeheavy-uniform-run.txt > ./writeheavy/writeheavy-uniform-data.txt
rm ./writeheavy/writeheavy-uniform-load.txt ./writeheavy/writeheavy-uniform-run.txt