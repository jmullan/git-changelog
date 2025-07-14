#!/bin/bash

mkdir -p test_data
cd test_data
curl 'https://zenodo.org/records/8189044/files/test.jsonl.gz?download=1' \
    -o commit-chronicle.jsonl.gz
unzip commit-chronicle.jsonl.gz
cat commit-chronicle.jsonl | jq -c "del(.mods)" > commit-chronicle.nodiff.jsonl
rm -f commit-chronicle.jsonl.gz commit-chronicle.jsonl

# the following dataset appears to be modified from the original messages

#curl 'https://zenodo.org/records/5025758/files/filtered_data.tar.gz?download=1' \
#    -o mcmd_filtered_data.tar.gz
#tar xvzf mcmd_filtered_data.tar.gz --wildcards '*/*/valid.msg.txt'
#rm mcmd_filtered_data.tar.gz
#cat */*/valid.msg.txt > mcmd_msg.txt
#rm */*/valid.msg.txt
#find . -empty -delete
