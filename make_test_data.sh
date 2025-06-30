#!/bin/bash

mkdir -p test_data
curl 'https://zenodo.org/records/8189044/files/test.jsonl.gz?download=1' \
    -o test_data/commit-chronicle.jsonl.gz
curl 'https://zenodo.org/records/5025758/files/filtered_data.tar.gz?download=1' \
    -o test_data/mcmd_filtered_data.tar.gz
