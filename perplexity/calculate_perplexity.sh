#!/bin/bash

TEST_PATH="/fastdisk/EJC_Data/ejc_test/"
TOPICS_PATH="/fastdisk/EJC_Output/dm/global_topics/"
ALGORITHM="dm"

python ./get_perplexity.py \
--test_path ${TEST_PATH} \
--topics_path ${TOPICS_PATH} \
--algorithm ${ALGORITHM} 
