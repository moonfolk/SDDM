#!/bin/bash

DATA_PATH="/fastdisk/EJC_Data/ejc_time_group/"
PATH_TO_SAVE="/fastdisk/EJC_Output/sddm/"
NUM_CORES=40
DELTA=0.7
PROP_DISCARD=0.5
PROP_N=0.001

python run_cosacs.py \
--data_path ${DATA_PATH} \
--path_to_save ${PATH_TO_SAVE} \
--num_cores ${NUM_CORES} \
--delta ${DELTA} \
--prop_discard ${PROP_DISCARD} \
--prop_n ${PROP_N} \


