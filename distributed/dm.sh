#!/bin/bash

DATA_PATH="/fastdisk/EJC_Data/ejc_group/"
PATH_TO_SAVE="/fastdisk/EJC_Output/dm/"
GLOBAL_TOPICS_PATH="/fastdisk/EJC_Output/dm/"
META_PATH="/fastdisk/EJC_Data/ejc_time_group_meta_data/"
TAU1=2
GAMMA=1
NUM_CORES=40
DELTA=0.7
PROP_DISCARD=0.5
PROP_N=0.001

/usr/bin/python main_distributed.py \
--data_path ${DATA_PATH} \
--path_to_save ${PATH_TO_SAVE} \
--global_topics_path ${GLOBAL_TOPICS_PATH} \
--meta_path ${META_PATH} \
--tau1 ${TAU1} \
--gamma ${GAMMA} \
--num_cores ${NUM_CORES} \
--delta ${DELTA} \
--prop_discard ${PROP_DISCARD} \
--prop_n ${PROP_N} \

