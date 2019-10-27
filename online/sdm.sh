#!/bin/bash

DATA_PATH="/fastdisk/EJC_Data/ejc_time/"
PATH_TO_SAVE="/fastdisk/EJC_Output/sdm/"
GLOBAL_TOPICS_PATH="/fastdisk/EJC_Output/sdm/"
META_PATH="/fastdisk/EJC_Data/ejc_time_group_meta_data/"
TAU0=2
TAU1=1
GAMMA=1
NUM_CORES=40
DELTA=0.7
PROP_DISCARD=0.5
PROP_N=0.001

/usr/bin/python main_online.py \
--data_path ${DATA_PATH} \
--path_to_save ${PATH_TO_SAVE} \
--global_topics_path ${GLOBAL_TOPICS_PATH} \
--meta_path ${META_PATH} \
--tau0 ${TAU0} \
--tau1 ${TAU1} \
--gamma ${GAMMA} \
--num_cores ${NUM_CORES} \
--delta ${DELTA} \
--prop_discard ${PROP_DISCARD} \
--prop_n ${PROP_N} \

