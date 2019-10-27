#!/bin/bash

DATA_PATH="/fastdisk/EJC_Data/ejc_time_group/"
PATH_TO_SAVE="/fastdisk/EJC_Output/sddm/"
META_PATH="/fastdisk/EJC_Data/ejc_time_group_meta_data/"
COSAC_PATH="/fastdisk/EJC_Output/sddm/"
TAU0=4
TAU1=2
GAMMA=2
IT=5

python run_aggregator.py \
--data_path ${DATA_PATH} \
--path_to_save ${PATH_TO_SAVE} \
--meta_path ${META_PATH} \
--cosac_path ${COSAC_PATH} \
--tau0 ${TAU0} \
--tau1 ${TAU1} \
--gamma ${GAMMA} \
--it ${IT}
