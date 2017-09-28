#!/usr/bin/env bash

INPUT_DIR=/media/yi/DATA/data-orig/mpii_human_pose_v1_sequences_3
OUTPUT_DIR=/home/yi/Downloads/mpii-3-64
FILE_LIST=../mpii-3/mpii_file.list
SIZE=64

python create_file_list.py --input_dir=${INPUT_DIR} --output_file=${FILE_LIST}
# python resize_image.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}
python resize_image_mp.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}

