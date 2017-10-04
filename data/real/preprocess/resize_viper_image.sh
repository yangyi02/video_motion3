#!/usr/bin/env bash

# INPUT_DIR=/home/yi/Downloads/VIPER/train/img
INPUT_DIR=/home/yi/Downloads/VIPER/val/img
# OUTPUT_DIR=/home/yi/code/video_motion_data/viper128-train
OUTPUT_DIR=/home/yi/code/video_motion_data/viper128-val
# FILE_LIST=./viper_file_train.list
FILE_LIST=./viper_file_val.list
SIZE=128

python create_file_list.py --input_dir=${INPUT_DIR} --output_file=${FILE_LIST}
# python resize_image.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}
python resize_image_mp.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}
