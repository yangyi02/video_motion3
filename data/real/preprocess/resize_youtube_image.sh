#!/usr/bin/env bash

INPUT_DIR=/home/yi/Downloads/youtube
OUTPUT_DIR=/home/yi/Downloads/youtube-64
FILE_LIST=../youtube/youtube_file.list
SIZE=64

python create_file_list.py --input_dir=${INPUT_DIR} --output_file=${FILE_LIST}
# python resize_image.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}
python resize_image_mp.py --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} --size=${SIZE} --file_list=${FILE_LIST}

