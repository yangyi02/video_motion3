#!/bin/bash

source ../../set_path.sh

CUDA_VISIBLE_DEVICES=1 python ../demo.py --train --data=robot128c --batch_size=32 --image_size=128 --motion_range=2 --num_frame=4 --min_diff_thresh=0.02 --max_diff_thresh=0.2 --diff_div_thresh=1.1 --train_epoch=2000 --test_interval=200 --test_epoch=20 --learning_rate=0.001 2>&1 | tee train.log
