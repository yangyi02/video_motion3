#!/bin/bash

source ../../../set_path.sh

python ../../demo.py --test --data=mpii64_sample --init_model=../model.pth --batch_size=64 --image_size=64 --motion_range=2 --num_frame=3 --test_epoch=1 --display_all --save_display --save_display_dir=./

sh trim.sh
