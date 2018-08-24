#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=robot128 --init_model=./model.pth --batch_size=32 --image_size=128 --num_frame=2 --net_depth=7 --min_diff_thresh=0.02 --test_epoch=20 --display --save_display --save_display_dir=./ 2>&1 | tee test.log

sh trim.sh