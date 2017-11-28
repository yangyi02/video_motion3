#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=robot128c --init_model=./model.pth --batch_size=32 --image_size=128 --motion_range=2 --num_frame=4 --min_diff_thresh=0.02 --max_diff_thresh=0.2 --diff_div_thresh=1.1 --test_epoch=20 --display --save_display --save_display_dir=./ 2>&1 | tee test.log

sh trim.sh
