#!/bin/bash

source ../../set_path.sh

python ../demo.py --test_gt --data=box --batch_size=64 --image_size=32 --motion_range=2 --num_frame=3 --rand_noise=0.4 --test_epoch=20 --display --save_display --save_display_dir=./ 2>&1 | tee test_gt.log

sh trim.sh
