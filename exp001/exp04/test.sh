#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=robot64 --init_model=./model.pth --batch_size=64 --image_size=64 --motion_range=2 --num_frame=3 --test_epoch=20 --display --save_display --save_display_dir=./ 2>&1 | tee test.log

sh trim.sh
