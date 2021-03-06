#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=viper128 --init_model=./model.pth --batch_size=32 --image_size=128 --motion_range=4 --num_frame=4 --test_epoch=20 --display --save_display --save_display_dir=./ 2>&1 | tee test.log

sh trim.sh
