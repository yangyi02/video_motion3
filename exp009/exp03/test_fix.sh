#!/bin/bash

source ../../set_path.sh

python ../demo.py --test --data=mnist --init_model=./model.pth --batch_size=64 --image_size=32 --motion_range=4 --num_frame=4 --bg_move --test_epoch=20 --fixed_data --display --save_display --save_display_dir=./

sh trim.sh
