#!/bin/bash

source ../../set_path.sh

python ../../data/synthetic/mnist_bidata.py --batch_size=1 --image_size=32 --image_channel=1 --motion_range=1 --num_frame=1 --num_objects=1 --save_display --save_display_dir=./

python ../../data/synthetic/mnist_bidata.py --batch_size=2 --image_size=32 --motion_range=2 --num_frame=3 --num_objects=2 --save_display --save_display_dir=./

python ../../data/synthetic/mnist_bidata.py --batch_size=3 --image_size=32 --motion_range=3 --num_frame=5 --num_objects=3 --save_display --save_display_dir=./

python ../../data/synthetic/mnist_bidata.py --batch_size=4 --image_size=32 --motion_range=4 --num_frame=7 --num_objects=4 --bg_move --save_display --save_display_dir=./

sh trim.sh
