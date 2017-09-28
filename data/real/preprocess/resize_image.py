import io
import os
import cv2
import argparse
from time import time
import logging
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--file_list', default='')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    size = args.size
    file_list = args.file_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_dirs = os.listdir(os.path.join(input_dir))
    for image_dir in image_dirs:
        if not os.path.exists(os.path.join(output_dir, image_dir)):
            os.makedirs(os.path.join(output_dir, image_dir))
        sub_dirs = os.listdir(os.path.join(input_dir, image_dir))
        for sub_dir in sub_dirs:
            if not os.path.exists(os.path.join(output_dir, image_dir, sub_dir)):
                os.makedirs(os.path.join(output_dir, image_dir, sub_dir))

    image_files = io.open(file_list).readlines()
    num_files = len(image_files)

    num_images = 0
    total_time = 0
    for image_name in image_files:
        num_images += 1
        start_time = time()
        image_name = image_name.strip()
        im = cv2.imread(os.path.join(input_dir, image_name))
        if im.shape[0] < im.shape[1]:
            new_height = size
            new_width = int(round(float(size) / im.shape[0] * im.shape[1]))
        else:
            new_width = size
            new_height = int(round(float(size) / im.shape[1] * im.shape[0]))
        # It is best to use cv2.INTER_AREA when shrinking an image
        # I didn't see significant difference by using a Gaussian Blur before resize
        # im = cv2.GaussianBlur(im, (5, 5), 0)
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_image_name = os.path.join(output_dir, image_name)
        cv2.imwrite(new_image_name, im)
        end_time = time()
        total_time += end_time - start_time
        avg_time = total_time / num_images
        approximate_finish_time = (num_files - num_images) * avg_time / 60 / 60
        logging.info('resize image save to %s, takes %.2f second', new_image_name, end_time - start_time)
        logging.info('approximate finish time: %.2f hours', approximate_finish_time)
