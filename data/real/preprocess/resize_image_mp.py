import io
import os
import sys
import cv2
import argparse
import logging
import time
import multiprocessing
logging.getLogger().setLevel(logging.INFO)


def resize_image(image_file):
    image_name = image_file['image_name']
    input_dir = image_file['input_dir']
    output_dir = image_file['output_dir']
    im_size = image_file['im_size']
    im = cv2.imread(os.path.join(input_dir, image_name))
    if im.shape[0] < im.shape[1]:
        new_height = im_size
        new_width = int(round(float(im_size) / im.shape[0] * im.shape[1]))
    else:
        new_width = im_size
        new_height = int(round(float(im_size) / im.shape[1] * im.shape[0]))
    # It is best to use cv2.INTER_AREA when shrinking an image
    im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)
    new_image_name = os.path.join(output_dir, image_name)
    cv2.imwrite(new_image_name, im)
    im = None
    logging.info('Resize image save to %s, height %d, width %d', new_image_name, new_height, new_width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--file_list', default='')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    im_size = args.size
    file_list = args.file_list

    # Create output directories
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

    files = io.open(file_list).readlines()
    image_files = []
    for image_name in files:
        image_file = {'input_dir': input_dir, 'output_dir': output_dir, 'image_name': image_name.strip(), 'im_size': im_size}
        image_files.append(image_file)

    # Start multiprocessing image resize
    start_time = time.time()
    core_ct = os.sysconf('SC_NPROCESSORS_ONLN')
    pool = multiprocessing.Pool(processes=core_ct)
    pool.map(resize_image, image_files)
    pool.close()
    pool.join()
    end_time = time.time()
    total_time = end_time - start_time
    logging.info('finish in %.2f hours', total_time / 60 / 60)

if __name__ == '__main__':
    main()
