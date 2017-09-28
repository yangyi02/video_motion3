import os
import sys
import numpy
from PIL import Image
import cv2

from real_data import RealData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Nyuv2Data(RealData):
    def __init__(self, args):
        super(Nyuv2Data, self).__init__(args)
        self.name = 'nyuv2'
        self.train_dir = '/home/yi/code/video_motion_data/nyuv2-train'
        self.test_dir = '/home/yi/code/video_motion_data/nyuv2-test'
        self.train_images = self.get_meta(self.train_dir)
        self.test_images = self.get_meta(self.test_dir)
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def get_meta(self, image_dir):
        meta, cnt = {}, 0
        for sub_dir in os.listdir(image_dir):
            for sub_sub_dir in os.listdir(os.path.join(image_dir, sub_dir)):
                image_files = os.listdir(os.path.join(image_dir, sub_dir, sub_sub_dir))
                image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
                image_names = [os.path.join(image_dir, sub_dir, sub_sub_dir, f) for f in image_files]
                num_images = len(image_names)
                for i in range(num_images):
                    meta[cnt] = image_names[i]
                    cnt += 1
        return meta

    def generate_data(self, meta):
        batch_size, im_size, num_frame = self.batch_size, self.im_size, self.num_frame
        min_diff_thresh, max_diff_thresh = self.min_diff_thresh, self.max_diff_thresh
        diff_div_thresh = self.diff_div_thresh
        idx = numpy.random.permutation(len(meta))
        if num_frame > 3:
            logging.error('maximum number of frames: 3')
            sys.exit()
        im = numpy.zeros((batch_size, num_frame, self.im_channel, im_size, im_size))
        i, cnt = 0, 0
        while i < batch_size:
            image_name = meta[idx[cnt]]
            if self.im_channel == 1:
                image = numpy.array(Image.open(image_name).convert('L')) / 255.0
                image = numpy.expand_dims(image, 3)
            elif self.im_channel == 3:
                image = numpy.array(Image.open(image_name)) / 255.0
            images = numpy.zeros((3, 70, 100, self.im_channel))
            im1 = image[:, :320, :]
            if self.im_channel == 1:
                images[0, :, :, 0] = cv2.resize(im1, (100, 70), interpolation=cv2.INTER_AREA)
            elif self.im_channel == 3:
                images[0, :, :, :] = cv2.resize(im1, (100, 70), interpolation=cv2.INTER_AREA)
            im2 = image[:, 320:640, :]
            if self.im_channel == 1:
                images[1, :, :, 0] = cv2.resize(im2, (100, 70), interpolation=cv2.INTER_AREA)
            elif self.im_channel == 3:
                images[1, :, :, :] = cv2.resize(im2, (100, 70), interpolation=cv2.INTER_AREA)
            im3 = image[:, 640:, :]
            if self.im_channel == 1:
                images[2, :, :, 0] = cv2.resize(im3, (100, 70), interpolation=cv2.INTER_AREA)
            elif self.im_channel == 3:
                images[2, :, :, :] = cv2.resize(im3, (100, 70), interpolation=cv2.INTER_AREA)
            images = images.transpose((0, 3, 1, 2))
            for j in range(num_frame):
                if j == 0:
                    _, _, height, width = images.shape
                    idx_h = numpy.random.randint(0, height + 1 - im_size)
                    idx_w = numpy.random.randint(0, width + 1 - im_size)
                im[i, j, :, :, :] = images[j, :, idx_h:idx_h+im_size, idx_w:idx_w+im_size]
            cnt = cnt + 1
            im_diff = numpy.zeros((num_frame - 1))
            for j in range(num_frame - 1):
                diff = numpy.abs(im[i, j, :, :, :] - im[i, j+1, :, :, :])
                im_diff[j] = numpy.sum(diff) / self.im_channel / im_size / im_size
            if any(im_diff < min_diff_thresh) or any(im_diff > max_diff_thresh):
                continue
            if num_frame > 2:
                im_diff_div = im_diff / (numpy.median(im_diff) + 1e-5)
                if any(im_diff_div > diff_div_thresh) or any(im_diff_div < 1/diff_div_thresh):
                    continue
            i = i + 1
        return im

    def get_next_batch(self, meta):
        im = self.generate_data(meta)
        return im


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = Nyuv2Data(args)
    im = data.get_next_batch(data.train_images)
    data.display(im)


if __name__ == '__main__':
    unit_test()
