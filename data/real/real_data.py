import os
import math
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
import pickle
import cv2

from visualize.base_visualizer import BaseVisualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class RealData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.im_size = args.image_size
        self.im_channel = args.image_channel
        self.m_range = args.motion_range
        self.num_frame = args.num_frame
        self.m_dict, self.reverse_m_dict, self.m_kernel = self.motion_dict()
        self.visualizer = BaseVisualizer(args, self.reverse_m_dict)
        self.save_display = args.save_display
        self.save_display_dir = args.save_display_dir
        self.min_diff_thresh = args.min_diff_thresh
        self.max_diff_thresh = args.max_diff_thresh
        self.diff_div_thresh = args.diff_div_thresh
        self.fixed_data = args.fixed_data
        if args.fixed_data:
            numpy.random.seed(args.seed)
        self.rand_noise = args.rand_noise
        self.augment_reverse = args.augment_reverse

    def motion_dict(self):
        m_range = self.m_range
        m_dict, reverse_m_dict = {}, {}
        x = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
        y = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
        m_x, m_y = numpy.meshgrid(x, y)
        m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
        m_kernel = numpy.zeros((1, len(m_x), 2 * m_range + 1, 2 * m_range + 1))
        for i in range(len(m_x)):
            m_dict[(m_x[i], m_y[i])] = i
            reverse_m_dict[i] = (m_x[i], m_y[i])
            m_kernel[:, i, -m_y[i] + m_range, -m_x[i] + m_range] = 1
        return m_dict, reverse_m_dict, m_kernel

    def get_meta(self, image_dir):
        meta, cnt = {}, 0
        for sub_dir in os.listdir(image_dir):
            for sub_sub_dir in os.listdir(os.path.join(image_dir, sub_dir)):
                image_files = os.listdir(os.path.join(image_dir, sub_dir, sub_sub_dir))
                image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
                image_names = [os.path.join(image_dir, sub_dir, sub_sub_dir, f) for f in image_files]
                num_images = len(image_names)
                if num_images < self.num_frame:
                    continue
                for i in range(num_images - self.num_frame + 1):
                    meta[cnt] = image_names[i:i + self.num_frame]
                    cnt += 1
        return meta

    def generate_data(self, meta):
        batch_size, im_size, num_frame = self.batch_size, self.im_size, self.num_frame
        im_channel = self.im_channel
        min_diff_thresh, max_diff_thresh = self.min_diff_thresh, self.max_diff_thresh
        diff_div_thresh = self.diff_div_thresh
        idx = numpy.random.permutation(len(meta))
        im = numpy.zeros((batch_size, num_frame, im_channel, im_size, im_size))
        i, cnt = 0, 0
        while i < batch_size:
            image_names = meta[idx[cnt]]
            for j in range(len(image_names)):
                if im_channel == 1:
                    image = numpy.array(Image.open(image_names[j]).convert('L')) / 255.0
                    image = numpy.expand_dims(image, 3)
                elif im_channel == 3:
                    image = numpy.array(Image.open(image_names[j])) / 255.0
                if j == 0:
                    height, width = image.shape[0], image.shape[1]
                    idx_h = numpy.random.randint(0, height + 1 - im_size)
                    idx_w = numpy.random.randint(0, width + 1 - im_size)
                image = image.transpose((2, 0, 1))
                im[i, j, :, :, :] = image[:, idx_h:idx_h+im_size, idx_w:idx_w+im_size]
            cnt = cnt + 1
            im_diff = numpy.zeros((num_frame - 1))
            for j in range(num_frame - 1):
                diff = numpy.abs(im[i, j, :, :, :] - im[i, j+1, :, :, :])
                im_diff[j] = numpy.sum(diff) / im_channel / im_size / im_size
            if any(im_diff < min_diff_thresh) or any(im_diff > max_diff_thresh):
                continue
            if num_frame > 2:
                im_diff_div = im_diff / (numpy.median(im_diff) + 1e-5)
                if any(im_diff_div > diff_div_thresh) or any(im_diff_div < 1/diff_div_thresh):
                    continue
            if self.augment_reverse:
                if numpy.random.rand() < 0.5:
                    im[i, :, :, :, :] = im[i, ::-1, :, :, :]
            i = i + 1
        im = im + (numpy.random.rand(batch_size, num_frame, im_channel, im_size, im_size) - 0.5) * self.rand_noise
        im[im > 1] = 1
        im[im < 0] = 0
        return im

    def display(self, im):
        num_frame, im_channel = self.num_frame, self.im_channel
        width, height = self.visualizer.get_img_size(2, num_frame)
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(num_frame):
            curr_im = im[0, i, :, :, :].transpose(1, 2, 0)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(1, i + 1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.visualizer.get_img_coordinate(2, i + 1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        if self.save_display:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_display_dir, 'data.png'))
        else:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
