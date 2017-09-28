import os
import math
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from visualize.base_visualizer import BaseVisualizer
from visualize import flowlib
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Synthetic2Data(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_objects = args.num_objects
        self.im_size = args.image_size
        self.im_channel = args.image_channel
        self.m_range = args.motion_range
        self.m_type = args.motion_type
        self.num_frame = args.num_frame
        self.bg_move = args.bg_move
        self.bg_noise = args.bg_noise
        self.m_dict, self.reverse_m_dict, self.m_kernel = self.motion_dict()
        self.visualizer = BaseVisualizer(args, self.reverse_m_dict)
        self.save_display = args.save_display
        self.save_display_dir = args.save_display_dir
        if args.fixed_data:
            numpy.random.seed(args.seed)

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

    def generate_data(self, src_image, src_mask):
        batch_size, im_size, im_channel = self.batch_size, self.im_size, self.im_channel
        # generate foreground motion
        src_motion, m = self.generate_motion(self.num_objects)
        # move foreground
        fg_im, fg_motion, fg_mask = self.move_foreground(src_image, src_motion, src_mask, m)
        # generate background
        src_bg = numpy.random.rand(batch_size, im_size, im_size, im_channel) * self.bg_noise
        # generate background motion
        if self.bg_move:
            src_motion, m = self.generate_motion(num_objects=1)
        else:
            src_motion = numpy.zeros((1, batch_size, im_size, im_size, 2))
            m = numpy.zeros((1, batch_size, 2, 3))
            m[0, :, 0, 0] = 1
            m[0, :, 1, 1] = 1
        # move background
        bg_im, bg_motion = self.move_background(src_bg, src_motion, m)
        # merge foreground and background, merge foreground motion and background motion
        mask = fg_mask == 0
        motion_mask = numpy.concatenate((mask, mask), 4)
        fg_motion[motion_mask] = bg_motion[motion_mask]
        if self.im_channel == 1:
            im_mask = mask
        elif self.im_channel == 3:
            im_mask = numpy.concatenate((mask, mask, mask), 4)
        fg_im[im_mask] = bg_im[im_mask]
        # swap axes between bacth size and frame
        im = fg_im.swapaxes(0, 1).swapaxes(3, 4).swapaxes(2, 3)
        motion = fg_motion.swapaxes(0, 1).swapaxes(3, 4).swapaxes(2, 3)
        seg_layer = fg_mask.swapaxes(0, 1).swapaxes(3, 4).swapaxes(2, 3)
        return im, motion, seg_layer.astype(int)

    def generate_motion(self, num_objects):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        m_type = self.m_type
        m = numpy.zeros((num_objects, batch_size, 2, 3))
        if m_type == 'translation_discrete':
            t_x = numpy.random.randint(-m_range, m_range, size=(num_objects, batch_size))
            t_y = numpy.random.randint(-m_range, m_range, size=(num_objects, batch_size))
            m[:, :, 0, 0] = 1
            m[:, :, 1, 1] = 1
            m[:, :, 0, 2] = t_x
            m[:, :, 1, 2] = t_y
        elif m_type == 'translation':
            t_x = numpy.random.uniform(-m_range, m_range, size=(num_objects, batch_size))
            t_y = numpy.random.uniform(-m_range, m_range, size=(num_objects, batch_size))
            m[:, :, 0, 0] = 1
            m[:, :, 1, 1] = 1
            m[:, :, 0, 2] = t_x
            m[:, :, 1, 2] = t_y
        elif m_type == 'rotation':
            max_angle = m_range * 1.0 / im_size * 180 / math.pi * 1.8
            angle = numpy.random.uniform(-max_angle, max_angle, size=(num_objects, batch_size))
            for i in range(num_objects):
                for j in range(batch_size):
                    m[i, j, :, :] = cv2.getRotationMatrix2D((im_size/2, im_size/2), angle[i, j], 1)
        elif m_type == 'affine':
            rand_idx = numpy.random.randint(0, 4)
            if rand_idx == 0:
                pts1 = numpy.float32([[0, 0], [im_size, 0], [im_size, im_size]])
            elif rand_idx == 1:
                pts1 = numpy.float32([[im_size, 0], [im_size, im_size], [0, im_size]])
            elif rand_idx == 2:
                pts1 = numpy.float32([[im_size, im_size], [0, im_size], [0, 0]])
            elif rand_idx == 3:
                pts1 = numpy.float32([[0, im_size], [0, 0], [im_size, 0]])
            max_shift = m_range * 0.33
            for i in range(num_objects):
                for j in range(batch_size):
                    shift = numpy.random.uniform(-max_shift, max_shift, size=(3, 2))
                    pts2 = pts1 + shift.astype(numpy.float32)
                    m[i, j, :, :] = cv2.getAffineTransform(pts1, pts2)
        elif m_type == 'perspective':
            logging.error('motion generation for perspective transformation is not implemented')
            pts1 = numpy.float32([[0, 0], [im_size, 0], [0, im_size], [im_size, im_size]])
            for i in range(num_objects):
                for j in range(batch_size):
                    shift = numpy.random.uniform(-m_range, m_range, size=(4, 2))
                    pts2 = pts1 + shift.astype(numpy.float32)
                    m[i, j, :, :] = cv2.getPerspectiveTransform(pts1, pts2)
        src_motion = numpy.zeros((num_objects, batch_size, im_size, im_size, 2))
        pts = numpy.mgrid[0:im_size, 0:im_size].reshape(2, -1)
        pts = pts[::-1, :]
        pts = numpy.concatenate((pts, numpy.ones((1, pts.shape[1]))), 0)
        for i in range(num_objects):
            for j in range(batch_size):
                if m_type == 'perspective':
                    logging.error('ground truth for perspective transformation is not implemented')
                motion = numpy.dot(m[i, j, :, :], pts) - pts[:2, :]
                src_motion[i, j, :, :, :] = motion.swapaxes(0, 1).reshape(im_size, im_size, 2)
        return src_motion, m

    def move_foreground(self, src_image, src_motion, src_mask, m):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, im_size, im_size, self.im_channel))
        motion = numpy.zeros((num_frame, batch_size, im_size, im_size, 2))
        mask = numpy.zeros((num_frame, batch_size, im_size, im_size, 1))
        for i in range(num_frame):
            im[i, ...], motion[i, ...], mask[i, ...] = \
                self.merge_objects(src_image, src_motion, src_mask)
            src_image = self.move_image_fg(src_image, m)
            src_mask = self.move_mask(src_mask, m)
        return im, motion, mask

    def merge_objects(self, src_image, src_motion, src_mask):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((batch_size, im_size, im_size, self.im_channel))
        motion = numpy.zeros((batch_size, im_size, im_size, 2))
        mask = numpy.zeros((batch_size, im_size, im_size, 1))
        for i in range(num_objects):
            zero_mask = mask == 0
            zero_motion_mask = numpy.concatenate((zero_mask, zero_mask), 3)
            if self.im_channel == 1:
                zero_im_mask = zero_mask
            elif self.im_channel == 3:
                zero_im_mask = numpy.concatenate((zero_mask, zero_mask, zero_mask), 3)
            im[zero_im_mask] = src_image[i, zero_im_mask]
            motion[zero_motion_mask] = src_motion[i, zero_motion_mask]
            mask[zero_mask] = src_mask[i, zero_mask] * (num_objects - i)
        return im, motion, mask

    def move_image_fg(self, im, m):
        batch_size, num_objects, im_size, = self.batch_size, self.num_objects, self.im_size
        m_type = self.m_type
        im_new = numpy.zeros((num_objects, batch_size, im_size, im_size, self.im_channel))
        for i in range(num_objects):
            for j in range(batch_size):
                curr_im = im[i, j, :, :, :].squeeze()
                if m_type in ['translation', 'translation_discrete', 'rotation', 'affine']:
                    if self.im_channel == 1:
                        im_new[i, j, :, :, 0] = cv2.warpAffine(curr_im, m[i, j, :, :], (im_size, im_size))
                    elif self.im_channel == 3:
                        im_new[i, j, ...] = cv2.warpAffine(curr_im, m[i, j, :, :], (im_size, im_size))
                elif m_type == 'perspective':
                    if self.im_channel == 1:
                        im_new[i, j, :, :, 0] = cv2.warpAffine(curr_im, m[i, j, :, :], (im_size, im_size))
                    elif self.im_channel == 3:
                        im_new[i, j, ...] = cv2.warpPerspective(curr_im, m[i, j, :, :], (im_size, im_size))
        return im_new

    def move_mask(self, mask, m):
        batch_size, num_objects, im_size, = self.batch_size, self.num_objects, self.im_size
        m_type = self.m_type
        mask_new = numpy.zeros((num_objects, batch_size, im_size, im_size, 1))
        for i in range(num_objects):
            for j in range(batch_size):
                curr_mask = mask[i, j, :, :, :].squeeze()
                if m_type in ['translation', 'translation_discrete', 'rotation', 'affine']:
                    mask_new[i, j, :, :, 0] = cv2.warpAffine(curr_mask, m[i, j, :, :], (im_size, im_size))
                elif m_type == 'perspective':
                    mask_new[i, j, :, :, 0] = cv2.warpPerspective(curr_mask, m[i, j, :, :], (im_size, im_size))
        mask_new = numpy.round(mask_new)
        return mask_new

    def move_background(self, src_image, src_motion, m):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame, self.im_size
        im = numpy.zeros((num_frame, batch_size, im_size, im_size, self.im_channel))
        motion = numpy.zeros((num_frame, batch_size, im_size, im_size, 2))
        for i in range(num_frame):
            im[i, :, :, :, :] = src_image
            src_image = self.move_image_bg(src_image, m)
            motion[i, :, :, :, :] = src_motion
        return im, motion

    def move_image_bg(self, bg_im, m):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        m_type = self.m_type
        im_new = numpy.zeros((batch_size, im_size, im_size, self.im_channel))
        for i in range(batch_size):
            curr_im = bg_im[i, :, :, :].squeeze()
            if m_type in ['translation', 'translation_discrete', 'rotation', 'affine']:
                if self.im_channel == 1:
                    im_new[i, :, :, 0] = cv2.warpAffine(curr_im, m[0, i, :, :], (im_size, im_size))
                elif self.im_channel == 3:
                    im_new[i, ...] = cv2.warpAffine(curr_im, m[0, i, :, :], (im_size, im_size))
            elif m_type == 'perspective':
                if self.im_channel == 1:
                    im_new[i, :, :, 0] = cv2.warpPerspective(curr_im, m[0, i, :, :], (im_size, im_size))
                elif self.im_channel == 3:
                    im_new[i, ...] = cv2.warpPerspective(curr_im, m[0, i, :, :], (im_size, im_size))
        return im_new

    def display(self, im, motion, seg_layer):
        num_frame = self.num_frame
        width, height = self.visualizer.get_img_size(4, num_frame)
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

            flow = motion[0, i, :, :, :].transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(flow, self.m_range)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(3, i + 1)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

            seg_max = seg_layer[0, i, :, :, :].max()
            seg = seg_layer[0, i, :, :, :].squeeze() * 1.0 / seg_max
            cmap = plt.get_cmap('jet')
            seg_map = cmap(seg)[:, :, 0:3]
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(4, i + 1)
            img[y1:y2, x1:x2, :] = seg_map

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
