import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image

from visualize.base_bivisualizer import BaseVisualizer
from visualize import flowlib
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class SyntheticBiData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_objects = args.num_objects
        self.im_size = args.image_size
        self.im_channel = args.image_channel
        self.m_range = args.motion_range
        self.num_frame = args.num_frame
        self.num_frame_one_direction = (self.num_frame + 1) / 2
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
            m_kernel[:, i, m_y[i] + m_range, m_x[i] + m_range] = 1
        return m_dict, reverse_m_dict, m_kernel

    def generate_data(self, src_image, src_mask, bg_color=None):
        batch_size, im_size, im_channel = self.batch_size, self.im_size, self.im_channel
        m_dict = self.m_dict
        # generate foreground motion
        src_motion_f, src_motion_label_f, m_x_f, m_y_f, src_motion_b, src_motion_label_b, m_x_b, m_y_b = \
            self.generate_motion(self.num_objects, src_mask)
        # move foreground forward
        fg_im_f, fg_motion_f, fg_motion_f_r, fg_motion_label_f, fg_motion_label_f_r, fg_mask_f = \
            self.move_foreground(src_image, src_motion_f, src_motion_b, src_motion_label_f, src_motion_label_b, src_mask, m_x_f, m_y_f)
        # move foreground backward
        fg_im_b, fg_motion_b, fg_motion_b_r, fg_motion_label_b, fg_motion_label_b_r, fg_mask_b = \
            self.move_foreground(src_image, src_motion_b, src_motion_f, src_motion_label_b, src_motion_label_f, src_mask, m_x_b, m_y_b)
        # generate background
        if bg_color is None:
            src_bg = numpy.random.rand(batch_size, im_channel, im_size, im_size) * self.bg_noise
        else:
            src_bg = numpy.tile(bg_color[:, :, None, None], (1, 1, im_size, im_size))
            noise = (numpy.random.rand(batch_size, im_channel, im_size, im_size) - 0.5) * self.bg_noise
            src_bg = src_bg + noise
            src_bg[src_bg < 0] = 0
            src_bg[src_bg > 1] = 1
        # generate background motion
        if self.bg_move:
            src_motion_f, src_motion_label_f, m_x_f, m_y_f, src_motion_b, src_motion_label_b, m_x_b, m_y_b = self.generate_motion(1)
        else:
            src_motion_f = numpy.zeros((1, batch_size, 2, im_size, im_size))
            src_motion_label_f = m_dict[(0, 0)] * numpy.ones((1, batch_size, 1, im_size, im_size))
            m_x_f = numpy.zeros((1, batch_size)).astype(int)
            m_y_f = numpy.zeros((1, batch_size)).astype(int)
            src_motion_b = numpy.zeros((1, batch_size, 2, im_size, im_size))
            src_motion_label_b = m_dict[(0, 0)] * numpy.ones((1, batch_size, 1, im_size, im_size))
            m_x_b = numpy.zeros((1, batch_size)).astype(int)
            m_y_b = numpy.zeros((1, batch_size)).astype(int)
        # move background
        bg_im_f, bg_motion_f, bg_motion_f_r, bg_motion_label_f, bg_motion_label_f_r = \
            self.move_background(src_bg, src_motion_f, src_motion_b, src_motion_label_f, src_motion_label_b, m_x_f, m_y_f, bg_color)
        bg_im_b, bg_motion_b, bg_motion_b_r, bg_motion_label_b, bg_motion_label_b_r = \
            self.move_background(src_bg, src_motion_b, src_motion_f, src_motion_label_b, src_motion_label_f, m_x_b, m_y_b, bg_color)
        # merge foreground and background, merge foreground motion and background motion
        mask = fg_mask_f == 0
        fg_motion_label_f[mask] = bg_motion_label_f[mask]
        fg_motion_label_f_r[mask] = bg_motion_label_f_r[mask]
        motion_mask = numpy.concatenate((mask, mask), 2)
        fg_motion_f[motion_mask] = bg_motion_f[motion_mask]
        fg_motion_f_r[motion_mask] = bg_motion_f_r[motion_mask]
        if self.im_channel == 1:
            im_mask = mask
        elif self.im_channel == 3:
            im_mask = numpy.concatenate((mask, mask, mask), 2)
        fg_im_f[im_mask] = bg_im_f[im_mask]
        mask = fg_mask_b == 0
        fg_motion_label_b[mask] = bg_motion_label_b[mask]
        fg_motion_label_b_r[mask] = bg_motion_label_b_r[mask]
        motion_mask = numpy.concatenate((mask, mask), 2)
        fg_motion_b[motion_mask] = bg_motion_b[motion_mask]
        fg_motion_b_r[motion_mask] = bg_motion_b_r[motion_mask]
        if self.im_channel == 1:
            im_mask = mask
        elif self.im_channel == 3:
            im_mask = numpy.concatenate((mask, mask, mask), 2)
        fg_im_b[im_mask] = bg_im_b[im_mask]
        # merge forward and backward frames and motions
        im, motion, motion_r, motion_label, motion_label_r, seg_layer = \
            self.merge_forward_backward(fg_im_f, fg_motion_f, fg_motion_f_r, fg_motion_label_f, fg_motion_label_f_r, fg_im_b, fg_motion_b, fg_motion_b_r, fg_motion_label_b, fg_motion_label_b_r, fg_mask_f, fg_mask_b)
        # swap axes between batch size and frame
        im, motion, motion_r, motion_label, motion_label_r, seg_layer = im.swapaxes(0, 1), motion.swapaxes(0, 1), motion_r.swapaxes(0, 1), motion_label.swapaxes(0, 1), motion_label_r.swapaxes(0, 1), seg_layer.swapaxes(0, 1)
        return im, motion.astype(int), motion_r.astype(int), motion_label.astype(int), motion_label_r.astype(int), seg_layer.astype(int)

    def generate_motion(self, num_objects, src_mask=None):
        batch_size, im_size = self.batch_size, self.im_size
        m_dict, reverse_m_dict = self.m_dict, self.reverse_m_dict
        m_label = numpy.random.randint(0, len(m_dict), size=(num_objects, batch_size))
        m_x_f = numpy.zeros((num_objects, batch_size)).astype(int)
        m_y_f = numpy.zeros((num_objects, batch_size)).astype(int)
        for i in range(num_objects):
            for j in range(batch_size):
                (m_x_f[i, j], m_y_f[i, j]) = reverse_m_dict[m_label[i, j]]
        m_x_b = numpy.zeros((num_objects, batch_size)).astype(int)
        m_y_b = numpy.zeros((num_objects, batch_size)).astype(int)
        for i in range(num_objects):
            for j in range(batch_size):
                (m_x_b[i, j], m_y_b[i, j]) = (-m_x_f[i, j], -m_y_f[i, j])
        src_motion_f = numpy.zeros((num_objects, batch_size, 2, im_size, im_size))
        src_motion_b = numpy.zeros((num_objects, batch_size, 2, im_size, im_size))
        src_motion_label_f = m_dict[(0, 0)] * numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        src_motion_label_b = m_dict[(0, 0)] * numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        if src_mask is None:
            src_mask = numpy.ones((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                mask = src_mask[i, j, 0, :, :] > 0
                src_motion_f[i, j, 0, mask] = m_x_f[i, j]
                src_motion_f[i, j, 1, mask] = m_y_f[i, j]
                src_motion_b[i, j, 0, mask] = m_x_b[i, j]
                src_motion_b[i, j, 1, mask] = m_y_b[i, j]
                src_motion_label_f[i, j, 0, mask] = m_label[i, j]
                src_motion_label_b[i, j, 0, mask] = m_dict[(m_x_b[i, j], m_y_b[i, j])]
        return src_motion_f, src_motion_label_f, m_x_f, m_y_f, src_motion_b, src_motion_label_b, m_x_b, m_y_b

    def move_foreground(self, src_image, src_motion, src_motion_r, src_motion_label, src_motion_label_r, src_mask, m_x, m_y):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame_one_direction, self.im_size
        im = numpy.zeros((num_frame, batch_size, self.im_channel, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_r = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        motion_label_r = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        mask = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, ...], motion[i, ...], motion_r[i, ...], motion_label[i, ...], motion_label_r[i, ...], mask[i, ...] = \
                self.merge_objects(src_image, src_motion, src_motion_r, src_motion_label, src_motion_label_r, src_mask)
            src_image = self.move_image_fg(src_image, m_x, m_y)
            src_motion = self.move_motion(src_motion, m_x, m_y)
            src_motion_r = self.move_motion(src_motion_r, m_x, m_y)
            src_motion_label = self.move_motion_label(src_motion_label, m_x, m_y)
            src_motion_label_r = self.move_motion_label(src_motion_label_r, m_x, m_y)
            src_mask = self.move_mask(src_mask, m_x, m_y)
        return im, motion, motion_r, motion_label, motion_label_r, mask

    def merge_objects(self, src_image, src_motion, src_motion_r, src_motion_label, src_motion_label_r, src_mask):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((batch_size, self.im_channel, im_size, im_size))
        motion = numpy.zeros((batch_size, 2, im_size, im_size))
        motion_r = numpy.zeros((batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((batch_size, 1, im_size, im_size))
        motion_label_r = numpy.zeros((batch_size, 1, im_size, im_size))
        mask = numpy.zeros((batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            zero_mask = mask == 0
            zero_motion_mask = numpy.concatenate((zero_mask, zero_mask), 1)
            if self.im_channel == 1:
                zero_im_mask = zero_mask
            elif self.im_channel == 3:
                zero_im_mask = numpy.concatenate((zero_mask, zero_mask, zero_mask), 1)
            im[zero_im_mask] = src_image[i, zero_im_mask]
            motion[zero_motion_mask] = src_motion[i, zero_motion_mask]
            motion_r[zero_motion_mask] = src_motion_r[i, zero_motion_mask]
            motion_label[zero_mask] = src_motion_label[i, zero_mask]
            motion_label_r[zero_mask] = src_motion_label_r[i, zero_mask]
            mask[zero_mask] = src_mask[i, zero_mask]
        return im, motion, motion_r, motion_label, motion_label_r, mask

    def move_image_fg(self, im, m_x, m_y):
        batch_size, im_size, im_channel = self.batch_size, self.im_size, self.im_channel
        m_range, num_objects = self.m_range, self.num_objects
        im_big = numpy.zeros(
            (num_objects, batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
        im_big[:, :, :, m_range:-m_range, m_range:-m_range] = im
        im_new = numpy.zeros((num_objects, batch_size, im_channel, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                im_new[i, j, :, :, :] = im_big[i, j, :, y:y + im_size, x:x + im_size]
        return im_new

    def move_motion(self, motion, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        m_dict = self.m_dict
        motion_big = m_dict[(0, 0)] * numpy.ones(
            (num_objects, batch_size, 2, im_size + m_range * 2, im_size + m_range * 2))
        motion_big[:, :, :, m_range:-m_range, m_range:-m_range] = motion
        motion_new = numpy.zeros((num_objects, batch_size, 2, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                motion_new[i, j, :, :, :] = motion_big[i, j, :, y:y + im_size, x:x + im_size]
        return motion_new

    def move_motion_label(self, motion_label, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        motion_label_big = numpy.zeros(
            (num_objects, batch_size, 1, im_size + m_range * 2, im_size + m_range * 2))
        motion_label_big[:, :, :, m_range:-m_range, m_range:-m_range] = motion_label
        motion_label_new = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                motion_label_new[i, j, :, :, :] = \
                    motion_label_big[i, j, :, y:y + im_size, x:x + im_size]
        return motion_label_new

    def move_mask(self, mask, m_x, m_y):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        num_objects = self.num_objects
        mask_big = numpy.zeros(
            (num_objects, batch_size, 1, im_size + m_range * 2, im_size + m_range * 2))
        mask_big[:, :, :, m_range:-m_range, m_range:-m_range] = mask
        mask_new = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                x = m_range + m_x[i, j]
                y = m_range + m_y[i, j]
                mask_new[i, j, :, :, :] = mask_big[i, j, :, y:y + im_size, x:x + im_size]
        return mask_new

    def move_background(self, src_image, src_motion, src_motion_r, src_motion_label, src_motion_label_r, m_x, m_y, bg_color=None):
        batch_size, num_frame, im_size = self.batch_size, self.num_frame_one_direction, self.im_size
        im = numpy.zeros((num_frame, batch_size, self.im_channel, im_size, im_size))
        motion = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_r = numpy.zeros((num_frame, batch_size, 2, im_size, im_size))
        motion_label = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        motion_label_r = numpy.zeros((num_frame, batch_size, 1, im_size, im_size))
        for i in range(num_frame):
            im[i, :, :, :, :] = src_image
            src_image = self.move_image_bg(src_image, m_x, m_y, bg_color)
            motion[i, :, :, :, :] = src_motion
            motion_r[i, :, :, :] = src_motion_r
            motion_label[i, :, :, :, :] = src_motion_label
            motion_label_r[i, :, :, :, :] = src_motion_label_r
        return im, motion, motion_r, motion_label, motion_label_r

    def move_image_bg(self, bg_im, m_x, m_y, bg_color):
        batch_size, im_size, m_range = self.batch_size, self.im_size, self.m_range
        im_channel = self.im_channel
        bg_noise = self.bg_noise
        if bg_color is None:
            im_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2,
                                       im_size + m_range * 2) * bg_noise
        else:
            im_big = numpy.tile(bg_color[:, :, None, None], (1, 1, im_size + m_range * 2, im_size + m_range * 2))
            noise = (numpy.random.rand(batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2) - 0.5) * self.bg_noise
            im_big = im_big + noise
            im_big[im_big < 0] = 0
            im_big[im_big > 1] = 1

        im_big[:, :, m_range:-m_range, m_range:-m_range] = bg_im
        im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
        for i in range(batch_size):
            x = m_range + m_x[0, i]
            y = m_range + m_y[0, i]
            im_new[i, :, :, :] = im_big[i, :, y:y + im_size, x:x + im_size]
        return im_new

    @staticmethod
    def merge_forward_backward(im_f, motion_f, motion_f_r, motion_label_f, motion_label_f_r, im_b, motion_b, motion_b_r, motion_label_b, motion_label_b_r, mask_f, mask_b):
        im_f = im_f[1:, ...]
        im_b = im_b[::-1, ...]
        im = numpy.concatenate((im_b, im_f), 0)
        motion_f = motion_f[1:, ...]
        motion_b_r = motion_b_r[::-1, ...]
        motion = numpy.concatenate((motion_b_r, motion_f), 0)
        motion_f_r = motion_f_r[1:, ...]
        motion_b = motion_b[::-1, ...]
        motion_r = numpy.concatenate((motion_b, motion_f_r), 0)
        motion_label_f = motion_label_f[1:, ...]
        motion_label_b_r = motion_label_b_r[::-1, ...]
        motion_label = numpy.concatenate((motion_label_b_r, motion_label_f), 0)
        motion_label_f_r = motion_label_f_r[1:, ...]
        motion_label_b = motion_label_b[::-1, ...]
        motion_label_r = numpy.concatenate((motion_label_b, motion_label_f_r), 0)
        mask_f = mask_f[1:, ...]
        mask_b = mask_b[::-1, ...]
        mask = numpy.concatenate((mask_b, mask_f), 0)
        return im, motion, motion_r, motion_label, motion_label_r, mask

    def display(self, im, motion, motion_r, seg_layer):
        num_frame = self.num_frame
        width, height = self.visualizer.get_img_size(5, num_frame)
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

            flow = motion_r[0, i, :, :, :].transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(flow, self.m_range)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(4, i + 1)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

            seg_max = seg_layer[0, i, :, :, :].max()
            seg = seg_layer[0, i, :, :, :].squeeze() * 1.0 / seg_max
            cmap = plt.get_cmap('jet')
            seg_map = cmap(seg)[:, :, 0:3]
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(5, i + 1)
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
