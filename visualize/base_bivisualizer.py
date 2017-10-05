import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import flowlib
import pyflowlib


class BaseVisualizer(object):
    def __init__(self, args, reverse_m_dict):
        self.reverse_m_dict = reverse_m_dict
        self.im_height = args.image_size
        self.im_width = args.image_size
        self.im_channel = args.image_channel
        self.num_frame = args.num_frame
        self.num_inputs = (self.num_frame - 1) / 2
        self.save_display = args.save_display
        self.save_display_dir = args.save_display_dir

    def visualize_result(self, im_input_f, im_input_b, im_output, im_pred, pred_motion_f,
                         gt_motion_f, attn_f, pred_motion_b, gt_motion_b, attn_b,
                         file_name='tmp.png', idx=0):
        width, height = self.get_img_size(4, max(self.num_frame + 1, 3))
        im_channel = self.im_channel
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(self.num_inputs):
            curr_im = im_input_f[idx, i*im_channel:(i+1)*im_channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.get_img_coordinate(1, i + 1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.get_img_coordinate(2, i)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        im_output = im_output[idx].cpu().data.numpy().transpose(1, 2, 0)
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_inputs + 1)
        img[y1:y2, x1:x2, :] = im_output

        im_diff = numpy.abs(im_output - prev_im)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_inputs)
        img[y1:y2, x1:x2, :] = im_diff

        for i in range(self.num_inputs):
            curr_im = im_input_b[idx, i*im_channel:(i+1)*im_channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame - i)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame - i + 1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        im_diff = numpy.abs(im_output - prev_im)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_inputs + 2)
        img[y1:y2, x1:x2, :] = im_diff

        im_pred = im_pred[idx].cpu().data.numpy().transpose(1, 2, 0)
        im_pred[im_pred > 1] = 1
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame + 1)
        img[y1:y2, x1:x2, :] = im_pred

        im_diff = numpy.abs(im_pred - im_output)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame + 1)
        img[y1:y2, x1:x2, :] = im_diff

        pred_motion = pred_motion_f[idx].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(pred_motion)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 1)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        if gt_motion_f is None:
            prev_im = im_input_f[idx, -2*im_channel:-im_channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
            im = prev_im * 255.0
            if im_channel == 1:
                prvs_frame = im.astype(numpy.uint8)
            elif im_channel == 3:
                prvs_frame = cv2.cvtColor(im.astype(numpy.uint8), cv2.COLOR_RGB2GRAY)
            im = im_output * 255.0
            if im_channel == 1:
                next_frame = im.astype(numpy.uint8)
            elif im_channel == 3:
                next_frame = cv2.cvtColor(im.astype(numpy.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs_frame, next_frame, None, 0.5, 5, 5, 3, 5, 1.1, 0)
            optical_flow = flowlib.visualize_flow(flow)
            x1, y1, x2, y2 = self.get_img_coordinate(3, 2)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

            prvs_frame = numpy.asarray(prev_im * 255.0, order='C')
            next_frame = numpy.asarray(im_output * 255.0, order='C')
            flow = pyflowlib.calculate_flow(prvs_frame, next_frame)
            optical_flow = flowlib.visualize_flow(flow)
            x1, y1, x2, y2 = self.get_img_coordinate(3, 3)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0
        else:
            gt_motion = gt_motion_f[idx].cpu().data.numpy().transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(gt_motion)
            x1, y1, x2, y2 = self.get_img_coordinate(3, 2)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

        attn = attn_f[idx].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        attn_map = cmap(attn)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 4)
        img[y1:y2, x1:x2, :] = attn_map

        pred_motion = pred_motion_b[idx].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(pred_motion)
        x1, y1, x2, y2 = self.get_img_coordinate(4, 1)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        if gt_motion_b is None:
            prev_im = im_input_b[idx, -2*im_channel:-im_channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
            im = prev_im * 255.0
            if im_channel == 1:
                prvs_frame = im.astype(numpy.uint8)
            elif im_channel == 3:
                prvs_frame = cv2.cvtColor(im.astype(numpy.uint8), cv2.COLOR_RGB2GRAY)
            im = im_output * 255.0
            if im_channel == 1:
                next_frame = im.astype(numpy.uint8)
            elif im_channel == 3:
                next_frame = cv2.cvtColor(im.astype(numpy.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs_frame, next_frame, None, 0.5, 5, 5, 3, 5, 1.1, 0)
            optical_flow = flowlib.visualize_flow(flow)
            x1, y1, x2, y2 = self.get_img_coordinate(4, 2)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

            prvs_frame = numpy.asarray(prev_im * 255.0, order='C')
            next_frame = numpy.asarray(im_output * 255.0, order='C')
            flow = pyflowlib.calculate_flow(prvs_frame, next_frame)
            optical_flow = flowlib.visualize_flow(flow)
            x1, y1, x2, y2 = self.get_img_coordinate(4, 3)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0
        else:
            gt_motion = gt_motion_b[idx].cpu().data.numpy().transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(gt_motion)
            x1, y1, x2, y2 = self.get_img_coordinate(4, 2)
            img[y1:y2, x1:x2, :] = optical_flow / 255.0

        attn = attn_b[0].cpu().data.numpy().squeeze()
        cmap = plt.get_cmap('jet')
        attn_map = cmap(attn)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(4, 4)
        img[y1:y2, x1:x2, :] = attn_map

        if self.save_display:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_display_dir, file_name))
        else:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def get_img_size(self, n_row, n_col):
        im_width, im_height = self.im_width, self.im_height
        height = n_row * im_height + (n_row - 1) * int(im_height/10)
        width = n_col * im_width + (n_col - 1) * int(im_width/10)
        return width, height

    def get_img_coordinate(self, row, col):
        im_width, im_height = self.im_width, self.im_height
        y1 = (row - 1) * im_height + (row - 1) * int(im_height/10)
        y2 = y1 + im_height
        x1 = (col - 1) * im_width + (col - 1) * int(im_width/10)
        x2 = x1 + im_width
        return x1, y1, x2, y2
