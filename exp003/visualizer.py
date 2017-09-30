import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from visualize.base_visualizer import BaseVisualizer
from visualize import flowlib


class Visualizer(BaseVisualizer):
    def __init__(self, args, reverse_m_dict):
        super(Visualizer, self).__init__(args, reverse_m_dict)

    def visualize_result(self, im_input, im_output, im_pred, pred_motion, gt_motion, depth, file_name='tmp.png', idx=0):
        width, height = self.get_img_size(3, max(self.num_frame + 1, 4))
        im_channel = self.im_channel
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(self.num_frame - 1):
            curr_im = im_input[idx, i*im_channel:(i+1)*im_channel, :, :].cpu().data.numpy().transpose(1, 2, 0)
            x1, y1, x2, y2 = self.get_img_coordinate(1, i + 1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.get_img_coordinate(2, i)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        im_output = im_output[idx].cpu().data.numpy().transpose(1, 2, 0)
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame)
        img[y1:y2, x1:x2, :] = im_output

        im_diff = numpy.abs(im_output - prev_im)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame - 1)
        img[y1:y2, x1:x2, :] = im_diff

        im_pred = im_pred[idx].cpu().data.numpy().transpose(1, 2, 0)
        im_pred[im_pred > 1] = 1
        x1, y1, x2, y2 = self.get_img_coordinate(1, self.num_frame + 1)
        img[y1:y2, x1:x2, :] = im_pred

        im_diff = numpy.abs(im_pred - im_output)
        x1, y1, x2, y2 = self.get_img_coordinate(2, self.num_frame)
        img[y1:y2, x1:x2, :] = im_diff

        pred_motion = pred_motion[idx].cpu().data.numpy().transpose(1, 2, 0)
        optical_flow = flowlib.visualize_flow(pred_motion)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 1)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        if gt_motion is None:
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
        else:
            gt_motion = gt_motion[idx].cpu().data.numpy().transpose(1, 2, 0)
            optical_flow = flowlib.visualize_flow(gt_motion)
        x1, y1, x2, y2 = self.get_img_coordinate(3, 2)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        depth = depth[idx].cpu().data.numpy().squeeze()
        depth = depth * 1.0 / depth.max()
        cmap = plt.get_cmap('jet')
        depth_map = cmap(depth)[:, :, 0:3]
        x1, y1, x2, y2 = self.get_img_coordinate(3, 4)
        img[y1:y2, x1:x2, :] = depth_map

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
