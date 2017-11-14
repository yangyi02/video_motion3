import os
import numpy
import Image

from real_data import RealData
from visualize import flowlib
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class Kitti128Sample(RealData):
    def __init__(self, args):
        super(Kitti128Sample, self).__init__(args)
        self.name = 'kitti128_sample'
        self.train_dir = '/home/yi/code/video_motion_data/kitti128-train'
        self.test_dir = '/home/yi/code/video_motion_data/kitti128-test'
        self.gt_dir = '/media/yi/DATA/data-orig/kitti_flow/kitti_stereo_flow/training/flow_noc'
        self.train_images = self.get_meta(self.train_dir)
        self.test_images = self.get_meta(self.test_dir)
        if args.fixed_data:
            numpy.random.seed(args.seed)
        self.cnt = 0

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
                meta[cnt] = [image_names, sub_sub_dir]
                cnt += 1
        return meta

    def generate_data(self, meta):
        batch_size, im_size, num_frame = self.batch_size, self.im_size, self.num_frame
        im = numpy.zeros((batch_size, num_frame, self.im_channel, im_size, im_size))
        ratio = 370.0 / 128.0
        mo_size = int(im_size * ratio)
        motion = numpy.zeros((batch_size, mo_size, mo_size, 3))
        for i in range(batch_size):
            image_names = meta[self.cnt][0]
            image_names = image_names[12-num_frame:12]
            for j in range(num_frame):
                image = numpy.zeros((im_size, im_size, self.im_channel))
                if self.im_channel == 1:
                    image = numpy.array(Image.open(image_names[j]).convert('L')) / 255.0
                    image = numpy.expand_dims(image, 3)
                elif self.im_channel == 3:
                    image = numpy.array(Image.open(image_names[j])) / 255.0
                height, width = image.shape[0], image.shape[1]
                image = numpy.transpose(image, (2, 0, 1))
                center_h, center_w = height / 2, width / 2
                im[i, j, :, :, :] = image[:, center_h-im_size/2:center_h+im_size/2, center_w-im_size/2:center_w+im_size/2]
            # obtain sub folder name to acquire ground truth flow
            sub_sub_dir = meta[self.cnt][1]
            gt_png_file = os.path.join(self.gt_dir, '000' + sub_sub_dir + '_10.png')
            flow = flowlib.read_flow_png(gt_png_file)
            # img = flowlib.visualize_flow(motion)
            # import matplotlib.pyplot as plt
            # plt.figure(1)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            height, width = flow.shape[0], flow.shape[1]
            center_h, center_w = height / 2, width / 2
            flow = flow[center_h-mo_size/2:center_h+mo_size/2, center_w-mo_size/2:center_w+mo_size/2, :]
            motion[i, :, :, :] = flow
            self.cnt += 1
            if self.cnt > len(meta):
                self.cnt = 0
        return im, motion

    def get_next_batch(self, meta):
        im = self.generate_data(meta)
        return im


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = Kitti128Sample(args)
    im = data.get_next_batch(data.test_images)
    data.display(im)


if __name__ == '__main__':
    unit_test()
