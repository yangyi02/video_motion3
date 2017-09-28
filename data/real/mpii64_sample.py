import os
import numpy
import Image

from real_data import RealData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class Mpii64Sample(RealData):
    def __init__(self, args):
        super(Mpii64Sample, self).__init__(args)
        self.name = 'mpii64_sample'
        script_dir = os.path.dirname(__file__)  # absolute dir the script is in
        self.train_dir = '/home/yi/code/video_motion_data/mpii64-train'
        self.test_dir = os.path.join(script_dir, 'mpii64-test-sample')
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
                meta[cnt] = image_names
                cnt += 1
        return meta

    def generate_data(self, meta):
        batch_size, im_size, num_frame = self.batch_size, self.im_size, self.num_frame
        im = numpy.zeros((batch_size, num_frame, self.im_channel, im_size, im_size))
        for i in range(batch_size):
            image_names = meta[self.cnt]
            for j in range(num_frame):
                image = numpy.zeros((im_size, im_size, self.im_channel))
                if self.im_channel == 1:
                    image = numpy.array(Image.open(image_names[j]).convert('L')) / 255.0
                    image = numpy.expand_dims(image, 3)
                elif self.im_channel == 3:
                    image = numpy.array(Image.open(image_names[j])) / 255.0
                image = numpy.transpose(image, (2, 0, 1))
                im[i, j, :, :, :] = image[:, 0:im_size, 0:im_size]
            self.cnt += 1
            if self.cnt > len(meta):
                self.cnt = 0
        return im

    def get_next_batch(self, meta):
        im = self.generate_data(meta)
        return im


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = Mpii64Sample(args)
    im = data.get_next_batch(data.test_images)
    data.display(im)


if __name__ == '__main__':
    unit_test()
