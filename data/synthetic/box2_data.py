import numpy
import cv2

from synthetic2_data import Synthetic2Data
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class Box2Data(Synthetic2Data):
    def __init__(self, args):
        super(Box2Data, self).__init__(args)
        self.name = 'box2'
        self.fg_noise = args.fg_noise
        self.bg_noise = args.bg_noise
        self.train_images, self.test_images = None, None
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def generate_source_image(self):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((num_objects, batch_size, im_size, im_size, self.im_channel))
        mask = numpy.zeros((num_objects, batch_size, im_size, im_size, 1))
        for i in range(num_objects):
            for j in range(batch_size):
                width = numpy.random.randint(int(im_size / 8), int(im_size * 3 / 4))
                height = numpy.random.randint(int(im_size / 8), int(im_size * 3 / 4))
                x = numpy.random.randint(0, im_size - width)
                y = numpy.random.randint(0, im_size - height)
                color = numpy.random.uniform(self.bg_noise, 1 - self.fg_noise, self.im_channel)
                for k in range(self.im_channel):
                    im[i, j, y:y+height, x:x+width, k] = color[k]
                noise = numpy.random.rand(height, width, self.im_channel) * self.fg_noise
                im[i, j, y:y+height, x:x+width, :] = im[i, j, y:y+height, x:x+width, :] + noise
                # im[i, j, :, :, :] = cv2.GaussianBlur(im[i, j, :, :, :], (5, 5), 0)
                mask[i, j, y:y+height, x:x+width, 0] = 1
        return im, mask

    def get_next_batch(self, images=None):
        src_image, src_mask = self.generate_source_image()
        im, motion, seg_layer = self.generate_data(src_image, src_mask)
        return im, motion, seg_layer


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = Box2Data(args)
    im, motion, seg_layer = data.get_next_batch()
    print motion.max(), motion.min()
    data.display(im, motion, seg_layer)

if __name__ == '__main__':
    unit_test()
