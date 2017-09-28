import numpy

from synthetic_data import SyntheticData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class BoxData(SyntheticData):
    def __init__(self, args):
        super(BoxData, self).__init__(args)
        self.name = 'box'
        self.fg_noise = args.fg_noise
        self.bg_noise = args.bg_noise
        self.train_images, self.test_images = None, None
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def generate_source_image(self):
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((num_objects, batch_size, self.im_channel, im_size, im_size))
        mask = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            for j in range(batch_size):
                width = numpy.random.randint(im_size/8, im_size*3/4)
                height = numpy.random.randint(im_size/8, im_size*3/4)
                x = numpy.random.randint(0, im_size - width)
                y = numpy.random.randint(0, im_size - height)
                color = numpy.random.uniform(self.bg_noise, 1 - self.fg_noise, self.im_channel)
                for k in range(self.im_channel):
                    im[i, j, k, y:y+height, x:x+width] = color[k]
                noise = numpy.random.rand(self.im_channel, height, width) * self.fg_noise
                im[i, j, :, y:y+height, x:x+width] = im[i, j, :, y:y+height, x:x+width] + noise
                mask[i, j, 0, y:y+height, x:x+width] = num_objects - i
        return im, mask

    def get_next_batch(self, images=None):
        src_image, src_mask = self.generate_source_image()
        im, motion, motion_label, seg_layer = self.generate_data(src_image, src_mask)
        return im, motion, motion_label, seg_layer


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = BoxData(args)
    im, motion, motion_label, seg_layer = data.get_next_batch()
    data.display(im, motion, seg_layer)

if __name__ == '__main__':
    unit_test()
