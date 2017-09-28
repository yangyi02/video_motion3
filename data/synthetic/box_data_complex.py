import numpy

from synthetic_data import SyntheticData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class BoxDataComplex(SyntheticData):
    def __init__(self, args):
        super(BoxDataComplex, self).__init__(args)
        self.name = 'box_complex'
        self.train_images, self.test_images = None, None
        self.fg_noise = args.fg_noise
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
                color = numpy.random.uniform(0, 1, self.im_channel)
                for k in range(self.im_channel):
                    im[i, j, k, y:y+height, x:x+width] = color[k]
                noise = (numpy.random.rand(self.im_channel, height, width) - 0.5) * self.fg_noise
                im[i, j, :, y:y+height, x:x+width] = im[i, j, :, y:y+height, x:x+width] + noise
                im[im < 0] = 0
                im[im > 1] = 1
                mask[i, j, 0, y:y+height, x:x+width] = num_objects - i
        bg_color = numpy.zeros((batch_size, self.im_channel))
        for i in range(batch_size):
            bg_color[i, :] = numpy.random.uniform(0, 1, self.im_channel)
        return im, mask, bg_color

    def get_next_batch(self, images=None):
        src_image, src_mask, bg_color = self.generate_source_image()
        im, motion, motion_label, seg_layer = self.generate_data(src_image, src_mask, bg_color)
        return im, motion, motion_label, seg_layer


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = BoxDataComplex(args)
    im, motion, motion_label, seg_layer = data.get_next_batch()
    data.display(im, motion, seg_layer)

if __name__ == '__main__':
    unit_test()
