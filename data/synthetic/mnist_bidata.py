import os
import numpy
import h5py

from synthetic_bidata import SyntheticBiData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class MnistData(SyntheticBiData):
    def __init__(self, args):
        super(MnistData, self).__init__(args)
        self.name = 'mnist'
        script_dir = os.path.dirname(__file__)  # absolute dir the script is in
        self.file_name = os.path.join(script_dir, 'mnist.h5')
        self.train_images, self.test_images = self.load_mnist()
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def load_mnist(self):
        f = h5py.File(self.file_name)
        train_images = f['train'].value.reshape(-1, 28, 28)
        train_images = numpy.expand_dims(train_images, 1)
        if self.im_channel == 3:
            train_images = numpy.concatenate((train_images, train_images, train_images), 1)
        test_images = f['test'].value.reshape(-1, 28, 28)
        test_images = numpy.expand_dims(test_images, 1)
        if self.im_channel == 3:
            test_images = numpy.concatenate((test_images, test_images, test_images), 1)
        return train_images, test_images

    def generate_source_image(self, images):
        max_shift = self.im_size / 4
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im = numpy.zeros((num_objects, batch_size, self.im_channel, im_size, im_size))
        mask = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        for i in range(num_objects):
            idx = numpy.random.permutation(images.shape[0])
            mnist_im = images[idx[0:batch_size], :, :, :]
            for j in range(batch_size):
                width = mnist_im.shape[3]
                height = mnist_im.shape[2]
                x = numpy.random.randint(0, im_size - width)
                y = numpy.random.randint(0, im_size - height)
                im[i, j, :, y:y+height, x:x+width] = mnist_im[j, :, :, :]
                nonzero_mask = im[i, j, :, :, :].sum(0) > 0
                mask[i, j, 0, nonzero_mask] = num_objects - i
                shift = numpy.random.randint(-max_shift, max_shift, size=2)
                im[i, j, :, :, :] = self.shift_image(im[i, j, :, :, :], shift, max_shift)
                mask[i, j, :, :, :] = self.shift_image(mask[i, j, :, :, :], shift, max_shift)
        return im, mask

    @staticmethod
    def shift_image(im, shift, max_shift):
        [im_channel, im_height, im_width] = im.shape
        im_big = numpy.zeros((im_channel, im_height + 2 * max_shift, im_width + 2 * max_shift))
        im_big[:, max_shift:-max_shift, max_shift:-max_shift] = im
        x = max_shift + shift[0]
        y = max_shift + shift[1]
        im = im_big[:, y:y + im_height, x:x + im_width]
        return im

    def get_next_batch(self, images):
        src_image, src_mask = self.generate_source_image(images)
        im, motion, motion_r, motion_label, motion_label_r, seg_layer = self.generate_data(src_image, src_mask)
        return im, motion, motion_r, motion_label, motion_label_r, seg_layer


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = MnistData(args)
    im, motion, motion_r, motion_label, motion_label_r, seg_layer = data.get_next_batch(data.train_images)
    data.display(im, motion, motion_r, seg_layer)


if __name__ == '__main__':
    unit_test()
