import numpy

from real_data import RealData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Viper128Data(RealData):
    def __init__(self, args):
        super(Viper128Data, self).__init__(args)
        self.name = 'viper128'
        self.train_dir = '/home/yi/code/video_motion_data/viper128-train'
        self.test_dir = '/home/yi/code/video_motion_data/viper128-val'
        self.train_images = self.get_meta(self.train_dir)
        self.test_images = self.get_meta(self.test_dir)
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def get_next_batch(self, meta):
        im = self.generate_data(meta)
        return im


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = Viper128Data(args)
    im = data.get_next_batch(data.train_images)
    data.display(im)


if __name__ == '__main__':
    unit_test()
