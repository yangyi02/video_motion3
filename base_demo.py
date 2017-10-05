import os
import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data.synthetic.box_data import BoxData
from data.synthetic.box_data_complex import BoxDataComplex
from data.synthetic.mnist_data import MnistData
from data.synthetic.box2_data import Box2Data
from data.synthetic.mnist2_data import Mnist2Data
from data.real.robot64_data import Robot64Data
from data.real.robot128_data import Robot128Data
from data.real.mpii64_data import Mpii64Data
from data.real.mpii128_data import Mpii128Data
from data.real.mpii64_sample import Mpii64Sample
from data.real.nyuv2_data import Nyuv2Data
from data.real.viper64_data import Viper64Data
from data.real.viper128_data import Viper128Data
from base_model import BaseNet, BaseGtNet
from visualize.base_visualizer import BaseVisualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class BaseDemo(object):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.train_epoch = args.train_epoch
        self.test_epoch = args.test_epoch
        self.test_interval = args.test_interval
        self.save_interval = args.save_interval
        self.save_dir = args.save_dir
        self.display = args.display
        self.display_all = args.display_all
        self.best_improve_percent = -1e10
        self.batch_size = args.batch_size
        self.im_size = args.image_size
        self.im_channel = args.image_channel
        self.num_frame = args.num_frame
        self.m_range = args.motion_range
        if args.data == 'box':
            self.data = BoxData(args)
        elif args.data == 'box_complex':
            self.data = BoxDataComplex(args)
        elif args.data == 'mnist':
            self.data = MnistData(args)
        elif args.data == 'box2':
            self.data = Box2Data(args)
        elif args.data == 'mnist2':
            self.data = Mnist2Data(args)
        elif args.data == 'robot64':
            self.data = Robot64Data(args)
        elif args.data == 'robot128':
            self.data = Robot128Data(args)
        elif args.data == 'mpii64':
            self.data = Mpii64Data(args)
        elif args.data == 'mpii128':
            self.data = Mpii128Data(args)
        elif args.data == 'mpii64_sample':
            self.data = Mpii64Sample(args)
        elif args.data == 'nyuv2':
            self.data = Nyuv2Data(args)
        elif args.data == 'viper64':
            self.data = Viper64Data(args)
        elif args.data == 'viper128':
            self.data = Viper128Data(args)
        else:
            logging.error('%s data not supported' % args.data)
        self.init_model_path = args.init_model_path
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = BaseVisualizer(args, self.data.reverse_m_dict)

    def init_model(self, m_kernel):
        self.model = BaseNet(self.im_size, self.im_size, self.im_channel, self.num_frame - 1,
                             m_kernel.shape[1], self.m_range, m_kernel)
        self.model_gt = BaseGtNet(self.im_size, self.im_size, self.im_channel, self.num_frame - 1,
                                  m_kernel.shape[1], self.m_range, m_kernel)
        if torch.cuda.is_available():
            # model = torch.nn.DataParallel(model).cuda()
            self.model = self.model.cuda()
            self.model_gt = self.model_gt.cuda()
        if self.init_model_path is not '':
            self.model.load_state_dict(torch.load(self.init_model_path))
        return self.model, self.model_gt

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        base_loss, train_loss = [], []
        for epoch in range(self.train_epoch):
            optimizer.zero_grad()
            if self.data.name in ['box', 'mnist', 'box_complex']:
                im, _, _, _ = self.data.get_next_batch(self.data.train_images)
            elif self.data.name in ['box2', 'mnist2']:
                im, _, _ = self.data.get_next_batch(self.data.train_images)
            elif self.data.name in ['robot64', 'mpii64', 'mpi128', 'nyuv2', 'robot128', 'viper64', 'viper128']:
                im = self.data.get_next_batch(self.data.train_images)
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, m_mask, disappear, appear = self.model(im_input)
            loss = torch.abs(im_pred - im_output).sum()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])
            if len(train_loss) > 100:
                train_loss.pop(0)
            ave_train_loss = sum(train_loss) / float(len(train_loss))
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            if len(base_loss) > 100:
                base_loss.pop(0)
            ave_base_loss = sum(base_loss) / float(len(base_loss))
            logging.info('epoch %d, train loss: %.2f, average train loss: %.2f, base loss: %.2f',
                         epoch, loss.data[0], ave_train_loss, ave_base_loss)
            if (epoch+1) % self.save_interval == 0:
                logging.info('epoch %d, saving model', epoch)
                with open(os.path.join(self.save_dir, '%d.pth' % epoch), 'w') as handle:
                    torch.save(self.model.state_dict(), handle)
            if (epoch+1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                self.validate()

    def validate(self):
        improve_percent = self.test()
        if improve_percent >= self.best_improve_percent:
            logging.info('model save to %s', os.path.join(self.save_dir, 'model.pth'))
            with open(os.path.join(self.save_dir, 'model.pth'), 'w') as handle:
                torch.save(self.model.state_dict(), handle)
            self.best_improve_percent = improve_percent
        logging.info('current best improved percent: %.2f', self.best_improve_percent)

    def test(self):
        base_loss, test_loss = [], []
        test_epe = []
        motion = None
        for epoch in range(self.test_epoch):
            if self.data.name in ['box', 'mnist', 'box_complex']:
                im, motion, _, _ = self.data.get_next_batch(self.data.test_images)
            elif self.data.name in ['box2', 'mnist2']:
                im, motion, _ = self.data.get_next_batch(self.data.test_images)
            elif self.data.name in ['robot64', 'mpii64', 'mpi128', 'nyuv2', 'robot128', 'viper64', 'viper128']:
                im, motion = self.data.get_next_batch(self.data.test_images), None
            elif self.data.name in ['mpii64_sample']:
                im, motion = self.data.get_next_batch(self.data.test_images), None
                im = im[:, -self.num_frame:, :, :, :]
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float(), volatile=True)
            im_output = Variable(torch.from_numpy(im_output).float(), volatile=True)
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, m_mask, disappear, appear = self.model(im_input)
            loss = torch.abs(im_pred - im_output).sum()

            test_loss.append(loss.data[0])
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            flow = self.motion2flow(m_mask)

            if motion is None:
                gt_motion = None
            else:
                gt_motion = motion[:, -2, :, :, :]
                gt_motion = Variable(torch.from_numpy(gt_motion).float())
                if torch.cuda.is_available():
                    gt_motion = gt_motion.cuda()
                epe = (flow - gt_motion) * (flow - gt_motion)
                epe = torch.sqrt(epe.sum(1))
                epe = epe.sum() / epe.numel()
                test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 disappear, appear, 'test_%d.png' % epoch)
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                     disappear, appear, 'test_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        if motion is not None:
            test_epe = numpy.mean(numpy.asarray(test_epe))
            logging.info('average test endpoint error: %.2f', test_epe)
        return improve_percent

    def test_gt(self):
        base_loss, test_loss = [], []
        test_epe = []
        for epoch in range(self.test_epoch):
            if self.data.name in ['box', 'mnist', 'box_complex']:
                im, motion, motion_label, _ = self.data.get_next_batch(self.data.test_images)
                gt_motion_label = motion_label[:, -2, :, :, :]
                gt_motion_label = Variable(torch.from_numpy(gt_motion_label))
                if torch.cuda.is_available():
                    gt_motion_label = gt_motion_label.cuda()
            elif self.data.name in ['box2', 'mnist2']:
                im, motion, _ = self.data.get_next_batch(self.data.test_images)
            else:
                logging.error('%s data not supported in test_gt' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            gt_motion = motion[:, -2, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float(), volatile=True)
            im_output = Variable(torch.from_numpy(im_output).float(), volatile=True)
            gt_motion = Variable(torch.from_numpy(gt_motion).float(), volatile=True)
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
            if self.data.name in ['box', 'mnist', 'box_complex']:
                im_pred, m_mask, disappear, appear = self.model_gt(im_input, gt_motion_label, 'label')
            elif self.data.name in['box2', 'mnist2']:
                im_pred, m_mask, disappear, appear = self.model_gt(im_input, gt_motion)
            loss = torch.abs(im_pred - im_output).sum()

            test_loss.append(loss.data[0])
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            flow = self.motion2flow(m_mask)
            epe = (flow - gt_motion) * (flow - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 disappear, appear, 'test_gt.png')
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                     disappear, appear, 'test_gt_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average ground truth test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_epe = numpy.mean(numpy.asarray(test_epe))
        logging.info('average ground truth test endpoint error: %.2f', test_epe)
        return improve_percent

    def test_all(self):
        for epoch in range(self.train_epoch):
            if (epoch + 1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                model_path = os.path.join(self.save_dir, '%d.pth' % epoch)
                self.model.load_state_dict(torch.load(model_path))
                improve_percent = self.test()
                if improve_percent >= self.best_improve_percent:
                    self.best_improve_percent = improve_percent
        logging.info('best improved percent: %.2f', self.best_improve_percent)

    def motion2flow(self, m_mask):
        reverse_m_dict = self.data.reverse_m_dict
        [batch_size, num_class, height, width] = m_mask.size()
        kernel_x = Variable(torch.zeros(batch_size, num_class, height, width))
        kernel_y = Variable(torch.zeros(batch_size, num_class, height, width))
        if torch.cuda.is_available():
            kernel_x = kernel_x.cuda()
            kernel_y = kernel_y.cuda()
        for i in range(num_class):
            (m_x, m_y) = reverse_m_dict[i]
            kernel_x[:, i, :, :] = m_x
            kernel_y[:, i, :, :] = m_y
        flow = Variable(torch.zeros(batch_size, 2, height, width))
        if torch.cuda.is_available():
            flow = flow.cuda()
        flow[:, 0, :, :] = (m_mask * kernel_x).sum(1)
        flow[:, 1, :, :] = (m_mask * kernel_y).sum(1)
        return flow


def main():
    args = parse_args()
    logging.info(args)
    demo = BaseDemo(args)
    if args.train:
        demo.train()
    if args.test:
        demo.test()
    if args.test_gt:
        demo.test_gt()
    if args.test_all:
        demo.test_all()


if __name__ == '__main__':
    main()
