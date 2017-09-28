import os
import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data.synthetic.mnist_data_bidirect import MnistDataBidirect
from data.synthetic.box_data_bidirect import BoxDataBidirect
from data.synthetic.box_data_bicomplex import BoxDataBicomplex
from base_demo import BaseDemo
from model import Net, GtNet
from visualizer import Visualizer
import logging

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Demo(BaseDemo):
    def __init__(self, args):
        super(Demo, self).__init__(args)
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = Visualizer(args, self.data.reverse_m_dict)

    def init_model(self, m_kernel):
        num_inputs = (self.num_frame - 1) / 2
        self.model = Net(self.im_size, self.im_size, self.im_channel, num_inputs,
                         m_kernel.shape[1], self.m_range, m_kernel)
        self.model_gt = GtNet(self.im_size, self.im_size, self.im_channel, num_inputs,
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
            if self.data.name in ['box_bidirect', 'mnist_bidirect', 'box_bicomplex']:
                im, _, _, _, _, _ = self.data.get_next_batch(self.data.train_images)
            elif self.data.name in ['robot64_bidirect', 'mpii64_bidirect', 'nyuv2_bidirect']:
                im = self.data.get_next_batch(self.data.train_meta)
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
            im_pred, m_mask_f, d_mask_f, attn_f, m_mask_b, d_mask_b, attn_b = self.model(im_input_f,
                                                                                         im_input_b)
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])
            if len(train_loss) > 100:
                train_loss.pop(0)
            ave_train_loss = sum(train_loss) / float(len(train_loss))
            im_base = (im_input_f[:, -self.im_channel:, :, :] + im_input_b[:, -self.im_channel:, :,
                                                                :]) / 2
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            if len(base_loss) > 100:
                base_loss.pop(0)
            ave_base_loss = sum(base_loss) / float(len(base_loss))
            logging.info('epoch %d, train loss: %.2f, average train loss: %.2f, base loss: %.2f',
                         epoch, loss.data[0], ave_train_loss, ave_base_loss)
            if (epoch + 1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                self.validate()

    def test(self):
        base_loss, test_loss = [], []
        test_epe = []
        for epoch in range(self.test_epoch):
            if self.data.name in ['box_bidirect', 'mnist_bidirect', 'box_bicomplex']:
                im, motion, motion_r, _, _, _ = self.data.get_next_batch(self.data.test_images)
            elif self.data.name in ['robot64_bidirect', 'mpii64_bidirect', 'nyuv2_bidirect']:
                im, motion = self.data.get_next_batch(self.data.test_meta), None
            elif self.data.name in ['mpii64_sample']:
                im, motion = self.data.get_next_batch(self.data.test_meta), None
                im = im[:, -self.num_frame:, :, :, :]
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            if motion is not None:
                gt_motion_f = motion[:, self.num_inputs - 1, :, :, :]
                gt_motion_b = motion_r[:, self.num_inputs + 1, :, :, :]
                gt_depth_f = gt_depth[:, self.num_inputs - 1, :, :, :]
                gt_depth_b = gt_depth[:, self.num_inputs + 1, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if motion is not None:
                gt_motion_f = Variable(torch.from_numpy(gt_motion_f).float())
                gt_motion_b = Variable(torch.from_numpy(gt_motion_b).float())
                gt_depth_f = Variable(torch.from_numpy(gt_depth_f).float())
                gt_depth_b = Variable(torch.from_numpy(gt_depth_b).float())
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
            im_pred, m_mask_f, d_mask_f, attn_f, m_mask_b, d_mask_b, attn_b = self.model(im_input_f,
                                                                                         im_input_b)
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()

            test_loss.append(loss.data[0])
            im_base = (im_input_f[:, -self.im_channel:, :, :] + im_input_b[:, -self.im_channel:, :,
                                                                :]) / 2
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            flow_f = self.motion2flow(m_mask_f)
            depth_f = self.mask2depth(d_mask_f)
            flow_b = self.motion2flow(m_mask_b)
            depth_b = self.mask2depth(d_mask_b)

            if motion is None:
                gt_motion = None
            else:
                gt_motion_f = motion[:, -2, :, :, :]
                gt_motion_f = Variable(torch.from_numpy(gt_motion_f).float())
                if torch.cuda.is_available():
                    gt_motion_f = gt_motion_f.cuda()
                epe = (flow_f - gt_motion_f) * (flow_f - gt_motion_f)
                epe = torch.sqrt(epe.sum(1))
                epe = epe.sum() / epe.numel()
                test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input_f, im_input_b, im_output, im_pred, flow_f,
                                                 gt_motion_f,
                                                 depth_f, attn_f, flow_b, gt_motion_b, depth_b,
                                                 attn_b,
                                                 'test_%d.png' % epoch)
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input_f, im_input_b, im_output, im_pred,
                                                     flow_f, gt_motion_f,
                                                     depth_f, attn_f, flow_b, gt_motion_b, depth_b,
                                                     attn_b,
                                                     'test_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        if gt_motion is not None:
            test_epe = numpy.mean(numpy.asarray(test_epe))
            logging.info('average test endpoint error: %.2f', test_epe)
        return improve_percent

    def test_gt(self):
        base_loss, test_loss = [], []
        test_epe = []
        for epoch in range(self.test_epoch):
            if self.data.name in ['box_bidirect', 'mnist_bidirect', 'box_bicomplex']:
                im, motion, motion_label, depth = self.data.get_next_batch(self.data.test_images)
                gt_motion_label = motion_label[:, -2, :, :, :]
                gt_motion_label = Variable(torch.from_numpy(gt_motion_label))
                if torch.cuda.is_available():
                    gt_motion_label = gt_motion_label.cuda()
            else:
                logging.error('%s data not supported in test_gt' % self.data.name)
                sys.exit()
            im_input_f = im[:, :self.num_inputs, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_input_b = im[:, :self.num_inputs:-1, :, :, :].reshape(
                self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, self.num_inputs, :, :, :]
            gt_motion_f = motion[:, self.num_inputs - 1, :, :, :]
            gt_motion_b = motion_r[:, self.num_inputs + 1, :, :, :]
            gt_motion_label_f = motion_label[:, self.num_inputs - 1, :, :, :]
            gt_motion_label_b = motion_label_r[:, self.num_inputs + 1, :, :, :]
            gt_depth_f = gt_depth[:, self.num_inputs - 1, :, :, :]
            gt_depth_b = gt_depth[:, self.num_inputs + 1, :, :, :]
            im_input_f = Variable(torch.from_numpy(im_input_f).float())
            im_input_b = Variable(torch.from_numpy(im_input_b).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion_f = Variable(torch.from_numpy(gt_motion_f).float())
            gt_motion_b = Variable(torch.from_numpy(gt_motion_b).float())
            gt_motion_label_f = Variable(torch.from_numpy(gt_motion_label_f))
            gt_motion_label_b = Variable(torch.from_numpy(gt_motion_label_b))
            gt_depth_f = Variable(torch.from_numpy(gt_depth_f).float())
            gt_depth_b = Variable(torch.from_numpy(gt_depth_b).float())
            if torch.cuda.is_available():
                im_input_f, im_input_b = im_input_f.cuda(), im_input_b.cuda()
                im_output = im_output.cuda()
                gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
                gt_motion_label_f, gt_motion_label_b = gt_motion_label_f.cuda(), gt_motion_label_b.cuda()
                gt_depth_f, gt_depth_b = gt_depth_f.cuda(), gt_depth_b.cuda()
            im_pred, m_mask_f, d_mask_f, attn_f, m_mask_b, d_mask_b, attn_b = self.model_gt(
                im_input_f, im_input_b, gt_motion_label, gt_depth, 'label')
            im_diff = im_pred - im_output
            loss = torch.abs(im_diff).sum()

            test_loss.append(loss.data[0])
            im_base = (im_input_f[:, -self.im_channel:, :, :] + im_input_b[:, -self.im_channel:, :,
                                                                :]) / 2
            base_loss.append(torch.abs(im_base - im_output).sum().data[0])
            flow_f = self.motion2flow(m_mask_f)
            depth_f = self.mask2depth(d_mask_f)
            epe = (flow_f - gt_motion) * (flow_f - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe.append(epe.cpu().data[0])
            flow_b = self.motion2flow(m_mask_b)
            depth_b = self.mask2depth(d_mask_b)
            epe = (flow_b - gt_motion) * (flow_b - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input_f, im_input_b, im_output, im_pred, flow_f,
                                                 gt_motion_f, depth_f, attn_f, flow_b, gt_motion_b,
                                                 depth_b, attn_b, 'test_gt.png')
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input_f, im_input_b, im_output, im_pred,
                                                     flow_f, gt_motion_f, depth_f, attn_f, flow_b,
                                                     gt_motion_b, depth_b, attn_b,
                                                     'test_gt_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average ground truth test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_epe = numpy.mean(numpy.asarray(test_epe))
        logging.info('average ground truth test endpoint error: %.2f', test_epe)
        return improve_percent

    def mask2depth(self, d_mask):
        [batch_size, num_depth, height, width] = d_mask.size()
        depth_number = Variable(torch.zeros(batch_size, num_depth, height, width))
        if torch.cuda.is_available():
            depth_number = depth_number.cuda()
        for i in range(num_depth):
            depth_number[:, i, :, :] = i
        depth = Variable(torch.zeros(batch_size, 1, height, width))
        if torch.cuda.is_available():
            depth = depth.cuda()
        depth[:, 0, :, :] = (d_mask * depth_number).sum(1)
        return depth


def main():
    args = parse_args()
    logging.info(args)
    demo = Demo(args)
    if args.train:
        demo.train()
    if args.test:
        demo.test()
    if args.test_gt:
        demo.test_gt()


if __name__ == '__main__':
    main()
