import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(Net, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(n_inputs*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.conv13 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)

        self.conv0_d = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0_d = nn.BatchNorm2d(num_hidden)
        self.conv1_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_d = nn.BatchNorm2d(num_hidden)
        self.conv2_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_d = nn.BatchNorm2d(num_hidden)
        self.conv3_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_d = nn.BatchNorm2d(num_hidden)
        self.conv4_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_d = nn.BatchNorm2d(num_hidden)
        self.conv5_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_d = nn.BatchNorm2d(num_hidden)
        self.conv6_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_d = nn.BatchNorm2d(num_hidden)
        self.conv7_d = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7_d = nn.BatchNorm2d(num_hidden)
        self.conv8_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn8_d = nn.BatchNorm2d(num_hidden)
        self.conv9_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn9_d = nn.BatchNorm2d(num_hidden)
        self.conv10_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn10_d = nn.BatchNorm2d(num_hidden)
        self.conv11_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn11_d = nn.BatchNorm2d(num_hidden)
        self.conv12_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn12_d = nn.BatchNorm2d(num_hidden)
        self.conv13_d = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
        self.bn13_d = nn.BatchNorm2d(num_hidden)
        self.conv_d = nn.Conv2d(num_hidden, 2, 3, 1, 1)

        self.conv0_v = nn.Conv2d(n_inputs*im_channel, num_hidden, 3, 1, 1)
        self.bn0_v = nn.BatchNorm2d(num_hidden)
        self.conv1_v = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_v = nn.BatchNorm2d(num_hidden)
        self.conv_v = nn.Conv2d(num_hidden, 3, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_inputs = n_inputs
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input):
        x = self.bn0(self.conv0(im_input))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x13 = torch.cat((x12, x1), 1)
        x13 = F.relu(self.bn13(self.conv13(x13)))
        m_mask = F.softmax(self.conv(x13))

        im = im_input[:, -self.im_channel:, :, :]
        x = self.bn0_d(self.conv0_d(im))
        x1 = F.relu(self.bn1_d(self.conv1_d(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2_d(self.conv2_d(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3_d(self.conv3_d(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4_d(self.conv4_d(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5_d(self.conv5_d(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6_d(self.conv6_d(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7_d(self.conv7_d(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8_d(self.conv8_d(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9_d(self.conv9_d(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10_d(self.conv10_d(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11_d(self.conv11_d(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12_d(self.conv12_d(x12)))
        x12 = self.upsample(x12)
        x13 = torch.cat((x12, x1), 1)
        x13 = F.relu(self.bn13_d(self.conv13_d(x13)))
        d_mask = F.softmax(self.conv_d(x13))

        x = F.relu(self.bn0_v(self.conv0_v(im_input)))
        x = F.relu(self.bn1_v(self.conv1_v(x)))
        bg = F.sigmoid(self.conv_v(x))

        out_mask = construct_mask(m_mask, d_mask, self.m_kernel, self.m_range)
        im = im_input[:, -self.im_channel:, :, :]
        pred = construct_image(im, out_mask, self.m_kernel, self.m_range)

        seg = out_mask.sum(1).unsqueeze(1)
        pred = pred * seg + bg * (1 - seg)
        return pred, m_mask, d_mask, 1 - seg


class GtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(GtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()
        self.n_depth = 2

    def forward(self, im_input, gt_motion, gt_depth, gt_type=None):
        if gt_type == 'label':
            m_mask = self.label2mask(gt_motion)
        else:
            m_mask = self.motion2mask(gt_motion)
        d_mask = self.depth2mask(gt_depth)

        out_mask = construct_mask(m_mask, d_mask, self.m_kernel, self.m_range)
        im = im_input[:, -self.im_channel:, :, :]
        pred = construct_image(im, out_mask, self.m_kernel, self.m_range)
        seg = out_mask.sum(1).unsqueeze(1)
        return pred, m_mask, d_mask, 1 - seg

    def label2mask(self, motion):
        m_mask = Variable(torch.Tensor(motion.size(0), self.n_class, motion.size(2), motion.size(3)))
        if torch.cuda.is_available():
            m_mask = m_mask.cuda()
        for i in range(motion.size(0)):
            for j in range(self.n_class):
                tmp = Variable(torch.zeros((motion.size(2), motion.size(3))))
                if torch.cuda.is_available():
                    tmp = tmp.cuda()
                tmp[motion[i, 0, :, :] == j] = 1
                m_mask[i, j, :, :] = tmp
        return m_mask

    def motion2mask(self, motion):
        m_mask = Variable(torch.Tensor(motion.size(0), self.n_class, motion.size(2), motion.size(3)))
        if torch.cuda.is_available():
            m_mask = m_mask.cuda()
        motion_floor = torch.floor(motion.cpu().data).long()
        for i in range(motion.size(0)):
            for j in range(motion.size(2)):
                for k in range(motion.size(3)):
                    a = Variable(torch.zeros(int(math.sqrt(self.n_class))))
                    b = Variable(torch.zeros(int(math.sqrt(self.n_class))))
                    idx = motion_floor[i, 0, j, k] + self.m_range
                    a[idx] = 1 - (motion[i, 0, j, k] - motion_floor[i, 0, j, k])
                    a[idx + 1] = 1 - a[idx]
                    idx = motion_floor[i, 1, j, k] + self.m_range
                    b[idx] = 1 - (motion[i, 1, j, k] - motion_floor[i, 1, j, k])
                    b[idx + 1] = 1 - b[idx]
                    tmp = torch.ger(b, a)
                    m_mask[i, :, j, k] = tmp.view(-1)
        return m_mask

    def depth2mask(self, depth):
        d_mask = Variable(torch.Tensor(depth.size(0), self.n_depth, depth.size(2), depth.size(3)))
        if torch.cuda.is_available():
            d_mask = d_mask.cuda()
        for i in range(depth.size(0)):
            for j in range(self.n_depth):
                tmp = Variable(torch.zeros((depth.size(2), depth.size(3))))
                if torch.cuda.is_available():
                    tmp = tmp.cuda()
                tmp[depth[i, 0, :, :] == j] = 1
                d_mask[i, j, :, :] = tmp
        return d_mask


def construct_mask(m_mask, d_mask, m_kernel, m_range):
    flow_mask = F.conv2d(m_mask, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
    out_mask = Variable(torch.zeros(flow_mask.size()))
    if torch.cuda.is_available():
        out_mask = out_mask.cuda()
    for i in range(d_mask.size(1)):
        depth_mask = d_mask[:, -i-1, :, :].unsqueeze(1).expand_as(flow_mask)
        curr_mask = F.conv2d(depth_mask, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
        curr_prob = flow_mask * curr_mask + 1e-8
        curr_total_prob = curr_prob.sum(1)
        left_prob = F.relu(1 - out_mask.sum(1))
        ratio = left_prob / curr_total_prob
        out_mask = out_mask + curr_prob * (1 - F.relu(1 - ratio)).unsqueeze(1)
    return out_mask


def construct_image(im, out_mask, m_kernel, m_range):
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(out_mask)
        nearby_im = F.conv2d(im_expand, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
        pred[:, i, :, :] = (nearby_im * out_mask).sum(1)
    return pred
