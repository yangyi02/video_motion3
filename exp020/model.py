import numpy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, net_depth):
        super(Net, self).__init__()
        num_hidden = 64
        if net_depth == 6:
            self.conv0 = nn.Conv2d(n_inputs * im_channel, num_hidden, 3, 1, 1)
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
            self.conv7 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
            self.bn7 = nn.BatchNorm2d(num_hidden)
            self.conv8 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
            self.bn8 = nn.BatchNorm2d(num_hidden)
            self.conv9 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
            self.bn9 = nn.BatchNorm2d(num_hidden)
            self.conv10 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
            self.bn10 = nn.BatchNorm2d(num_hidden)
            self.conv11 = nn.Conv2d(num_hidden * 2, num_hidden, 3, 1, 1)
            self.bn11 = nn.BatchNorm2d(num_hidden)
        elif net_depth == 7:
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
        self.conv = nn.Conv2d(num_hidden, 2, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_inputs = n_inputs
        self.n_class = n_class
        self.m_range = m_range
        self.net_depth = net_depth

        x = numpy.linspace(-1, 1, im_width)
        y = numpy.linspace(-1, 1, im_height)
        xv, yv = numpy.meshgrid(x, y)
        grid = numpy.dstack((xv, yv))
        grid = numpy.expand_dims(grid, 0)
        self.grid = Variable(torch.from_numpy(grid).float(), requires_grad=False)
        if torch.cuda.is_available():
            self.grid = self.grid.cuda()

    def forward(self, im_input, im_output):
        if self.net_depth == 6:
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
            x6 = self.upsample(x6)
            x7 = torch.cat((x6, x5), 1)
            x7 = F.relu(self.bn7(self.conv7(x7)))
            x7 = self.upsample(x7)
            x8 = torch.cat((x7, x4), 1)
            x8 = F.relu(self.bn8(self.conv8(x8)))
            x8 = self.upsample(x8)
            x9 = torch.cat((x8, x3), 1)
            x9 = F.relu(self.bn9(self.conv9(x9)))
            x9 = self.upsample(x9)
            x10 = torch.cat((x9, x2), 1)
            x10 = F.relu(self.bn10(self.conv10(x10)))
            x10 = self.upsample(x10)
            x11 = torch.cat((x10, x1), 1)
            x11 = F.relu(self.bn11(self.conv11(x11)))
            flow = self.conv(x11)
        elif self.net_depth == 7:
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
            flow = self.conv(x13)

        flow[:, 0, :, :] = flow[:, 0, :, :] * 2.0 / self.im_height
        flow[:, 1, :, :] = flow[:, 1, :, :] * 2.0 / self.im_width
        grid = self.grid + flow.permute(0, 2, 3, 1)
        pred = F.grid_sample(im_output, grid)

        return pred, flow


class GtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range):
        super(GtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range

        x = numpy.linspace(-1, 1, im_width)
        y = numpy.linspace(-1, 1, im_height)
        xv, yv = numpy.meshgrid(x, y)
        grid = numpy.dstack((xv, yv))
        grid = numpy.expand_dims(grid, 0)
        self.grid = Variable(torch.from_numpy(grid).float(), requires_grad=False)
        if torch.cuda.is_available():
            self.grid = self.grid.cuda()

    def forward(self, im_input, im_output, gt_motion):
        flow = Variable(torch.zeros(gt_motion.size()))
        if torch.cuda.is_available():
            flow = flow.cuda()
        # grid_sample use [-1, 1] as the image scale, so need to rescale flow to [-1, 1]
        flow[:, 0, :, :] = gt_motion[:, 0, :, :] * 2.0 / self.im_height
        flow[:, 1, :, :] = gt_motion[:, 1, :, :] * 2.0 / self.im_width
        grid = self.grid - flow.permute(0, 2, 3, 1)
        pred = F.grid_sample(im_output, grid)
        return pred, flow
