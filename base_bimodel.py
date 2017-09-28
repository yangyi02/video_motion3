import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BaseNet, self).__init__()
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
        self.conv7 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)

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

    def forward(self, im_input_f, im_input_b):
        x = self.bn0(self.conv0(im_input_f))
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
        m_mask_f = F.softmax(self.conv(x11))

        x = self.bn0(self.conv0(im_input_b))
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
        m_mask_b = F.softmax(self.conv(x11))

        out_mask_f = construct_mask(m_mask_f, self.m_kernel, self.m_range)
        im = im_input_f[:, -self.im_channel:, :, :]
        pred_f = construct_image(im, out_mask_f, self.m_kernel, self.m_range)

        out_mask_b = construct_mask(m_mask_b, self.m_kernel, self.m_range)
        im = im_input_b[:, -self.im_channel:, :, :]
        pred_b = construct_image(im, out_mask_b, self.m_kernel, self.m_range)

        seg_f = out_mask_f.sum(1).unsqueeze(1)
        seg_b = out_mask_b.sum(1).unsqueeze(1)
        attn_f = (seg_f + 1e-8) / (seg_f + seg_b + 2e-8)
        attn_b = 1 - attn_f
        pred = pred_f * attn_f + pred_b * attn_b
        return pred, m_mask_f, attn_f, m_mask_b, attn_b


class BaseGtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BaseGtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input_f, im_input_b, gt_motion_f, gt_motion_b, gt_type=None):
        if gt_type == 'label':
            m_mask_f = self.label2mask(gt_motion_f)
            m_mask_b = self.label2mask(gt_motion_b)
        else:
            m_mask_f = self.motion2mask(gt_motion_f)
            m_mask_b = self.motion2mask(gt_motion_b)

        out_mask_f = construct_mask(m_mask_f, self.m_kernel, self.m_range)
        im = im_input_f[:, -self.im_channel:, :, :]
        pred_f = construct_image(im, out_mask_f, self.m_kernel, self.m_range)

        out_mask_b = construct_mask(m_mask_b, self.m_kernel, self.m_range)
        im = im_input_b[:, -self.im_channel:, :, :]
        pred_b = construct_image(im, out_mask_b, self.m_kernel, self.m_range)

        seg_f = out_mask_f.sum(1).unsqueeze(1)
        seg_b = out_mask_b.sum(1).unsqueeze(1)
        attn_f = (seg_f + 1e-8) / (seg_f + seg_b + 2e-8)
        attn_b = 1 - attn_f
        pred = pred_f * attn_f + pred_b * attn_b
        return pred, m_mask_f, attn_f, m_mask_b, attn_b

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


def construct_mask(m_mask, m_kernel, m_range):
    out_mask = F.conv2d(m_mask, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
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
