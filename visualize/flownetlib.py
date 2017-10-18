from __future__ import print_function

import os, sys, numpy as np
import caffe
import tempfile
from math import ceil
from scipy import misc


def calculate_flow(im1=None, im2=None, caffemodel='flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5',
                   deployproto='flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template'):

    if not os.path.exists(caffemodel):
        raise BaseException('caffemodel does not exist: ' + caffemodel)
    if not os.path.exists(deployproto):
        raise BaseException('deploy-proto does not exist: ' + deployproto)

    # im1 = misc.imread('flownet2/viper_example/042_00742.jpg')
    # im2 = misc.imread('flownet2/viper_example/042_00743.jpg')

    num_blobs = 2
    input_data = []
    if len(im1.shape) < 3:
        input_data.append(im1[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(im1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    if len(im2.shape) < 3:
        input_data.append(im2[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(im2[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

    proto = open(deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

    caffe.set_logging_disabled()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, caffemodel, caffe.TEST)

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    print('Network forward pass using %s.' % caffemodel)
    i = 1
    while i<=5:
        i+=1

        net.forward(**input_dict)

        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    flow = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    return flow


if __name__ == '__main__':
    calculate_flow()
