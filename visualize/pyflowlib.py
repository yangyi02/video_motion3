import numpy
import pyflow.pyflow as pyflow


def calculate_flow(im1, im2, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=0):
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = numpy.concatenate((u[..., None], v[..., None]), axis=2)
    return flow
