import numpy as np
import mxnet as mx
import torch

N, C, H, W = 2, 3, 8, 8
D = 4
kh, kw = 3, 3
dtype = np.float32 
np_data = np.random.normal(size=(N,C,H,W)).astype(dtype)
np_weight = np.random.normal(size=(D, C, kh, kw)).astype(dtype)
np_bias = np.random.normal(size=(D,)).astype(dtype)

def test_mx_conv():
    data = mx.nd.array(np_data)
    weight = mx.nd.array(np_weight)
    bias = mx.nd.array(np_bias)
    conv = mx.gluon.nn.Conv2D(D, (kh, kw), use_bias=True)
    conv.initialize()
    conv(data)
    conv.weight.data()[:] = weight
    conv.bias.data()[:] = bias
    return conv(data).asnumpy()

def test_th_conv():
    data = torch.tensor(np_data)
    weight = torch.tensor(np_weight)
    bias = torch.tensor(np_bias)
    conv = torch.nn.Conv2d(C, D, (kh, kw), bias=True)
    conv.weight.data = weight
    conv.bias.data = bias
    return conv(data).detach().numpy()

out1 = test_mx_conv()
out2 = test_th_conv()
np.testing.assert_almost_equal(out1,out2, decimal=5)
print(np.abs(out1-out2).max())
print("Okay")
