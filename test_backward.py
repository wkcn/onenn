import numpy as np
import mxnet as mx
import torch

def test_mx_backward(data, data2):
    data = mx.nd.array(data)
    data2 = mx.nd.array(data2)
    data.attach_grad()
    with mx.autograd.record():
        loss = (data * data2).sum()
    loss.backward()
    return data.grad.asnumpy()

def test_th_backward(data, data2):
    data = torch.tensor(data)
    data2 = torch.tensor(data2)
    data.requires_grad = True
    loss = (data * data2).sum()
    loss.backward()
    return data.grad.numpy()



N, C, H, W = 2, 3, 4, 4
data = np.random.normal(size=(N,C,H,W))
data2 = np.random.normal(size=(N,C,H,W))
g1 = (test_mx_backward(data, data2))
g2 = (test_th_backward(data, data2))
np.testing.assert_almost_equal(g1, g2)
print(np.abs(g1-g2).max())
print("Okay")
