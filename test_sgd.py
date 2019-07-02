import numpy as np
import mxnet as mx
import torch

N, C = 2, 16
D = 8
dtype = np.float32

np_data = np.random.normal(size=(N,C)).astype(dtype)
np_weight = np.random.normal(size=(D, C)).astype(dtype)
np_bias = np.random.normal(size=(D, )).astype(dtype)
np_target = np.random.normal(size=(N, D)).astype(dtype)
epoch = 5

def test_mx():
    mx_data = mx.nd.array(np_data)
    mx_weight = mx.nd.array(np_weight)
    mx_bias = mx.nd.array(np_bias)
    mx_target = mx.nd.array(np_target)
    fc = mx.gluon.nn.Dense(D, use_bias=True)
    fc.initialize()
    fc(mx_data)
    fc.weight.data()[:] = mx_weight
    fc.bias.data()[:] = mx_bias
    datas = []
    grads = []
    for e in range(epoch):
        with mx.autograd.record():
            loss = (fc(mx_data) - mx_target).square().sum()
            loss.backward()
            datas.append((fc.weight.data().asnumpy(), fc.bias.data().asnumpy()))
            grads.append((fc.weight.data().grad.asnumpy(), fc.bias.data().grad.asnumpy()))
    return datas, grads

def test_th():
    th_data = torch.tensor(np_data)
    th_weight = torch.tensor(np_weight)
    th_bias = torch.tensor(np_bias)
    th_target = torch.tensor(np_target)
    fc = torch.nn.Linear(C, D, bias=True)
    with torch.no_grad():
        fc.weight[:] = th_weight
        fc.bias[:] = th_bias
    datas = []
    grads = []
    for e in range(epoch):
        loss = ((fc(th_data) - th_target) ** 2).sum()
        fc.zero_grad()
        loss.backward()
        datas.append((fc.weight.detach().numpy(), fc.bias.detach().numpy()))
        grads.append((fc.weight.grad.numpy(), fc.bias.grad.numpy()))
    return datas, grads
    
def test_result(xs, ys):
    if isinstance(xs, (list, tuple)):
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            test_result(x, y)
    else:
        assert xs.shape == ys.shape, (xs.shape, ys.shape)
        np.testing.assert_almost_equal(xs, ys, decimal=5)


out1 = test_mx()
out2 = test_th()
test_result(out1[0], out2[0])
test_result(out1[1], out2[1])
print("Okay")
