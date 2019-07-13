import numpy as np
import mxnet as mx
import torch
print(mx, torch)

N, C = 2, 3
D = 4
dtype = np.float32

np.random.seed(39)
np_data = np.random.normal(size=(N,C)).astype(dtype)
np_weight = np.random.normal(size=(D, C)).astype(dtype)
np_bias = np.random.normal(size=(D, )).astype(dtype)
np_target = np.random.normal(size=(N, D)).astype(dtype)
epoch = 3
lr = 1e-3
wd = 1e-2
momentum = 0.9

def test_np():
    data = np_data.copy()
    weight = np_weight.copy()
    bias = np_bias.copy()
    target = np_target.copy()
    state_weight = None
    state_bias = None
    losses = []
    datas = []
    grads = []
    def fc(data):
        return np.dot(data, weight.T) + bias
    for e in range(epoch):
        out = fc(data)
        loss = ((out - target) ** 2).sum()
        dy = 2 * (out - target)
        grad_weight = np.dot(dy.T, data)
        grad_bias = dy.sum(0).T

        if state_weight is None:
            state_weight = np.zeros_like(weight)
            state_bias = np.zeros_like(bias)
        rescale_grad_weight = lr * (grad_weight + wd * weight)
        rescale_grad_bias = lr * (grad_bias + wd * bias)
        state_weight = momentum * state_weight + rescale_grad_weight
        state_bias = momentum * state_bias + rescale_grad_bias

        datas.append((weight, bias))
        grads.append((grad_weight, grad_bias))
        losses.append(loss)

        # update weight
        weight = weight - state_weight
        bias = bias - state_bias
    return dict(losses=losses, datas=datas, grads=grads)

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
    losses = []
    datas = []
    grads = []
    trainer = mx.gluon.Trainer(
        fc.collect_params(),
        'sgd',
        dict(
            learning_rate=lr,
            wd=wd,
            momentum=momentum
        )
    )
    for e in range(epoch):
        with mx.autograd.record():
            loss = (fc(mx_data) - mx_target).square().sum()
        loss.backward()
        datas.append((fc.weight.data().asnumpy(), fc.bias.data().asnumpy()))
        grads.append((fc.weight.data().grad.asnumpy(), fc.bias.data().grad.asnumpy()))
        trainer.step(1)
        losses.append(loss.asnumpy())
    return dict(losses=losses, datas=datas, grads=grads)

def test_th():
    th_data = torch.tensor(np_data)
    th_weight = torch.tensor(np_weight)
    th_bias = torch.tensor(np_bias)
    th_target = torch.tensor(np_target)
    fc = torch.nn.Linear(C, D, bias=True)
    fc.weight.data = th_weight
    fc.bias.data = th_bias
    optimizer = torch.optim.SGD(
        fc.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )
    losses = []
    datas = []
    grads = []
    for e in range(epoch):
        optimizer.zero_grad()
        loss = ((fc(th_data) - th_target) ** 2).sum()
        loss.backward()
        datas.append((fc.weight.detach().numpy().copy(), fc.bias.detach().numpy().copy()))
        grads.append((fc.weight.grad.numpy().copy(), fc.bias.grad.numpy().copy()))
        optimizer.step()
        losses.append(loss.detach().numpy().copy())
    return dict(losses=losses, datas=datas, grads=grads)
    
def test_result(xs, ys, prefix=None):
    if prefix is None:
        prefix = 'out'
    if isinstance(xs, (list, tuple)):
        assert len(xs) == len(ys)
        for i, (x, y) in enumerate(zip(xs, ys)):
            test_result(x, y, prefix+'[{}]'.format(i))
    elif isinstance(xs, dict):
        for k in xs.keys():
            test_result(xs[k], ys[k], prefix+"['{}']".format(k))
    else:
        try:
            #assert xs.shape == ys.shape, (xs.shape, ys.shape)
            np.testing.assert_almost_equal(xs, ys, decimal=5)
        except Exception as e:
            print('========{}=========='.format(prefix))
            raise e


out0 = test_np()
out1 = test_mx()
out2 = test_th()
test_result(out0, out1)
test_result(out0, out2)
#test_result(out1, out2)
print("Okay")
