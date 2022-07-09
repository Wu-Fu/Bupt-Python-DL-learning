import random

import matplotlib_inline.backend_inline
import numpy as np
import torch
from matplotlib import pyplot as plt

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
# 产生一个 2 1000 满足正态分布的随机张量
# features[:,0] features 左侧的1000个元素
# features[:,1] features 右侧的1000个元素
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
# np.random.normal 满足正态分布的随机样本
print(features[0], labels[0])


# 生成散点图
def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
# for X,y in data_iter(batch_size, features, labels):
#     print(X,y)
#     break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型
# 下面是矢量计算表达式的表现
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义线性回归的损失函数
# 下面是最小二乘法
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for eqoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('eqoch %d, loss %f' % (eqoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
