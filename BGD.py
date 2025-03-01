import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

# 导入数据
df = pd.read_csv('Data.csv', index_col=0)
arr = df.values  # Pandas退化为Numpy数组
arr = arr.astype(np.float32)  # 转为float32类型
ts = torch.tensor(arr)  # 转为张量
ts = ts.to('cuda:0')
# print(ts.shape)            759,9       前8列是输入，最后一列是标签

train_size = int(len(ts) * 0.7)
test_size = len(ts) - train_size
ts = ts[torch.randperm(ts.size(0)), :]
train_data = ts[:train_size, :]
test_data = ts[train_size:, :]

class DNN(nn.Module):  # 通常包括init(搭建自己的神经系统)和forward方法，张量可自动计算梯度，无需backward
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.Sigmoid(),
            nn.Linear(32, 8), nn.Sigmoid(),
            nn.Linear(8, 4), nn.Sigmoid(),
            nn.Linear(4, 1), nn.Sigmoid()         #TODO:为什么最后一次还要加上激活函数
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = DNN().to('cuda:0')

loss_fn = nn.BCELoss(reduction='mean')
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 5000
losses = []

X = train_data[:, :-1]
Y = train_data[:, -1].reshape((-1, 1))  # 一阶升级到二阶

for epoch in range(epochs):
    Pred = model(X)
    loss = loss_fn(Pred, Y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

X = test_data[:, :-1]
Y = test_data[:, -1].reshape((-1, 1))
with torch.no_grad():
    Pred = model(X)
    Pred[Pred >= 0.5] = 1
    Pred[Pred < 0.5] = 0
    correct = torch.sum((Pred == Y).all(1))
    total = Y.size(0)
    print(f'测试集精度为:{100 * correct / total}%')
