# 使用小批量梯度下降时，需要用到三个pytorch内置工具
# DataSet用于封装数据集
# DataLoader用于加载数据不同的批次
# random_split用于划分数据集和测试集

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

# 制作数据集，需要继承Dataset，重写init(加载数据集)，getitem(获取数据索引)，len(获取数据总量)三个方法
class MyData(Dataset):
    filepath = r'/DNN/MBGD/Data.csv'

    def __init__(self, filepath):
        df = pd.read_csv(filepath, index_col=0)
        arr = df.values
        arr = arr.astype(np.float32)
        ts = torch.tensor(arr)
        ts = ts.to('cuda')
        self.X = ts[:, :-1]  # 输入输出特征的划分必须写在这里！
        self.Y = ts[:, -1].reshape((-1, 1))
        self.len = ts.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


Data = MyData('Data.csv')
train_size = int(len(Data) * 0.7)
test_size = len(Data) - train_size
train_Data, test_Data = random_split(Data, [train_size, test_size])

# 批次加载器
train_loader = DataLoader(dataset=train_Data, shuffle=True, batch_size=128)
test_loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=64)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.Sigmoid(),
            nn.Linear(32, 8), nn.Sigmoid(),
            nn.Linear(8, 4), nn.Sigmoid(),
            nn.Linear(4, 1), nn.Sigmoid()
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = DNN().to('cuda:0')        #TODO:为什么有时是cuda，有时是cuda:0

loss_fn = nn.BCELoss(reduction='mean')
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 500
losses = []
for epoch in range(epochs):
    for (x, y) in train_loader:
        Pred = model(x)
        loss = loss_fn(Pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Fig = plt.figure()
plt.plot(range(len(losses)), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

correct = 0
total = 0

with torch.no_grad():
    for (x, y) in test_loader:
        Pred = model(x)
        Pred[Pred >= 0.5] = 1
        Pred[Pred < 0.5] = 0
        correct = torch.sum((Pred == y).all(1))
        total = y.size(0)

print(f'测试集精度为:{100 * correct / total}%')