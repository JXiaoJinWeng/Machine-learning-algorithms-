import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

# 制作数据集    28*28=784
# ToTensor将图像数据转成张量，C*W*H，C为通道数，二维灰度图像是1，RGB彩图为3
# Normalize将输入数据转为标准正态分布，MNIST数据中均值为0.1307，标准差为0.3081

transform = transforms.Compose([  # 设定下载参数
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])
train_Data = datasets.MNIST(
    root=r'D:\Project\Pycharm Project\DNN\HandWritten_Number_Recognition',
    train=True,
    download=True,
    transform=transform
)
test_Data = datasets.MNIST(
    root=r'D:\Project\Pycharm Project\DNN\HandWritten_Number_Recognition',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(dataset=train_Data, shuffle=True, batch_size=64)
test_loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=64)


class DNN(nn.Module):
    def __init__(self):
        '''搭建各层神经网络'''
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # 二维铺平成一维
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        '''前向传播'''
        y = self.net(x)
        return y


model = DNN().to('cuda:0')

loss_fn = nn.CrossEntropyLoss()  # 自带一个softmax激活函数  TODO:什么是softmax
learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.5  # 动量，使梯度下降算法有了力和惯性
)

epochs = 5
losses = []

for epoch in range(epochs):
    for (x, y) in train_loader:
        x, y = x.to('cuda:0'), y.to('cuda:0')  # 把小批次搬到GPU上
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
        x, y = x.to('cuda:0'), y.to('cuda:0')  # 把小批次搬到GPU上
        Pred = model(x)
        _, predicted = torch.max(Pred.data, dim=1)
        # a,b = ...是说找到Pred每一行的最大值，数值赋给a，位置赋给b，
        # 就相当于把独热编码转成普通阿拉伯数字，这样就可以直接与y做比较
        # predicted和y是一阶张量，所以不用.all(1)
        correct += torch.sum((predicted == y))
        total += y.size(0)

print(f'测试集精度为:{100 * correct / total}%')