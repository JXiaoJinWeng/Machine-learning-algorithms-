import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 展示高清图
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

# 生成数据集

X1 = torch.rand(10000, 1)  # 0~1区域
X2 = torch.rand(10000, 1)
X3 = torch.rand(10000, 1)  # 此时三者shape都是10000,1

# 采取独热编码(One-Hot)，将每个类别映射为0,1，避免错误的关系假设
Y1 = ((X1 + X2 + X3) < 1).float()
Y2 = ((1 < (X1 + X2 + X3)) & ((X1 + X2 + X3) < 2)).float()
Y3 = ((X1 + X2 + X3) > 2).float()

Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集
Data = Data.to('cuda:0')  # 搬到cuda上
# print(Data.shape)

# 划分训练集和测试集

train_size = int(len(Data) * 0.7)
test_size = len(Data) - train_size
Data = Data[torch.randperm(Data.size(0)), :]
train_data = Data[:train_size, :]
test_data = Data[train_size:, :]


# print(train_data.shape,test_data.shape)

# 搭建神经网络(继承nn.Module父类)

class DNN(nn.Module):  # 通常包括init(搭建自己的神经系统)和forward方法，张量可自动计算梯度，无需backward
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(  # 按顺序搭建各层
            nn.Linear(3, 5), nn.ReLU(),  # 第一层：前一层(输入层)节点为3，本层节点为5，全连接层
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 3)  # 二者开头结尾都是3，必须与输入输出特征的数量一致
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = DNN().to('cuda:0')

# 查看参数(均为初始随机参数)
# for name, param in model.named_parameters():
#    print(f"参数:{name}\n形状:{param.shape}\n数值:{param}\n")

# 查看超参数https://pytorch.org/docs/1.12/nn.html可以查看层，激活函数，损失函数，学习率与优化算法
loss_fn = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)     #TODO:为什么改成Adam准确率为0
# 训练网络
epochs = 1000
losses = []

X = train_data[:, :3]
Y = train_data[:, -3:]

for epoch in range(epochs):
    Pred = model(X)  # 一次前向传播
    loss = loss_fn(Pred, Y)  # 计算损失函数(张量)
    losses.append(loss.item())  # .item()降级为一个元素，记录损失函数的变化
    optimizer.zero_grad()  # 清除上一轮滞留的梯度
    loss.backward()  # 一次反向传播
    optimizer.step()  # 优化内部参数

# 绘图
Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# 测试网络
X = test_data[:, :3]
Y = test_data[:, -3:]
with torch.no_grad():  # 局部关闭梯度计算(测试不需要)
    Pred = model(X)
    Pred[:, torch.argmax(Pred, axis=1)] = 1  # 每行最大的数字变成1，其余为0
    Pred[Pred != 1] = 0
    correct = torch.sum((Pred == Y).all(1))
    # Pred==Y返回3000，3的bool型张量
    # Pred==Y).all(1)返回3000,1    1为按行扫描，一行均为true结果才为true
    total = Y.size(0)
    print(f'测试集精度为:{100 * correct / total}%')


# 保存网络
# save_path = r'E:\Software\NeuronNetwork\demo.pth'
# torch.save(model, save_path)
# 导入网络用
# load_path = r'E:\Software\NeuronNetwork\demo.pth'
# new_model = torch.load(load_path)
#
# X = test_data[:, :3]
# Y = test_data[:, -3:]
# with torch.no_grad():
#     Pred = new_model(X)
#     Pred[:, torch.argmax(Pred, axis=1)] = 1
#     Pred[Pred != 1] = 0
#     correct = torch.sum((Pred == Y).all(1))
#     total = Y.size(0)
#     print(f'测试集精度为:{100 * correct / total}%')


