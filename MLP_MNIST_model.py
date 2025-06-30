#!/usr/bin/env python
# encoding: utf-8

# 导入必要的库 / Import necessary libraries
import torch
import numpy as np
from torchvision.datasets import mnist  # MNIST数据集 / MNIST dataset
from torch import nn  # 神经网络模块 / Neural network module
from torch.autograd import Variable  # 自动求导变量 / Autograd variable
import torch.nn.functional as F  # 神经网络函数库 / Functional API for neural networks
from torch.utils.data import DataLoader  # 数据加载器 / Data loader
import pandas as pd
import pdb
import os

# 解决SSL证书验证问题，避免下载MNIST数据集时报错，如SSL不安全的问题
# Fix SSL certificate issues when downloading the dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 定义数据预处理函数 / Define data transformation function
def data_transform(x):
    """将输入图像数据转换为模型可处理的格式
    Convert input image to a format suitable for the model"""
    x = np.array(x, dtype='float32') / 255  # 归一化到[0,1]范围 / Normalize to [0,1]
    x = x.reshape((-1,))  # 展平图像(28x28 -> 784) / Flatten image
    x = torch.from_numpy(x)  # 转换为PyTorch张量 / Convert to PyTorch tensor
    return x

# 加载MNIST数据集 / Load MNIST dataset
trainset = mnist.MNIST(
    './datasets/mnist',  # 数据集保存路径 / Path to save dataset
    train=True,          # 加载训练集 / Load training set
    transform=data_transform,  # 应用预处理 / Apply transform
    download=True        # 如不存在则下载 / Download if not present
)
testset = mnist.MNIST(
    './datasets/mnist', 
    train=False,         # 加载测试集 / Load test set
    transform=data_transform, 
    download=True
)

# 创建数据加载器 / Create data loaders
train_data = DataLoader(trainset, batch_size=64, shuffle=True)  # 训练集批大小64，打乱 / Train loader
test_data = DataLoader(testset, batch_size=128, shuffle=False)  # 测试集批大小128，不打乱 / Test loader

# 定义多层感知机模型 / Define MLP model
class MLP(nn.Module):
    """MNIST分类器 / MNIST digit classifier"""
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)  # 输入层 -> 隐藏层1 / Input to hidden layer 1
        self.fc2 = nn.Linear(500, 250)      # 隐藏层1 -> 隐藏层2 / Hidden 1 to hidden 2
        self.fc3 = nn.Linear(250, 125)      # 隐藏层2 -> 隐藏层3 / Hidden 2 to hidden 3
        self.fc4 = nn.Linear(125, 10)       # 隐藏层3 -> 输出层 / Hidden 3 to output

    def forward(self, x):
        """前向传播 / Forward pass"""
        x = x.view(-1, 28 * 28)        # 展平输入图像 / Flatten input
        x = F.relu(self.fc1(x))       # 隐藏层1 + ReLU激活 / Hidden 1 + ReLU
        x = F.relu(self.fc2(x))       # 隐藏层2 + ReLU激活 / Hidden 2 + ReLU
        x = F.relu(self.fc3(x))       # 隐藏层3 + ReLU激活 / Hidden 3 + ReLU
        x = self.fc4(x)               # 输出层（无激活）/ Output layer (no activation)
        return x

# 初始化模型 / Initialize model
mlp = MLP()

# 定义损失函数（交叉熵）/ Define loss function (CrossEntropy)
criterion = nn.CrossEntropyLoss()

# 初始化记录列表 / Initialize tracking lists
losses = []       # 训练损失 / Training loss
acces = []        # 训练准确率 / Training accuracy
eval_losses = []  # 验证损失 / Evaluation loss
eval_acces = []   # 验证准确率 / Evaluation accuracy

model_path = ("./saved_model/MLP_MNIST.pkl")
pre_model_exist = os.path.isfile(model_path)

if pre_model_exist:
    print('load model parameters ...')
    mlp.load_state_dict(torch.load(model_path))
else:
    print('No Well-Trained Model!')


# 训练循环，共10轮 / Training loop for 10 epochs
for e in range(10):
    # 动态调整学习率 / Adjust learning rate based on epoch
    if e < 7:
        optimizer = torch.optim.Adam(mlp.parameters(), 1e-3)  # 前7轮：较大学习率 / First 7 epochs
    else:
        optimizer = torch.optim.Adam(mlp.parameters(), 1e-4)  # 后3轮：较小学习率 / Last 3 epochs
    
    train_loss = 0  # 累积训练损失 / Accumulated train loss
    train_acc = 0   # 累积训练准确率 / Accumulated train accuracy
    
    mlp.train()  # 模型设为训练模式 / Set model to train mode
    
    # 遍历训练数据 / Loop through training data
    for im, label in train_data:
        im = Variable(im)       # 转换为Variable / Convert to Variable
        label = Variable(label)
        
        out = mlp(im)           # 前向传播 / Forward pass
        loss = criterion(out, label)  # 计算损失 / Compute loss
        
        optimizer.zero_grad()   # 清空梯度 / Zero gradients
        loss.backward()         # 反向传播 / Backpropagation
        optimizer.step()        # 更新参数 / Update weights
        
        train_loss += loss.item()  # 累计损失 / Accumulate loss
        _, pred = out.max(1)       # 获取预测结果 / Get predictions
        num_correct = (pred == label).sum().item()  # 正确数 / Count correct predictions
        acc = num_correct / im.shape[0]             # 准确率 / Accuracy
        train_acc += acc
    
    # 记录本轮平均损失和准确率 / Save average loss and accuracy
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    
    eval_loss = 0  # 验证损失 / Eval loss
    eval_acc = 0   # 验证准确率 / Eval accuracy
    mlp.eval()     # 模型设为评估模式 / Set model to eval mode
    
    # 遍历测试数据 / Loop through test data
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = mlp(im)
        loss = criterion(out, label)
        
        eval_loss += loss.item()  # 累计验证损失 / Accumulate eval loss
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
    
    # 记录验证集平均指标 / Save eval metrics
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    
    # 打印当前轮训练与验证指标 / Print current epoch metrics
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

# 保存模型参数 / Save model weights
torch.save(mlp.state_dict(), 'saved_model/MLP_MNIST.pkl')  # 仅保存模型参数 / Save model parameters only

# 保存准确率和损失到CSV文件（默认注释）/ Save accuracy & loss to CSV (commented by default)
file = './results/MLP_MNIST_model/acc.csv'
data = pd.DataFrame(eval_acces)
data.to_csv(file, index=False)

file = './results/MLP_MNIST_model/loss.csv'
data = pd.DataFrame(eval_losses)
data.to_csv(file, index=False)
