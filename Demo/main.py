import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # 初始化
    def __init__(self):
        # 官方定义继承
        super(Net, self).__init__()
        # 定义全连接神经层
        self.fc1 = nn.Linear(3, 1)
        #self.fc2 = nn.Linear(3, 1)
        # 定义l损失率和优化器
        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
    # 定义神经网络
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

    def getRandomData(self, dataNum):
        inData = torch.rand(dataNum, 3) * 40 + 60
        outData = inData.sum(1)
        return (inData, outData)

    def train(self, trainNum, data):
        (inData, outData) = data
        for i in range(trainNum):
            self.optimizer.zero_grad()
            output = self(inData)
            l = self.loss(output, outData)
            l.backward()
            self.optimizer.step()

            if (i % 100 == 0):
                print("Epoch: {}, loss = {}".format(i, l.item()))

if __name__ == "__main__":
    net = Net()
    net.train(1000, net.getRandomData(5000))