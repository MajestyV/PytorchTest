import os                                    # Windows系统下pytorch和matplotlib同时用需要导入os模块
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 并进行以下设置

import torch
from tqdm import tqdm  # 导入进度条模块
import numpy as np
import matplotlib.pyplot as plt  # 可视化模块

if __name__ == '__main__':
    # f(x) = a*x**2 + b*x + c的最小值

    # 生成数据
    num_data = 100

    # def f(x,k=4.7,b=3.1): return k*x + b
    def f(x,a=2.3,b=4.86,c=3.22): return a*x**2 + b*x + c
    x_hat = np.linspace(-10,10,num_data)
    y_hat = f(x_hat)+np.random.rand(num_data)
    # print(y_hat)
    # exit()

    param = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    # Hypothesis function
    # def h(x): return param[0]*x + param[1]
    def h(x): return param[0]*x**2 + param[1]*x + param[2]


    # optimizer = torch.optim.SGD(params=[param], lr=0.01)  # 使用SGD优化器（废物）
    optimizer = torch.optim.Adam(params=[param], lr=0.01)  # 使用Adam优化器（好）

    num_epochs = 500
    # for i in tqdm(range(1000)):  # 显示进度条
    for epoch in range(num_epochs):  # 不显示进度条
        optimizer.zero_grad()  # 梯度清零，否则会累加
        loss = torch.mean((h(torch.tensor(x_hat)) - torch.tensor(y_hat))**2)
        loss.backward()
        optimizer.step()
        print(loss)

    print(param)

    network_output = h(torch.tensor(x_hat)).detach().numpy()

    plt.plot(x_hat, y_hat, 'ro', label='Original data')
    plt.plot(x_hat, network_output, label='Fitting Line')
    plt.show(block=True)

    # print("y=", f(x).data, ";", "x=", x.data)


    # x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
    # a = torch.tensor(1.0)
    # b = torch.tensor(-2.0)
    # c = torch.tensor(1.0)

    # optimizer = torch.optim.SGD(params=[x], lr=0.01)


    # def f(x):
        # result = a * torch.pow(x, 2) + b * x + c
        # return (result)



    # for i in tqdm(range(500)):
        # optimizer.zero_grad()
        # y = f(x)
        # y.backward()
        # optimizer.step()

    # print("y=", f(x).data, ";", "x=", x.data)

    print('Code finished ! ! !')
    exit()