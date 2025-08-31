"""
LSTM分类模型训练与保存

"""
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import pandas as pd
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import xlrd
import os
from torch.optim import lr_scheduler
import random


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=40, num_layers=5, output_size=4):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度 看你是几个指数
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数  也是分类类数
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_directions = 1  # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.num_layers, batch_first=True, bidirectional=False)  # 调用lstm
        self.linear = nn.Linear(hidden_layer_size, output_size)  # 调用全连接

    # 调用激活函数多分类激活函数
    def forward(self, input_x):
        # batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        # input_x = input_x.view(len(input_x), 1, -1)  # view函数是改变维度，input_x输入
        h0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)   # 隐状态
        c0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)   # 传输带conyor belt
        # hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
        #                torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, (h0, c0))  # 两个输入是 h0 和 c0，可以理解成网络的初始化参数
        linear_out = self.linear(lstm_out[:, -1, :])   # 使用全连接层linear, 取最后一个时间步的输出
        predictions = torch.softmax(linear_out, dim=1)  # 使用sigmoid函数   这里可以看看需不需要改，好像多分类可以用softmax
        # predictions = predictions[:, -1, :]
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # # output(batch_size, seq_len, num_directions * hidden_size)
        # output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        # pred = self.linear(output)  # (5, 30, 1)
        # pred = pred[:, -1, :]  # (5, 1)

        return predictions


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(r"D:\edge_download\training_pointsexcel1.xlsx")  # 读取样本数据记得样本要按读取训练的数据那样输入
    Y_train_2020 = df.iloc[:, 0]   # 训练标签列
    X_train_2020 = df.iloc[:, 1:]  # 训练样本列
    name = df.columns[1:]  # 第一列是类别标签

    # Y_train_2020 = Y_train_2020.values - 1 # 标签类别值要从0开始 这里看看你的标签是不是0开始。是1开始就用这一行
    Y_train_2020 = Y_train_2020.values# 标签类别值要从0开始 这里看看你的标签是不是0开始。是0开始就用这一行
    X_train_2020 = X_train_2020.values
    # rgbnir = X_train_2020[:, 0:4]
    # rgb = X_train_2020[:, 0:5]
    # zhishu = X_train_2020[:, 5:10]
    # 不同列对应的特征
    vv = X_train_2020[:, 0:24]
    vh = X_train_2020[:, 24:48]
    SAR_Sum = X_train_2020[:, 48:72]
    SAR_Diff = X_train_2020[:, 72:96]
    SAR_NDVI = X_train_2020[:, 96:120]


    x_2020 = np.zeros((vh.shape[0], vh.shape[1], 5))  #
    # x_2020 = np.zeros((vh.shape[0], vh.shape[1], 2))
    # x_2020[:, :, 0] = rgb
    # x_2020[:, :, 1] = zhishu
    x_2020[:, :, 0] = vv
    x_2020[:, :, 1] = vh
    x_2020[:, :, 2] = SAR_Sum
    x_2020[:, :, 3] = SAR_Diff
    x_2020[:, :, 4] = SAR_NDVI

    # 划分样本集测试集占总样本0.3，stratify=y按照标签等比例划分
    num_total = x_2020.shape[0]
    num_train = int(0.75 * num_total)  # 其中训练样本包括验证样本
    num_test = num_total - num_train
    Arange = list(range(num_total))
    random.shuffle(Arange)  # 打乱顺序，分配训练样本、验证和测试样本
    train_list = []
    for i in range(num_train):
        idx = Arange.pop()
        train_list.append(idx)

    x_train = x_2020[train_list, :, :]
    x_test = x_2020[Arange, :, :]
    y_train = Y_train_2020[train_list]
    # y_train = np.stack([y_train1,y_train1],axis=1)
    y_test = Y_train_2020[Arange]
    # y_test = np.stack([y_test1,y_test1],axis=1)

    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)

    x_train = torch.Tensor(np.array(x_train)).type(torch.FloatTensor)
    y_train = torch.Tensor(np.array(y_train)).type(torch.int64)
    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=x_train.shape[0],  # 每块的大小   加载的每批次样本数据大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    x_test = torch.Tensor(np.array(x_test)).type(torch.FloatTensor)
    y_test = torch.Tensor(np.array(y_test)).type(torch.int64)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=x_test.shape[0],  # 每块的大小   加载的每批次样本数据大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    # 初始化模型
    # 建模三件套：loss，优化，epochs
    model = LSTMClassifier()  # 模型
    model.to('cuda')

    loss_fn1 = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器


    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.1)
    # 定义学习率衰减因子，学习率每7步衰减为原来的0.1倍,一般step_size是迭代次数的三分之一或者五分之一

    # 定义训练函数
    def fit(epoch, model, train_loader, test_loader):
        correct = 0
        running_loss = 0

        model.train()
        for x, y in tqdm(train_loader):
            if torch.cuda.is_available():
                x, y, = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn1(y_pred, y)
            # loss = loss_fn1(y_pred, y) + loss_fn2(y_b, z) + 0.4 * loss_fn1(y_aux,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # y_pred = torch.argmax(y_pred,dim=1)
                # correct += (y_pred==y).sum().item()
                running_loss += loss.item()

        # exp_lr_scheduler.step()
        epoch_loss = running_loss

        test_correct = 0
        test_running_loss = 0

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x)
                loss = loss_fn1(y_pred, y)

                # y_pred = torch.argmax(y_pred, dim=1)
                # test_correct += (y_pred == y).sum().item()
                test_running_loss += loss.item()

        epoch_test_loss = test_running_loss

        print(
            'epoch:', epoch + 1,
            '\n',
            'train_loss:', round(epoch_loss, 5),
            'test_loss', round(epoch_test_loss, 5),
        )
        return epoch_loss, epoch_test_loss


    epochs = 5000#训练的伦次
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        epoch_loss, epoch_test_loss = fit(epoch, model, train_loader, test_loader)
        train_loss.append(round(epoch_loss, 5))
        test_loss.append(round(epoch_test_loss, 5))

        PATH = r"D:\edge_download\rd_lstm_vhvv15_10k.pth"  # 保存模型
        torch.save(model.state_dict(), PATH)

    x = range(0, epochs)
    plt_save_path = r'D:\edge_download\rd_lstm_vhvv15_10k'

    if not os.path.exists(plt_save_path):  # 如果路径不存在
        os.makedirs(plt_save_path)  # 创建路径

    plt.figure(figsize=(4, 4))
    plt.plot(x, train_loss, 'ro-', label='Train loss', linewidth=1, markersize=3)
    plt.plot(x, test_loss, 'bs-', label='Test loss', linewidth=1, markersize=3)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(plt_save_path + '\\' + 'loss_1')
    plt.show()

