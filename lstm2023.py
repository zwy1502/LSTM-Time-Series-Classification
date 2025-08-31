
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, classification_report

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=False):
        """
        初始化早停机制
        :param patience: 当验证损失不再改善时，最多允许的epoch数
        :param min_delta: 损失改善的最小变化，低于此值则认为没有改善
        :param verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        """
        检查验证损失是否有改善
        :param val_loss: 当前epoch的验证损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"验证损失连续{self.counter}个epoch没有改善")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
            if self.verbose:
                print(f"验证损失改善到 {self.best_loss}. 计数器重置.")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, num_layers=3, output_size=5, dropout_rate=0.5):#类别数5
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
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_layer_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout_rate
                            )  # 调用lstm

        # 添加一个Dropout层在LSTM输出和全连接层之间
        self.dropout = nn.Dropout(dropout_rate)  # 新增的Dropout层

        self.linear = nn.Linear(hidden_layer_size, output_size)  # 调用全连接

    # 调用激活函数多分类激活函数
    def forward(self, input_x):
        h0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)   # 隐状态
        c0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)   # 传输带conyor belt
        lstm_out, (h_n, h_c) = self.lstm(input_x, (h0, c0))  # 两个输入是 h0 和 c0，可以理解成网络的初始化参数
        lstm_out = self.dropout(lstm_out[:, -1, :])  # 在全连接层前应用Dropout
        linear_out = self.linear(lstm_out)   # 使用全连接层linear, 取最后一个时间步的输出
        # predictions = torch.softmax(linear_out, dim=1)  # 使用sigmoid函数   这里可以看看需不需要改，好像多分类可以用softmax
        # return predictions
        return linear_out


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S1_2023时序训练数据月度数据清洗后.xlsx")  # 读取样本数据记得样本要按读取训练的数据那样输入，原始是r"D:\edge_download\training_pointsexcel1.xlsx"
    Y_train_2020 = df.iloc[:, 0]   # 训练标签列
    X_train_2020 = df.iloc[:, 1:]  # 训练样本列
    name = df.columns[1:]  # 第一列是类别标签

    Y_train_2020 = Y_train_2020.values - 1 # 标签类别值要从0开始 这里看看你的标签是不是0开始。是1开始就用这一行
    # Y_train_2020 = Y_train_2020.values# 标签类别值要从0开始 这里看看你的标签是不是0开始。是0开始就用这一行
    X_train_2020 = X_train_2020.values
    # rgbnir = X_train_2020[:, 0:4]
    # rgb = X_train_2020[:, 0:5]
    # zhishu = X_train_2020[:, 5:10]
    # 不同列对应的特征
    # vv = X_train_2020[:, 0:23]
    # vh = X_train_2020[:, 23:46]
    # SAR_Sum = X_train_2020[:, 46:69]
    # SAR_Diff = X_train_2020[:, 69:92]
    # SAR_NDVI = X_train_2020[:, 92:115]

    vv = X_train_2020[:, 0:11]
    vh = X_train_2020[:, 11:22]
    SAR_Sum = X_train_2020[:, 22:33]
    SAR_Diff = X_train_2020[:, 33:44]
    SAR_NDVI = X_train_2020[:, 44:55]

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
    num_train = int(0.8 * num_total)  # 其中训练样本包括验证样本
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
        batch_size=64,  # 每块的大小   加载的每批次样本数据大小，以前这个是x_train.shape[0]修改
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    x_test = torch.Tensor(np.array(x_test)).type(torch.FloatTensor)
    y_test = torch.Tensor(np.array(y_test)).type(torch.int64)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=64,  # 每块的大小   加载的每批次样本数据大小，以前这个是x_train.shape[0]修改
        shuffle=False,  # 要不要打乱数据 (打乱比较好)#修改这里我举得测试不打乱
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    # 初始化模型
    # 建模三件套：loss，优化，epochs
    model = LSTMClassifier()  # 模型
    model.to('cuda')

    loss_fn1 = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)#添加权重衰减（L2正则化）weight_decay

    exp_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True,min_lr=1e-6)
    #mode='min':表示当监控的指标（在这里是验证损失val_loss）不再减小时，调整学习率。
    # factor=0.1:表示当验证损失在设定的patience周期内没有改善时，将学习率减少为原来的10%。
    # 例如，如果当前学习率是0.001，则调整后的学习率将是0.0001。min_lr=1e-6最小学习率是0.000001
    # patience = 10:表示在验证损失未改善的情况下等待多少个epoch后再调整学习率。如果验证损失在10个连续的epoch中没有改善，则调整学习率。


    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.1)#加了学习率衰减（修改）
    # 定义学习率衰减因子，学习率每7步衰减为原来的0.1倍,一般step_size是迭代次数的三分之一或者五分之一

    # 定义训练函数
    def fit(epoch, model, train_loader, test_loader):
        model.train()
        correct = 0
        total = 0#加了这行修改
        running_loss = 0
        current_lr = optimizer.param_groups[0]['lr']#加了这行修改
        print(f"学习率: {current_lr}")#加了这行修改

        for x, y in train_loader:
            if torch.cuda.is_available():
                x, y, = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn1(y_pred, y)
            # loss = loss_fn1(y_pred, y) + loss_fn2(y_b, z) + 0.4 * loss_fn1(y_aux,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.argmax(y_pred,dim=1)#加了这行修改
                correct += (y_pred==y).sum().item()#加了这行修改
                total += y.size(0)#加了这行修改
                running_loss += loss.item()
        train_accuracy = 100 * correct / total#加了这行修改
        epoch_loss = running_loss / len(train_loader)#加了这行修改
        # exp_lr_scheduler.step()#加了学习率衰减（修改）
        # epoch_loss = running_loss#这行修改
        # 在每个 epoch 结束时再次打印学习率
        current_lr = optimizer.param_groups[0]['lr']#加了这行修改
        # print("=" * 50)
        print(f"经过ReduceLROnPlateau调度器调整后的学习率： {current_lr}")#加了这行修改
        # print("=" * 50)


        test_correct = 0
        test_total = 0# 添加这行
        test_running_loss = 0

        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x)
                loss = loss_fn1(y_pred, y)

                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)  # 添加这行
                test_running_loss += loss.item()
        test_accuracy = 100 * test_correct / test_total
        epoch_test_loss = test_running_loss / len(test_loader)  # 修改这行

        # 准备表格数据
        table = [
            ['Epoch', 'Train Loss','Train Accuracy', 'Validation Loss','Validation Accuracy'],
            [epoch + 1, round(epoch_loss, 5), round(train_accuracy, 4),
             round(epoch_test_loss, 5),
             round(test_accuracy, 4)],
        ]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=False))

        # print(
        #     'epoch:', epoch + 1,
        #     '\n',
        #     'train_loss:', round(epoch_loss, 5),
        #     'train_accuracy:', round(train_accuracy, 4),
        #     'test_loss', round(epoch_test_loss, 5),
        #     'test_accuracy:', round(test_accuracy, 4),
        # )
        #
        return epoch_loss, epoch_test_loss, train_accuracy, test_accuracy# 修改返回值


    best_test_loss = float('inf')  # 初始化最佳损失为正无穷大#加两行
    best_model_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S12023best_model月度.pth"  # 保存最佳模型的路径#加两行

    epochs = 5000#训练的伦次
    train_loss = []
    test_loss = []
    train_acc = []  # 添加这行
    test_acc = []  # 添加这行
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=True)
    #早停10（patience）个epoch没有改善就停止训练，注意ReduceLROnPlateau的patience要小于早停的patience
    # 改善的损失减少是阈值是0.001（min_delta）verbose是打印早停信息
    for epoch in range(epochs):
        epoch_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc = fit(epoch, model, train_loader, test_loader)#修改
        train_loss.append(round(epoch_loss, 5))
        test_loss.append(round(epoch_test_loss, 5))
        train_acc.append(round(epoch_train_acc, 2))  # 添加这行
        test_acc.append(round(epoch_test_acc, 2))  # 添加这行

        # PATH = r"D:\edge_download\rd_lstm_vhvv15_10k.pth"  # 保存模型
        # torch.save(model.state_dict(), PATH)
        # 检查当前验证损失是否为最小值
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            torch.save(model.state_dict(), best_model_path)  # 保存当前模型参数
            print(f"保存了新的最佳模型，验证损失: {best_test_loss}")

        # 使用早停机制检查验证损失是否有改善
        early_stopping(epoch_test_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练。")
            break

        # 在每个epoch结束时调用学习率调度器
        exp_lr_scheduler.step(epoch_test_loss)
        # 定期保存模型（可选）
        # if (epoch + 1) % 500 == 0:
        #     PATH =r"D:\edge_download\rd_lstm_epoch_{epoch + 1}.pth"
        #     torch.save(model.state_dict(), PATH)

    # 保存最终模型（作为备份）
    # final_model_path =r"D:\edge_download\final_model.pth"
    # torch.save(model.state_dict(), final_model_path)

    x = range(0, epochs)
    plt_save_path = r'D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\rd_lstmS12023月度'

    if not os.path.exists(plt_save_path):  # 如果路径不存在
        os.makedirs(plt_save_path)  # 创建路径

    # plt.figure(figsize=(15, 8))#改了这个
    # plt.grid(True)#加了这个
    # plt.plot(x, train_loss, 'ro-', label='Train loss', linewidth=1, markersize=3)
    # plt.plot(x, test_loss, 'bs-', label='Test loss', linewidth=1, markersize=3)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(loc='upper right')
    # plt.savefig(plt_save_path + '\\' + 'loss_1')
    # plt.show()

    x = range(0, len(train_loss))  # 使用实际训练步数而不是固定的100

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(x, train_loss, 'ro-', label='Train loss', linewidth=1, markersize=3)
    plt.plot(x, test_loss, 'bs-', label='Test loss', linewidth=1, markersize=3)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(x, train_acc, 'ro-', label='Train accuracy', linewidth=1, markersize=3)
    plt.plot(x, test_acc, 'bs-', label='Test accuracy', linewidth=1, markersize=3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend(loc='lower right')
    plt.title('Training and Test Accuracy')

    plt.savefig(plt_save_path + '\\' + 'loss_and_accuracy')
    plt.show()

