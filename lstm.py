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
import random
import xlrd
from sklearn.metrics import accuracy_score

# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        # im_data = np.array([im_data])
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=20, num_layers=3, output_size=7):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数  也是分类类数
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.num_directions = 1  # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.num_layers, batch_first=True)  # 调用lstm   , batch_first=True
        self.linear = nn.Linear(hidden_layer_size, output_size)  # 调用全连接

    # 调用激活函数多分类激活函数

    def forward(self, input_x):
        # batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        # input_x = input_x.view(len(input_x), 1, -1)  # view函数是改变维度，input_x输入
        h0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)
        c0 = torch.zeros(self.num_directions * self.num_layers, input_x.shape[0], self.hidden_layer_size).to(device)
        # hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
        #                torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, (h0, c0))  # 两个输入是 h0 和 c0，可以理解成网络的初始化参数
        linear_out = self.linear(lstm_out[ :, -1, :])  # =self.linear(lstm_out[:, -1, :])#使用全连接层linear
        predictions = torch.softmax(linear_out, dim=0)  # 使用softmax函数(多分类)


        return predictions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(r"C:\Users\Viac\Desktop\大论文实验\时序曲线3\总.xlsx")
    Y_train_2020 = df.iloc[:, 0]
    X_train_2020 = df.iloc[:, 1:]
    name = df.columns[1:]

    Y_train_2020 = Y_train_2020.values
    X_train_2020 = X_train_2020.values
    rgb = X_train_2020[:,0:5]
    zhishu = X_train_2020[:,5:10]
    vh20 = X_train_2020[:, 10:15]
    vv20 = X_train_2020[:, 15:20]
    # dpsvim20 = X_train_2020[:, 36:48]

    x_2020 = np.zeros((vh20.shape[0], vh20.shape[1], 4))
    x_2020[:,:,0] = rgb
    x_2020[:, :, 1] = zhishu
    x_2020[:, :, 2] = vh20
    x_2020[:, :, 3] = vv20

    # roi_2019 = pd.read_excel("D:\\data_set\\shiyan_nr2\\s1_2019\\roi\\roi_2019.xls")
    # Y_train_2019 = roi_2019.iloc[:, 0]
    # X_train_2019 = roi_2019.iloc[:, 1:]
    # Y_train_2019 = Y_train_2019.values
    # X_train_2019 = X_train_2019.values
    # vh19 = X_train_2019[:, 0:12]
    # vv19 = X_train_2019[:, 12:24]
    # dpsvim19 = X_train_2019[:, 24:36]
    # x_2019 = np.zeros((vh19.shape[0], vh19.shape[1], 3))
    # x_2019[:, :, 0] = vh19
    # x_2019[:, :, 1] = vv19
    # x_2019[:, :, 2] = dpsvim19

    # roi_2021 = pd.read_excel("D:\\data_set\\shiyan_nr2\\s1_2021\\ROI\\roi_2021.xls")
    # Y_train_2021 = roi_2021.iloc[:, 0]
    # X_train_2021 = roi_2021.iloc[:, 1:]
    # Y_train_2021 = Y_train_2021.values
    # X_train_2021 = X_train_2021.values
    # vh21 = X_train_2021[:, 0:12]
    # vv21 = X_train_2021[:, 12:24]
    # dpsvim21 = X_train_2021[:, 24:36]
    #
    # x_2021 = np.zeros((vh21.shape[0], vh21.shape[1], 3))
    # x_2021[:, :, 0] = vh21
    # x_2021[:, :, 1] = vv21
    # x_2021[:, :, 2] = dpsvim21


    # x = np.concatenate((x_1, x_2021), axis=0)
    # y_1 = np.concatenate((Y_train_2019, Y_train_2020), axis=0)
    # y = np.concatenate((y_1, Y_train_2021), axis=0)
    # # y[y == 5] = 3
    # y = y - 1
    x = x_2020
    y = Y_train_2020 - 1

    num_total = x.shape[0]#样本数量
    num_train = int(0.75 * num_total)  # 其中训练样本包括验证样本
    num_test = num_total - num_train
    Arange = list(range(num_total))
    random.shuffle(Arange)  # 打乱顺序，分配训练样本、验证和测试样本
    train_list = []
    for i in range(num_train):
        idx = Arange.pop()
        train_list.append(idx)

    x_train = x[train_list, :, :]
    x_test = x[Arange, :, :]
    y_train = y[train_list]
    y_test = y[Arange]

    # # 划分样本集测试集占总样本0.3，stratify=y按照标签等比例划分
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,train_size=0.7,stratify=y,random_state=4)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    y_train = np.array(y_train)
    # y_train=y_train.reshape([1,x_train.shape[1]])
    x_train = torch.Tensor(np.array(x_train))
    y_train = torch.Tensor(y_train)
    print(x_train.shape)
    print(y_train.shape)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=x_train.shape[0],  # 每块的大小   加载的每批次样本数据大小   注意好像这个参数设的不好
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    x_test = torch.Tensor(np.array(x_test))
    y_test = torch.Tensor(np.array(y_test))
    print(x_test.shape)
    print(y_test.shape)
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=x_test.shape[0],  # 每块的大小   加载的每批次样本数据大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )
    # 初始化模型
    # 建模三件套：loss，优化，epochs
    model = LSTMClassifier()  # 模型
    model.to(device=device)

    loss_function = nn.CrossEntropyLoss()  # loss多分类损失函数对应softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 优化器  adm优化器就是梯度下降，更牛逼   Ir是学习率
    epochs = 5000
    # 开始训练
    single_loss0 = []
    single_loss11 = []
    acc11=[]
    for i in tqdm(range(epochs)):
        model.train()
        for seq, labels in train_loader:
            # seq=seq.transpose(0,1)
            seq = seq.to(device)
            labels = labels.squeeze()
            labels = labels.to(device)
            print(seq.shape)

            optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss关于weight的导数变成0.
            y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除
            y_pred = y_pred.to(device)
            print(y_pred.shape)

            # labels = F.one_hot(labels.to(torch.long)) / 1.0  #独热编码  交叉熵损失函数会自动转换独热编码
            labels = labels.to(device)
            labels = labels.long()
            print(labels.shape)
            # y_pred2=np.argmax(y_pred,axis=1)
            # y_pred3=y_pred2+1
            single_loss = loss_function(y_pred, labels)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()  # 反向传播计算得到每个参数的梯度值
            optimizer.step()  # 通过梯度下降执行一步参数更新
            print("Train Step:", i, " loss: ", single_loss)
            single_loss += single_loss

        single_loss /= len(train_loader)
        single_loss = single_loss.cpu().detach().numpy()
        single_loss0.append(single_loss)

        # 开始验证

        model.eval()
        for seq2, labels2 in test_loader:
            # seq2 = seq2.transpose(0, 1)
            seq2 = seq2.to(device)
            labels2 = labels2.squeeze()
            # labels2 = F.one_hot(labels2.to(torch.long)) / 1.0
            labels2 = labels2.to(device)
            labels2 = labels2.long()
            print(labels2.shape)

            y_pred2 = model(seq2).squeeze()  # 压缩维度：得到输出，并将维度为1的去除

            single_loss1 = loss_function(y_pred2, labels2)
            print("EVAL Step:", i, " loss: ", single_loss1)
            single_loss1 += single_loss1
            # im_pred2 = torch.argmax(y_pred2, dim=1)
            # im_pred2 = im_pred2.detach().cpu().numpy()
            # labels2 = labels2.detach().cpu().numpy()
            # acc1=accuracy_score(labels2,im_pred2)
            # acc1 +=acc1

        single_loss1 /= len(test_loader)
        single_loss1 = single_loss1.cpu().detach().numpy()
        single_loss11.append(single_loss1)
        # acc1 /=len(test_loader)
        # acc11.append(acc1)


    ## 保存模型
    torch.save(model.state_dict(), r'E:\nzw\model\model_lstm3.pth')
    # 绘制损失函数图
    x_axis = [range(0, epochs)]
    x2_axis = np.array(x_axis).reshape(-1, 1)
    # plt.figure(dpi=600)
    plt.plot(x2_axis, single_loss0, 'b', label='y_trn')
    plt.plot(x2_axis, single_loss11, 'y', label='pre_trn')
    # plt.plot(x2_axis, acc11, 'r', label='acc')

    plt.xlabel('epochs')
    plt.ylabel('Vce')
    plt.legend()
    plt.show()
    print("ok")

    ## 读取模型
    # model = torch.load('model_name.pth')

    # 评估模型
    # with torch.no_grad():
    #     test_inputs = torch.Tensor(x_test)
    #     test_labels = torch.Tensor(y_test).unsqueeze(1)
    #     test_outputs = model(test_inputs)
    #     test_loss = criterion(test_outputs, test_labels)
    #     predicted = (test_outputs > 0.5).squeeze().long()
    #     accuracy = (predicted == test_labels.squeeze().long()).sum().item() / len(test_labels)
    # #     print("Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(test_loss.item(), accuracy * 100))
    # dataset1 = gdal.Open(r"D:\data_set\shiyan_nr2\all\VH_mean_mon_1.tif")
    # im_data1 = dataset1.ReadAsArray()
    # Tif_geotrans = dataset1.GetGeoTransform()  # 获取仿射矩阵信息
    # Tif_proj = dataset1.GetProjection()  # 获取投影信息
    # dataset2 = gdal.Open(r"D:\data_set\shiyan_nr2\all\VV_mean_mon_1.tif")
    # im_data2 = dataset2.ReadAsArray()
    # dataset3 = gdal.Open(r"D:\data_set\shiyan_nr2\all\DPSVIm_1.tif")
    # im_data3 = dataset3.ReadAsArray()
    # im11 = im_data1.reshape(im_data1.shape[0], -1)
    # im22 = im_data1.reshape(im_data2.shape[0], -1)
    # im33 = im_data1.reshape(im_data3.shape[0], -1)
    # id = np.logical_not(np.isnan(im11[0, :]))
    # im111 = im11[:, id]
    # im222 = im22[:, id]
    # im333 = im33[:, id]
    # im111 = im111.T.copy()
    # im222 = im222.T.copy()
    # im333 = im333.T.copy()
    # im_ = np.zeros((im111.shape[0], im111.shape[1], 3))
    # im_[:, :, 0] = im111
    # im_[:, :, 1] = im222
    # im_[:, :, 2] = im333
    # im = torch.Tensor(im_).to(device)
    # del im_data2
    # del im_data3
    # del im22
    # del im111
    # del im222
    # del im33
    # del im333
    # del im_
    # print(im.shape[0])
    # b_pre = []
    #
    # test_loader3 = Data.DataLoader(
    #     dataset=Data.TensorDataset(im),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
    #     batch_size=30091,  # 每块的大小   加载的每批次样本数据大小
    #     shuffle=True,  # 要不要打乱数据 (打乱比较好)
    #     num_workers=0,  # 多进程（multiprocess）来读数据
    # )
    # for seq3 in test_loader3:  # 这里偷个懒，就用训练数据验证哈！
    #     seq3 = seq3[0].to(device)
    #     y_pred3 = model(seq3).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
    #     im_pred3 = torch.argmax(y_pred3, dim=1)
    #     im_pred3 = im_pred3.detach().cpu().numpy()
    #     b_pre.append((im_pred3))
    # # for i in tqdm(range(0,im.shape[0],1)):
    # #     im_pred = model(im[i,:,:]).squeeze()
    # #     im_pred2=torch.argmax(im_pred,dim=0)
    # #     b_pre.append(im_pred2.detach().cpu().numpy())
    #
    # # b_pre=b_pre.detach.cpu().numpy()
    # a = np.full((1, im11.shape[1]), np.nan)
    # num = im.shape[0]
    # b_pre = np.array(b_pre)
    # b_pre = b_pre.reshape(1, num)
    # b_pre = b_pre + 1
    # a[:, id] = b_pre
    # a = a.astype(np.uint8)
    # a = a.reshape(im_data1.shape[1], im_data1.shape[2])
    # savepath = r"D:\data_set\shiyan_nr2\result\jieguo_lstm_1.tif"
    # writeTiff(a, Tif_geotrans, Tif_proj, savepath)
    # print('ok')
