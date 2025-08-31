# -*- coding: utf-8 -*-
# @Time   :  2023/5/15 20:33
# @Author : WangHui
# @Email  : wanghuiviac@163.com
import numpy as np
from osgeo import gdal
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, input_size=5, hidden_layer_size=40, num_layers=5, output_size=4):
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
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.num_layers, batch_first=True, bidirectional=False)  # 调用lstm
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
        linear_out = self.linear(lstm_out[:, -1, :])  # =self.linear(lstm_out[:, -1, :])#使用全连接层linear
        predictions = torch.softmax(linear_out, dim=1)  # 使用sigmoid函数
        # predictions = predictions[:, -1, :]
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # # output(batch_size, seq_len, num_directions * hidden_size)
        # output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        # pred = self.linear(output)  # (5, 30, 1)
        # pred = pred[:, -1, :]  # (5, 1)

        return predictions


def load_img(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # （0，0）是读取的起始位置

    # if im_data.ndim == 3:
    #     im_data = im_data.transpose((1,2,0))
    return im_data


if __name__ == '__main__':
    # 读取模型
    model = LSTMClassifier()
    model.cuda()
    model.load_state_dict(torch.load(r"D:\edge_download\rd_lstm_vhvv15_10k.pth"))  # 读取训练好的模型
    model.eval()
    dataset1 = gdal.Open(r"D:\edge_download\merged_image.tif")  # 读取数据
    im_data1 = dataset1.ReadAsArray()
    Tif_geotrans = dataset1.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset1.GetProjection()  # 获取投影信息

    im11 = im_data1.reshape(im_data1.shape[0], -1)
    id = np.logical_not(np.isnan(im11[0, :]))
    im111 = im11[:, id]
    im111 = im111.T.copy()
    # rgb = im111[:, 0:5]  ##
    # zhishu = im111[:, 5:10]
    VV = im111[:, 0:24]
    VH = im111[:, 24:48]
    SAR_Sum = im111[:, 48:72]
    SAR_Diff = im111[:, 72:96]
    SAR_NDVI = im111[:, 96:120]
    # SDWI = VH_VV_DPSVIm[48:72, :, :].reshape(24, VH_VV_DPSVIm.shape[1] * VH_VV_DPSVIm.shape[2]).T
    # DPSVIm = VH_VV_DPSVIm[72:96, :, :].reshape(24, VH_VV_DPSVIm.shape[1] * VH_VV_DPSVIm.shape[2]).T

    pred_img = np.zeros((im111.shape[0], 24, 5))  ##
    # pred_img[:, :, 0] = rgb
    # pred_img[:, :, 1] = zhishu
    pred_img[:, :, 0] = VV
    pred_img[:, :, 1] = VH
    pred_img[:, :, 2] = SAR_Sum
    pred_img[:, :, 3] = SAR_Diff
    pred_img[:, :, 4] = SAR_NDVI
    pred_img = torch.Tensor(pred_img).to(device)

    # im_ = np.zeros((im111.shape[0], im111.shape[1],1))
    # im_[:, :, 0] = im111
    #
    # im = torch.Tensor(im_).to(device)
    # del im_
    # print(im.shape[0])

    b_pre = []
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(pred_img),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=10000,  # 每块的大小   加载的每批次样本数据大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )
    for seq2 in test_loader:
        # seq2 = seq2[0].transpose(0, 1)
        seq2 = seq2[0].to(device)
        y_pred2 = model(seq2).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        im_pred2 = torch.argmax(y_pred2, dim=1)
        im_pred2 = im_pred2.detach().cpu().tolist()
        b_pre.extend((im_pred2))
    # for i in tqdm(range(0,im.shape[1],1)):
    #     im_pred = model(im[:,i:i+1,:]).squeeze()
    #     im_pred2=torch.argmax(im_pred,dim=0)
    #     b_pre.append(im_pred2.detach().cpu().numpy())

    a = np.full((1, im11.shape[1]), np.nan)
    num = pred_img.shape[0]
    b_pre = np.array(b_pre)
    b_pre = b_pre.reshape(1, num)
    b_pre = b_pre + 1
    a[:, id] = b_pre
    a = a.astype(np.uint8)
    a = a.reshape(im_data1.shape[1], im_data1.shape[2])
    save_path = r"D:\edge_download\S1_lstmpredict.tif"
    writeTiff(a, Tif_geotrans, Tif_proj, save_path)
    print('OK')
