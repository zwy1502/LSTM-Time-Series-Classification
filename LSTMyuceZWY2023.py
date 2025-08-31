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
    def __init__(self, input_size=5, hidden_layer_size=128, num_layers=3, output_size=5, dropout_rate=0.5):
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
    model.load_state_dict(torch.load(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S12023best_model月度.pth"))  # 读取训练好的模型
    model.eval()
    dataset1 = gdal.Open(r"F:\ZWY\JS2023_2\新建文件夹\S12023_ROI_0_0_Monthly-0000000000-0000000000.tif")  # 读取数据 r"D:\edge_download\merged_image.tif"
    im_data1 = dataset1.ReadAsArray()
    Tif_geotrans = dataset1.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset1.GetProjection()  # 获取投影信息

    im11 = im_data1.reshape(im_data1.shape[0], -1)
    id = np.logical_not(np.isnan(im11[0, :]))
    im111 = im11[:, id]
    im111 = im111.T.copy()
    # rgb = im111[:, 0:5]  ##
    # zhishu = im111[:, 5:10]
    # VV = im111[:, 0:23]  # 从0:24改为0:23
    # VH = im111[:, 23:46]  # 从24:48改为23:46
    # SAR_Sum = im111[:, 46:69]  # 从48:72改为46:69
    # SAR_Diff = im111[:, 69:92]  # 从72:96改为69:92
    # SAR_NDVI = im111[:, 92:115]  # 从96:120改为92:115
    VV = im111[:, 0:11]
    VH = im111[:, 11:22]
    SAR_Sum = im111[:, 22:33]
    SAR_Diff = im111[:, 33:44]
    SAR_NDVI = im111[:, 44:55]
    # SDWI = VH_VV_DPSVIm[48:72, :, :].reshape(24, VH_VV_DPSVIm.shape[1] * VH_VV_DPSVIm.shape[2]).T
    # DPSVIm = VH_VV_DPSVIm[72:96, :, :].reshape(24, VH_VV_DPSVIm.shape[1] * VH_VV_DPSVIm.shape[2]).T

    # pred_img = np.zeros((im111.shape[0], 23, 5))  ##记得第二个是维度
    pred_img = np.zeros((im111.shape[0], 11, 5))  ##记得第二个是维度

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
        y_pred2 = torch.softmax(y_pred2, dim=1)  # 手动应用softmax
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
    # a = a.astype(np.uint8)
    a = np.nan_to_num(a, nan=0).astype(np.uint8)  # 将NaN替换为0，然后转换为uint8类型
    a = a.reshape(im_data1.shape[1], im_data1.shape[2])
    save_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\S1结果分块\S12023_ROI_7_0测试.tif" #r"D:\edge_download\S1_lstmpredict早停11.tif"
    writeTiff(a, Tif_geotrans, Tif_proj, save_path)
    print('OK')
