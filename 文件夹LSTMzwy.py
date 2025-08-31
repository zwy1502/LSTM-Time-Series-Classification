import os
from osgeo import gdal
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np

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

def get_tif_paths(folder):
    tif_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_paths.append(os.path.join(root, file))
    return tif_paths


# 输入文件夹路径
input_folder = r"F:\ZWY\JS2023_2"
image_paths = get_tif_paths(input_folder)

# 预测结果保存路径
output_folder = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\S1结果分块月度"
os.makedirs(output_folder, exist_ok=True)

# 读取模型
model = LSTMClassifier()
model.cuda()
model.load_state_dict(
    torch.load(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S12023best_model月度.pth"))
model.eval()

# 使用 tqdm 显示进度条
for img_path in tqdm(image_paths, desc="Processing images"):
    try:
        dataset = gdal.Open(img_path)
        im_data1 = dataset.ReadAsArray()
        Tif_geotrans = dataset.GetGeoTransform()
        Tif_proj = dataset.GetProjection()

        im11 = im_data1.reshape(im_data1.shape[0], -1)
        id = np.logical_not(np.isnan(im11[0, :]))
        im111 = im11[:, id].T.copy()
        # VV, VH, SAR_Sum, SAR_Diff, SAR_NDVI = im111[:, 0:24], im111[:, 24:48], im111[:, 48:72], im111[:, 72:96], im111[:, 96:120]
        # VV, VH, SAR_Sum, SAR_Diff, SAR_NDVI = im111[:, 0:23], im111[:, 23:46], im111[:, 46:69], im111[:, 69:92], im111[:, 92:115]
        VV, VH, SAR_Sum, SAR_Diff, SAR_NDVI = im111[:, 0:11], im111[:, 11:22], im111[:, 22:33], im111[:, 33:44], im111[:, 44:55]

        # pred_img = np.zeros((im111.shape[0], 24, 5))
        # pred_img = np.zeros((im111.shape[0], 23, 5))
        pred_img = np.zeros((im111.shape[0], 11, 5))  # 将23改为11

        pred_img[:, :, 0] = VV
        pred_img[:, :, 1] = VH
        pred_img[:, :, 2] = SAR_Sum
        pred_img[:, :, 3] = SAR_Diff
        pred_img[:, :, 4] = SAR_NDVI
        pred_img = torch.Tensor(pred_img).to(device)

        b_pre = []
        test_loader = Data.DataLoader(
            dataset=Data.TensorDataset(pred_img),
            batch_size=10000,
            shuffle=False,
            num_workers=0,
        )

        for seq2 in test_loader:
            seq2 = seq2[0].to(device)
            y_pred2 = model(seq2).squeeze()
            y_pred2 = torch.softmax(y_pred2, dim=1)
            im_pred2 = torch.argmax(y_pred2, dim=1)
            b_pre.extend(im_pred2.detach().cpu().tolist())

        a = np.full((1, im11.shape[1]), np.nan)
        b_pre = np.array(b_pre).reshape(1, pred_img.shape[0]) + 1
        a[:, id] = b_pre
        a = np.nan_to_num(a, nan=0).astype(np.uint8).reshape(im_data1.shape[1], im_data1.shape[2])

        # 保存结果
        output_file_path = os.path.join(output_folder, os.path.basename(img_path).replace('.tif', '_预测.tif'))
        writeTiff(a, Tif_geotrans, Tif_proj, output_file_path)
        print(f'Saved prediction for {img_path} to {output_file_path}')

    except Exception as e:
        print(f"Error processing {img_path}: {e}")