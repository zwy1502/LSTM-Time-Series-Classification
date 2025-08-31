import numpy as np
from osgeo import gdal
from torch.nn import functional as F
import torch
import torch.nn as nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_img(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # （0，0）是读取的起始位置

    # if im_data.ndim == 3:
    #     im_data = im_data.transpose((1,2,0))
    return im_data
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


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)

        # 计算基础attention scores
        attention_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)

        # 应用softmax得到最终的attention权重
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 将attention权重应用到hidden states
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context, attention_weights


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, num_layers=3, output_size=5, dropout_rate=0.5):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.num_directions = 1  # 单向LSTM

        # LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # 注意力层
        self.attention = TemporalAttention(hidden_layer_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0],
                         self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0],
                         self.hidden_layer_size).to(x.device)

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # 应用注意力机制
        context, attention_weights = self.attention(lstm_out)

        # 应用dropout
        context = self.dropout(context)

        # 全连接层
        output = self.fc(context)

        return output, attention_weights










if __name__ == '__main__':
    # 加载训练好的模型
    model = AttentionLSTMClassifier()
    model.load_state_dict(torch.load(r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1YCLSTM2020best_model_attention.pth"))
    model.to(device)
    model.eval()

    # 读取数据
    dataset1 = gdal.Open(r"D:\research\fenlei\dalunwen\JS\画图文件\2020YCLSTMS1.tif")
    im_data1 = dataset1.ReadAsArray()
    Tif_geotrans = dataset1.GetGeoTransform()
    Tif_proj = dataset1.GetProjection()

    # 处理数据
    im11 = im_data1.reshape(im_data1.shape[0], -1)
    id = np.logical_not(np.isnan(im11[0, :]))
    im111 = im11[:, id].T.copy()

    VV = im111[:, 0:12]
    VH = im111[:, 12:24]
    SAR_Sum = im111[:, 24:36]
    SAR_Diff = im111[:, 36:48]
    SAR_NDVI = im111[:, 48:60]

    pred_img = np.zeros((im111.shape[0], 12, 5))
    pred_img[:, :, 0] = VV
    pred_img[:, :, 1] = VH
    pred_img[:, :, 2] = SAR_Sum
    pred_img[:, :, 3] = SAR_Diff
    pred_img[:, :, 4] = SAR_NDVI
    pred_img = torch.Tensor(pred_img).to(device)

    # 预测
    b_pre = []
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(pred_img),
        batch_size=5000,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for seq2 in tqdm(test_loader, desc="预测进度"):
            seq2 = seq2[0].to(device)
            y_pred2, _ = model(seq2)  # 忽略注意力权重
            y_pred2 = torch.softmax(y_pred2, dim=1)
            im_pred2 = torch.argmax(y_pred2, dim=1)
            im_pred2 = im_pred2.detach().cpu().tolist()
            b_pre.extend((im_pred2))

    # 保存结果
    a = np.full((1, im11.shape[1]), np.nan)
    num = pred_img.shape[0]
    b_pre = np.array(b_pre).reshape(1, num) + 1
    a[:, id] = b_pre
    a = np.nan_to_num(a, nan=0).astype(np.uint8)
    a = a.reshape(im_data1.shape[1], im_data1.shape[2])

    save_path = r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\2020YCS1_predict_result_attention.tif"
    writeTiff(a, Tif_geotrans, Tif_proj, save_path)
    print('ok')