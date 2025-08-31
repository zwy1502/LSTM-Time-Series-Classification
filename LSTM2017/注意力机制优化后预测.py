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
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
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
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

class MLPAttention(nn.Module):
    def __init__(self, hidden_size, attention_size=64):
        super(MLPAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = attention_weights.squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states)
        context = context.squeeze(1)
        return context, attention_weights

class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_size=64):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([MLPAttention(hidden_size, attention_size) for _ in range(num_heads)])
        # 修改这里，确保 scale 的初始化与训练代码一致
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    def forward(self, hidden_states):
        contexts = []
        attention_weights = []
        for head in self.attention_heads:
            context, weights = head(hidden_states)
            contexts.append(context)
            attention_weights.append(weights)

        context = torch.cat(contexts, dim=1)
        attention_weights = torch.mean(torch.stack(attention_weights), dim=0) / self.scale
        return context, attention_weights
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, num_layers=3, output_size=5, dropout_rate=0.5,
                 num_attention_heads=4, attention_size=64, attention_regularization=0.01):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.attention_regularization = attention_regularization

        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.attention = MultiHeadedAttention(hidden_layer_size, num_attention_heads, attention_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_attention_heads * hidden_layer_size, output_size)
        self.residual_fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        residual = self.residual_fc(x[:, -1, :])

        h0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0],
                         self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0],
                         self.hidden_layer_size).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        context, attention_weights = self.attention(lstm_out)

        context = self.dropout(context)

        output = self.fc(context)
        output = output + residual

        if self.attention_regularization > 0:
            attention_loss = self.attention_regularization * torch.norm(attention_weights, p=1)
            output = output - attention_loss

        return output, attention_weights


if __name__ == '__main__':
    # 加载训练好的模型
    checkpoint = torch.load(r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1YCLSTM2020best_model_attention.pth",
                            map_location=device)

    # 只传递模型需要的参数
    model = AttentionLSTMClassifier(
        input_size=5,
        output_size=5,
        hidden_layer_size=checkpoint['params']['hidden_layer_size'],
        num_layers=checkpoint['params']['num_layers'],
        dropout_rate=checkpoint['params']['dropout_rate'],
        num_attention_heads=checkpoint['params']['num_attention_heads'],
        attention_size=checkpoint['params']['attention_size'],
        attention_regularization=checkpoint['params']['attention_regularization']
    )

    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 读取数据
    dataset1 = gdal.Open(r"D:\research\fenlei\dalunwen\JS\画图文件\2020YCLSTMS1.tif")
    im_data1 = dataset1.ReadAsArray()
    Tif_geotrans = dataset1.GetGeoTransform()
    Tif_proj = dataset1.GetProjection()

    # 处理数据
    im11 = im_data1.reshape(im_data1.shape[0], -1)
    id = np.logical_not(np.isnan(im11).any(axis=0))  # 检查每一列是否有任何NaN
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
            y_pred2, _ = model(seq2)
            # y_pred2 = torch.softmax(y_pred2, dim=1) # 这行不需要，因为模型输出已经过了最后的处理
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

    save_path = r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\2020YCS1_predict_result_attention5.tif"
    writeTiff(a, Tif_geotrans, Tif_proj, save_path)
    print('ok')