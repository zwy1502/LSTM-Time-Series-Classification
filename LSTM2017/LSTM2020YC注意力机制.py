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
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
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
            self.counter = 0
            if self.verbose:
                print(f"验证损失改善到 {self.best_loss}. 计数器重置.")



def calculate_metrics(y_true, y_pred, n_classes):
    """
    计算各类精度指标
    """
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算总体精度
    overall_accuracy = accuracy_score(y_true, y_pred)

    # 计算Kappa系数
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算每个类别的F1分数、用户精度(UA)和生产者精度(PA)
    ua = np.zeros(n_classes)
    pa = np.zeros(n_classes)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i in range(n_classes):
        # 用户精度 (precision)
        ua[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) != 0 else 0
        # 生产者精度 (recall)
        pa[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i]) if np.sum(conf_matrix[:, i]) != 0 else 0

    return overall_accuracy, kappa, f1_macro, ua, pa, conf_matrix, f1_per_class


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


def fit(epoch, model, train_loader, test_loader, n_classes=5):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    # 修改：确保在正确的设备上初始化
    attention_weights_sum = torch.zeros(12).to(device)
    attention_weights_count = 0

    # 用于收集所有预测结果
    all_train_preds = []
    all_train_labels = []

    for x, y in train_loader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')

        y_pred, attention_weights = model(x)
        loss = loss_fn1(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_labels = torch.argmax(y_pred, dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
            # 修改：确保attention_weights在正确的设备上
            attention_weights_sum += attention_weights.mean(0)  # 已经在GPU上，不需要.cpu()
            attention_weights_count += 1

            # 收集预测结果
            all_train_preds.extend(pred_labels.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())

    train_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    # 修改：最后再转到CPU
    avg_attention_weights = (attention_weights_sum / attention_weights_count).cpu()

    # 计算训练集的详细指标
    train_metrics = calculate_metrics(
        np.array(all_train_labels),
        np.array(all_train_preds),
        n_classes
    )

    # 验证过程
    model.eval()
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    # 修改：确保在正确的设备上初始化
    test_attention_weights_sum = torch.zeros(12).to(device)
    test_attention_weights_count = 0

    # 用于收集所有验证集预测结果
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred, attention_weights = model(x)
            loss = loss_fn1(y_pred, y)

            pred_labels = torch.argmax(y_pred, dim=1)
            test_correct += (pred_labels == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            # 修改：确保attention_weights在正确的设备上
            test_attention_weights_sum += attention_weights.mean(0)  # 已经在GPU上，不需要.cpu()
            test_attention_weights_count += 1

            # 收集预测结果
            all_test_preds.extend(pred_labels.cpu().numpy())
            all_test_labels.extend(y.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    epoch_test_loss = test_running_loss / len(test_loader)
    # 修改：最后再转到CPU
    test_avg_attention_weights = (test_attention_weights_sum / test_attention_weights_count).cpu()

    # 计算验证集的详细指标
    test_metrics = calculate_metrics(
        np.array(all_test_labels),
        np.array(all_test_preds),
        n_classes
    )
    # 打印基础训练信息
    table = [
        ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'],
        [epoch + 1, round(epoch_loss, 5), round(train_accuracy, 4),
         round(epoch_test_loss, 5), round(test_accuracy, 4)],
    ]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=False))

    # 打印详细精度报告
    print("\n=== 详细精度报告 ===")
    print(f"训练集总体精度: {train_accuracy:.2f}%")
    print(f"训练集Kappa系数: {train_metrics[1]:.4f}")
    print(f"训练集F1分数(宏平均): {train_metrics[2]:.4f}")
    print(f"验证集总体精度: {test_accuracy:.2f}%")
    print(f"验证集Kappa系数: {test_metrics[1]:.4f}")
    print(f"验证集F1分数(宏平均): {test_metrics[2]:.4f}\n")
    # 打印每个类别的UA和PA
    class_metrics = [['类别', '训练UA(%)', '训练PA(%)', '训练F1(%)', '验证UA(%)', '验证PA(%)', '验证F1(%)']]
    for i in range(n_classes):
        class_metrics.append([
            f"类别 {i + 1}",
            f"{train_metrics[3][i] * 100:.2f}",  # UA
            f"{train_metrics[4][i] * 100:.2f}",  # PA
            f"{train_metrics[6][i] * 100:.2f}",  # F1 per class
            f"{test_metrics[3][i] * 100:.2f}",  # UA
            f"{test_metrics[4][i] * 100:.2f}",  # PA
            f"{test_metrics[6][i] * 100:.2f}"  # F1 per class
        ])
    print(tabulate(class_metrics, headers='firstrow', tablefmt='fancy_grid'))
    print("=" * 50)

    # 每10个epoch打印注意力权重
    if epoch % 10 == 0:
        print("\n月份注意力权重:")
        month_weights = [['月份', '训练权重', '验证权重']]
        for month, (train_weight, test_weight) in enumerate(zip(avg_attention_weights, test_avg_attention_weights), 1):
            month_weights.append([
                f"月份 {month}",
                f"{train_weight:.4f}",
                f"{test_weight:.4f}"
            ])
        print(tabulate(month_weights, headers='firstrow', tablefmt='fancy_grid'))
        print("=" * 50)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"经过ReduceLROnPlateau调度器调整后的学习率： {current_lr}")

    return epoch_loss, epoch_test_loss, train_accuracy, test_accuracy, train_metrics, test_metrics, avg_attention_weights, test_avg_attention_weights

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    df = pd.read_excel(r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1LSTM2020.xlsx")
    Y_train_2020 = df.iloc[:, 0]  # 训练标签列
    X_train_2020 = df.iloc[:, 1:]  # 训练样本列
    name = df.columns[1:]  # 特征名称

    # 标签处理（从1开始的标签转换为从0开始）
    Y_train_2020 = Y_train_2020.values - 1
    X_train_2020 = X_train_2020.values

    # 提取每个指标的12个月数据
    vv = X_train_2020[:, 0:12]  # 12个月的VV数据
    vh = X_train_2020[:, 12:24]  # 12个月的VH数据
    SAR_Sum = X_train_2020[:, 24:36]  # 12个月的SAR_Sum数据
    SAR_Diff = X_train_2020[:, 36:48]  # 12个月的SAR_Diff数据
    SAR_NDVI = X_train_2020[:, 48:60]  # 12个月的SAR_NDVI数据

    # 重组数据shape为(样本数, 时间步长, 特征数)
    x_2020 = np.zeros((vh.shape[0], vh.shape[1], 5))
    x_2020[:, :, 0] = vv
    x_2020[:, :, 1] = vh
    x_2020[:, :, 2] = SAR_Sum
    x_2020[:, :, 3] = SAR_Diff
    x_2020[:, :, 4] = SAR_NDVI

    # 划分训练集和测试集
    num_total = x_2020.shape[0]
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train
    Arange = list(range(num_total))
    random.shuffle(Arange)

    train_list = []
    for i in range(num_train):
        idx = Arange.pop()
        train_list.append(idx)

    # 获取训练集和测试集
    x_train = x_2020[train_list, :, :]
    x_test = x_2020[Arange, :, :]
    y_train = Y_train_2020[train_list]
    y_test = Y_train_2020[Arange]

    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)

    # 转换为PyTorch张量
    x_train = torch.Tensor(np.array(x_train)).type(torch.FloatTensor)
    y_train = torch.Tensor(np.array(y_train)).type(torch.int64)
    x_test = torch.Tensor(np.array(x_test)).type(torch.FloatTensor)
    y_test = torch.Tensor(np.array(y_test)).type(torch.int64)

    # 创建数据加载器
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    # 初始化模型和训练参数
    model = AttentionLSTMClassifier().to(device)
    loss_fn1 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    exp_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True, min_lr=1e-6)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

    # 训练过程
    epochs = 5000
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_test_loss = float('inf')
    best_model_path = r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1YCLSTM2020best_model_attention.pth"

    # 开始训练
    for epoch in range(epochs):
        epoch_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc, train_metrics, test_metrics, train_attention_weights, test_attention_weights = fit(
            epoch, model, train_loader, test_loader, n_classes=5
        )
        train_loss.append(round(epoch_loss, 5))
        test_loss.append(round(epoch_test_loss, 5))
        train_acc.append(round(epoch_train_acc, 2))
        test_acc.append(round(epoch_test_acc, 2))

        # 检查是否需要保存最佳模型
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)  # 保存当前模型参数
            print(f"保存了新的最佳模型，验证损失: {best_test_loss}，位于第 {best_epoch} 轮")
        # 早停检查
        early_stopping(epoch_test_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练。")
            break

        # 学习率调整
        exp_lr_scheduler.step(epoch_test_loss)

    # 绘制训练过程曲线
    x = range(0, len(train_loss))
    plt_save_path = r'D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\rd_lstmS1YC2020_attention'

    if not os.path.exists(plt_save_path):
        os.makedirs(plt_save_path)

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

    plt.savefig(os.path.join(plt_save_path, 'loss_and_accuracy_attention'))
    plt.show()