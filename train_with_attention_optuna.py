"""
@File    :   train_with_attention_optuna.py
@Time    :   2025-08-31
@Author  :   Wenyi Zhou (周文毅)
@Contact :   zhouzhouwang1502@gmail.com
@Desc    :   本项目是用于时序遥感影像分类的LSTM模型的高级实现版本。
             核心亮点包括：
             1. **注意力机制 (Attention Mechanism)**
             2. **自动化超参数寻优 (Optuna)**
             3. **处理样本不平衡 (SMOTE)**
             4. **详尽的日志与评估**
"""
import torch
import time
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import optuna
import os
# 在文件开头添加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 修改：添加时间戳到日志文件名，避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')

        # 修改：添加更详细的日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)  # 使用__name__替代默认logger

    def log_config(self, config_dict):
        self.logger.info("=== Training Configuration ===")
        # 修改：添加配置分类记录
        for category, params in {
            "Model Parameters": ['hidden_layer_size', 'num_layers', 'dropout_rate',
                                 'num_attention_heads', 'attention_size'],
            "Training Parameters": ['batch_size', 'optimizer', 'learning_rate',
                                    'weight_decay'],
            "System Parameters": ['device', 'total_epochs']
        }.items():
            self.logger.info(f"\n{category}:")
            for param in params:
                if param in config_dict:
                    self.logger.info(f"  {param}: {config_dict[param]}")

    def log_epoch(self, epoch_info):
        self.logger.info(f"\n=== Epoch {epoch_info['epoch']} ===")
        self.logger.info(
            f"训练损失: {epoch_info['train_loss']:.5f}, "
            f"验证损失: {epoch_info['val_loss']:.5f}"
        )
        self.logger.info(
            f"训练准确率: {epoch_info['train_acc']:.2f}%, "
            f"验证准确率: {epoch_info['val_acc']:.2f}%"
        )

    def log_best_model(self, model_info):
        self.logger.info("\n=== Best Model Information ===")
        self.logger.info(f"最佳轮次: {model_info['epoch']}")
        self.logger.info(f"最佳验证损失: {model_info['val_loss']:.5f}")
        self.logger.info(f"模型保存路径: {model_info['path']}")
    def log_final_metrics(self, metrics):
        self.logger.info("\n=== Final Model Performance ===")
        # 修改：添加时间戳和更详细的指标记录
        self.logger.info(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        self.logger.info(f"Kappa Coefficient: {metrics['kappa']:.4f}")
        self.logger.info(f"Macro F1 Score: {metrics['f1_macro']:.4f}")

        # 添加：每个类别的详细性能表格
        self.logger.info("\nPer-class Performance:")
        headers = ["Class", "UA (%)", "PA (%)", "F1 (%)"]
        class_data = []
        for i, (ua, pa, f1) in enumerate(zip(
                metrics['ua'], metrics['pa'], metrics['f1_per_class'])):
            class_data.append([
                f"Class {i + 1}",
                f"{ua * 100:.2f}",
                f"{pa * 100:.2f}",
                f"{f1 * 100:.2f}"
            ])
        self.logger.info("\n" + tabulate(class_data, headers=headers, tablefmt='grid'))
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=False):
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
    conf_matrix = confusion_matrix(y_true, y_pred)
    overall_accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    ua = np.zeros(n_classes)
    pa = np.zeros(n_classes)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i in range(n_classes):
        ua[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) != 0 else 0
        pa[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i]) if np.sum(conf_matrix[:, i]) != 0 else 0

    return overall_accuracy, kappa, f1_macro, ua, pa, conf_matrix, f1_per_class


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
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)  # 添加缩放因子

    def forward(self, hidden_states):
        contexts = []
        attention_weights = []
        for head in self.attention_heads:
            context, weights = head(hidden_states)
            contexts.append(context)
            attention_weights.append(weights)

        context = torch.cat(contexts, dim=1)
        attention_weights = torch.mean(torch.stack(attention_weights), dim=0) / self.scale  # 添加缩放
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
def fit(epoch, model, train_loader, test_loader, loss_fn1, optimizer, n_classes=5):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    attention_weights_sum = torch.zeros(12).to(device)
    attention_weights_count = 0

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
            attention_weights_sum += attention_weights.mean(0)
            attention_weights_count += 1
            all_train_preds.extend(pred_labels.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())

    train_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    avg_attention_weights = (attention_weights_sum / attention_weights_count).cpu()

    train_metrics = calculate_metrics(
        np.array(all_train_labels),
        np.array(all_train_preds),
        n_classes
    )

    model.eval()
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    test_attention_weights_sum = torch.zeros(12).to(device)
    test_attention_weights_count = 0

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
            test_attention_weights_sum += attention_weights.mean(0)
            test_attention_weights_count += 1

            all_test_preds.extend(pred_labels.cpu().numpy())
            all_test_labels.extend(y.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    epoch_test_loss = test_running_loss / len(test_loader)
    test_avg_attention_weights = (test_attention_weights_sum / test_attention_weights_count).cpu()

    test_metrics = calculate_metrics(
        np.array(all_test_labels),
        np.array(all_test_preds),
        n_classes
    )

    table = [
        ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'],
        [epoch + 1, round(epoch_loss, 5), round(train_accuracy, 4),
         round(epoch_test_loss, 5), round(test_accuracy, 4)],
    ]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=False))

    print("\n=== 详细精度报告 ===")
    print(f"训练集总体精度: {train_accuracy:.2f}%")
    print(f"训练集Kappa系数: {train_metrics[1]:.4f}")
    print(f"训练集F1分数(宏平均): {train_metrics[2]:.4f}")
    print(f"验证集总体精度: {test_accuracy:.2f}%")
    print(f"验证集Kappa系数: {test_metrics[1]:.4f}")
    print(f"验证集F1分数(宏平均): {test_metrics[2]:.4f}\n")

    class_metrics = [['类别', '训练UA(%)', '训练PA(%)', '训练F1(%)', '验证UA(%)', '验证PA(%)', '验证F1(%)']]
    for i in range(n_classes):
        class_metrics.append([
            f"类别 {i + 1}",
            f"{train_metrics[3][i] * 100:.2f}",
            f"{train_metrics[4][i] * 100:.2f}",
            f"{train_metrics[6][i] * 100:.2f}",
            f"{test_metrics[3][i] * 100:.2f}",
            f"{test_metrics[4][i] * 100:.2f}",
            f"{test_metrics[6][i] * 100:.2f}"
        ])
    print(tabulate(class_metrics, headers='firstrow', tablefmt='fancy_grid'))
    print("=" * 50)

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
    # 创建日志保存路径
    plt_save_path = r'D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\rd_lstmS1YC2020_attention'
    log_dir = os.path.join(plt_save_path, 'logs')
    logger = TrainingLogger(log_dir)
    df = pd.read_excel(r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1LSTM2020.xlsx")
    Y_train_2020 = df.iloc[:, 0]
    X_train_2020 = df.iloc[:, 1:]
    name = df.columns[1:]

    Y_train_2020 = Y_train_2020.values - 1
    X_train_2020 = X_train_2020.values

    vv = X_train_2020[:, 0:12]
    vh = X_train_2020[:, 12:24]
    SAR_Sum = X_train_2020[:, 24:36]
    SAR_Diff = X_train_2020[:, 36:48]
    SAR_NDVI = X_train_2020[:, 48:60]

    x_2020 = np.zeros((vh.shape[0], vh.shape[1], 5))
    x_2020[:, :, 0] = vv
    x_2020[:, :, 1] = vh
    x_2020[:, :, 2] = SAR_Sum
    x_2020[:, :, 3] = SAR_Diff
    x_2020[:, :, 4] = SAR_NDVI

    num_total = x_2020.shape[0]
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train
    Arange = list(range(num_total))
    random.shuffle(Arange)

    train_list = []
    for i in range(num_train):
        idx = Arange.pop()
        train_list.append(idx)

    x_train = x_2020[train_list, :, :]
    x_test = x_2020[Arange, :, :]
    y_train = Y_train_2020[train_list]
    y_test = Y_train_2020[Arange]

    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 使用 SMOTE 进行重采样
    smote = SMOTE(random_state=42)  # 确保可重复性
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    print("重采样前各类别数量：", np.bincount(y_train))
    print("重采样后各类别数量：", np.bincount(y_train_resampled))
    # 将重采样后的数据恢复为原始的张量形状
    x_train_resampled = x_train_resampled.reshape(-1, 12, 5)  # 根据你的数据形状修改
    x_test = x_test.reshape(-1, 12, 5)

    x_train = torch.Tensor(x_train_resampled).type(torch.FloatTensor)
    y_train = torch.Tensor(y_train_resampled).type(torch.int64)
    x_test = torch.Tensor(x_test).type(torch.FloatTensor)
    y_test = torch.Tensor(y_test).type(torch.int64)

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


    def objective(trial):
        # 在objective函数中使用了未定义的loss_fn1，需要在函数内部定义
        print(f"\n开始第 {trial.number + 1} 次试验")
        loss_fn1 = nn.CrossEntropyLoss()  # 需要添加这一行

        # DataLoader的创建在objective函数中需要使用全局变量
        global x_train, y_train, x_test, y_test  # 需要在函数开头添加这一行
        # 扩展模型架构的超参数搜索空间
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 32, 512)  # 扩大范围
        num_layers = trial.suggest_int('num_layers', 1, 6)  # 增加可能的层数
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)  # 扩大dropout范围
        num_attention_heads = trial.suggest_int('num_attention_heads', 1, 8)  # 增加注意力头数范围
        attention_size = trial.suggest_int('attention_size', 16, 256)  # 扩大注意力维度范围
        attention_regularization = trial.suggest_float('attention_regularization', 0.0001, 0.1, log=True)  # 使用对数尺度

        # 优化器相关超参数
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)  # 扩大学习率范围
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)  # 扩大权重衰减范围

        # 新增的超参数
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  # 批量大小选择
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])  # 优化器选择

        # scheduler相关超参数
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.5)  # 学习率衰减因子
        scheduler_patience = trial.suggest_int('scheduler_patience', 5, 15)  # 调度器耐心值

        # 初始化模型
        model = AttentionLSTMClassifier(
            input_size=5,
            hidden_layer_size=hidden_layer_size,
            num_layers=num_layers,
            output_size=5,
            dropout_rate=dropout_rate,
            num_attention_heads=num_attention_heads,
            attention_size=attention_size,
            attention_regularization=attention_regularization
        ).to(device)

        # 选择优化器
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 更新学习率调度器
        exp_lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
            min_lr=1e-6
        )

        # 更新数据加载器的批量大小
        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(x_train, y_train),
            batch_size=batch_size,  # 使用当前trial的batch_size
            shuffle=True,
            num_workers=0,
        )

        test_loader = Data.DataLoader(
            dataset=Data.TensorDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        # 初始化早停机制
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)
        # 训练循环
        epochs = 1000
        best_test_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss, epoch_test_loss, _, _, _, _, _, _ = fit(
                epoch,
                model,
                train_loader,
                test_loader,
                loss_fn1,
                optimizer,
                n_classes=5
            )

            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss

            early_stopping(epoch_test_loss)
            if early_stopping.early_stop:
                break

            exp_lr_scheduler.step(epoch_test_loss)

            trial.report(epoch_test_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_test_loss

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=3
        ),
        sampler=optuna.samplers.TPESampler(seed=42)  # 添加采样器并设置随机种子
    )
    # 增加优化试验次数
    study.optimize(objective, n_trials=20)  # 增加到50次或更多

    # 打印最佳超参数
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 使用最佳超参数训练最终模型
    # 加载所有最佳参数
    best_batch_size = trial.params['batch_size']
    best_optimizer_name = trial.params['optimizer']
    best_hidden_layer_size = trial.params['hidden_layer_size']
    best_num_layers = trial.params['num_layers']
    best_dropout_rate = trial.params['dropout_rate']
    best_num_attention_heads = trial.params['num_attention_heads']
    best_attention_size = trial.params['attention_size']
    best_attention_regularization = trial.params['attention_regularization']
    best_learning_rate = trial.params['learning_rate']
    best_weight_decay = trial.params['weight_decay']
    # 在获取最佳参数后，添加以下代码：
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_train, y_train),
        batch_size=best_batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),
        batch_size=best_batch_size,
        shuffle=False,
        num_workers=0,
    )
    final_model = AttentionLSTMClassifier(
        input_size=5,
        hidden_layer_size=best_hidden_layer_size,
        num_layers=best_num_layers,
        output_size=5,
        dropout_rate=best_dropout_rate,
        num_attention_heads=best_num_attention_heads,
        attention_size=best_attention_size,
        attention_regularization=best_attention_regularization
    ).to(device)

    loss_fn1 = nn.CrossEntropyLoss()
    # 根据最佳优化器名称选择优化器
    if best_optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
    elif best_optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
    else:
        optimizer = torch.optim.RMSprop(final_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
    exp_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True,
                                         min_lr=1e-6)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)
    # 训练过程
    epochs = 3000 # 训练最终模型时 epoch 可以设置大一点
    # 记录训练配置
    logger.log_config({
        'hidden_layer_size': best_hidden_layer_size,
        'num_layers': best_num_layers,
        'dropout_rate': best_dropout_rate,
        'num_attention_heads': best_num_attention_heads,
        'attention_size': best_attention_size,
        'batch_size': best_batch_size,
        'optimizer': best_optimizer_name,
        'learning_rate': best_learning_rate,
        'weight_decay': best_weight_decay,
        'device': str(device),
        'total_epochs': epochs
    })

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_path = r"D:\research\fenlei\dalunwen\JS\画图文件\YCLSTM\S1YCLSTM2020best_model_attention.pth"

    for epoch in range(epochs):
        epoch_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc, train_metrics, test_metrics, train_attention_weights, test_attention_weights = fit(
            epoch,
            final_model,
            train_loader,
            test_loader,
            loss_fn1,  # 添加这个参数
            optimizer,  # 添加这个参数
            n_classes=5
        )
        train_loss.append(round(epoch_loss, 5))
        test_loss.append(round(epoch_test_loss, 5))
        train_acc.append(round(epoch_train_acc, 2))
        test_acc.append(round(epoch_test_acc, 2))
        # 记录每个epoch的信息
        logger.log_epoch({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_test_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_test_acc
        })
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch
            # 保存模型参数和状态
            model_save = {
                'state_dict': final_model.state_dict(),
                'params': {
                    'hidden_layer_size': best_hidden_layer_size,
                    'num_layers': best_num_layers,
                    'dropout_rate': best_dropout_rate,
                    'num_attention_heads': best_num_attention_heads,
                    'attention_size': best_attention_size,
                    'attention_regularization': best_attention_regularization,
                    'optimizer_name': best_optimizer_name,
                    'batch_size': best_batch_size,
                    'learning_rate': best_learning_rate,
                    'weight_decay': best_weight_decay,
                    'best_epoch': best_epoch,
                    'best_loss': best_test_loss,
                    'training_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
            }
            torch.save(model_save, best_model_path)
            print(f"保存了新的最佳模型，验证损失: {best_test_loss}，位于第 {best_epoch} 轮")
            # 记录最佳模型信息
            logger.log_best_model({
                'epoch': best_epoch + 1,
                'val_loss': best_test_loss,
                'path': best_model_path
            })
        early_stopping(epoch_test_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练。")
            break
        exp_lr_scheduler.step(epoch_test_loss)
        # 记录最终模型性能
    logger.log_final_metrics({
        'accuracy': epoch_test_acc,
        'kappa': test_metrics[1],
        'f1_macro': test_metrics[2],
        'ua': test_metrics[3],
        'pa': test_metrics[4],
        'f1_per_class': test_metrics[6]
    })
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