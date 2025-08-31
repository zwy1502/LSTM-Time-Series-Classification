# 基于注意力机制与Optuna优化的遥感时序分类系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange) ![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple) ![GDAL](https://img.shields.io/badge/GDAL-Geospatial-green)

这是一个端到端的遥感影像智能分类项目。相较于基础版本，本项目进行了两项重大升级，显著提升了模型的性能与智能化水平：**引入了多头注意力机制**，并集成了 **Optuna 自动化超参数寻优框架**。

## 项目核心亮点

1.  **多头注意力机制 (Multi-Head Attention)**
    *   在标准的LSTM网络之上，创新性地构建了 **MLP-based** 的多头注意力层。
    *   该机制使模型能够在处理时间序列数据时，**自动学习并聚焦于对分类决策最关键的月份或物候期**，极大地增强了模型的可解释性和分类精度。

2.  **自动化超参数寻优 (Optuna Integration)**
    *   深度集成了业界领先的 **Optuna** 框架，能够对包括网络层数、隐藏层维度、Dropout率、学习率、优化器类型在内的**十余个核心超参数**进行自动化搜索与优化。
    *   采用了先进的 **TPE (Tree-structured Parzen Estimator)** 采样算法和 **MedianPruner** 剪枝策略，实现了高效的贝叶斯优化，能够在有限的试验次数内找到接近最优的参数组合。

3.  **处理类别不平衡 (SMOTE Oversampling)**
    *   在数据预处理阶段，引入 **SMOTE (Synthetic Minority Over-sampling Technique)** 算法，对少数类样本进行合成扩增，有效解决了遥感影像中常见的地物类别不平衡问题，提升了模型的泛化能力。

4.  **专业的工程化实践**
    *   **详尽的日志系统**: 构建了专业的 `TrainingLogger` 类，能够将每一轮的训练配置、性能指标、最佳模型信息及最终的精度评估报告（含Kappa系数、UA、PA等）**自动记录到日志文件中**，便于实验追踪与复现。
    *   **模块化代码结构**: 将训练与预测逻辑完全解耦，并定义了 `EarlyStopping`、`MLPAttention` 等多个可复用类，代码结构清晰，可维护性强。

## 技术栈

*   **核心框架**: PyTorch
*   **超参数优化**: Optuna
*   **数据处理**: Pandas, NumPy, Scikit-learn, Imbalanced-learn (for SMOTE)
*   **地理空间**: GDAL (osgeo)
*   **可视化与输出**: Matplotlib, Tabulate

## 如何运行

#### 1. 克隆并安装依赖

```bash
git clone https://github.com/zwy1502/LSTM-Time-Series-Classification.git
cd LSTM-Time-Series-Classification
# 建议在虚拟环境中安装
pip install torch pandas numpy matplotlib scikit-learn gdal tabulate tqdm optuna imbalanced-learn openpyxl