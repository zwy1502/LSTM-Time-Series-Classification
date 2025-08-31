import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 定义各月的生产者精度和用户精度数据
months = list(range(1, 13))
producer_accuracy = [
    0.8105,  # 一月
    0.9175,  # 二月
    0.9591,  # 三月
    0.9617,  # 四月
    0.9144,  # 五月
    0.9024,  # 六月
    0.8884,  # 七月
    0.7636,  # 八月
    0.9139,  # 九月
    0.9833,  # 十月
    0.9552,  # 十一月
    0.9744   # 十二月
]

user_accuracy = [
    0.8652,  # 一月
    0.9418,  # 二月
    0.9378,  # 三月
    0.9670,  # 四月
    0.9661,  # 五月
    0.8852,  # 六月
    0.8721,  # 七月
    0.6914,  # 八月
    0.9409,  # 九月
    0.9514,  # 十月
    0.9861,  # 十一月
    1.0      # 十二月
]

# 创建精度变化曲线图
plt.figure(figsize=(10, 6), dpi=120)
plt.plot(months, producer_accuracy, label='生产者精度', marker='o', color='blue')
plt.plot(months, user_accuracy, label='用户精度', marker='o', color='red')

# 添加图例和轴标签
plt.legend(fontsize=14)
plt.title('互花米草分类精度随月份变化', fontdict={'size': 18})
plt.xlabel('月份', fontdict={'size': 18})
plt.ylabel('精度', fontdict={'size': 18})

# 设置x轴的刻度为每个月，调整刻度文字的大小
plt.xticks(months, fontsize=18)
plt.yticks(fontsize=18)

# 显示网格线
plt.grid(True)

# 显示图表
plt.show()
