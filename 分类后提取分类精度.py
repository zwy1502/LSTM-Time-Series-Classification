import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

# 设置文件路径
# raster_path = r'D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult众数滤波边界清理0_5最终.tif'
# raster_path = r'D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S12020LSTM.tif'
# shapefile_path = r'D:\research\fenlei\dalunwen\JS\JS2020_9_15\2020JSybmerge.shp'



raster_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\S2S1预测结果\S1S2MergedResult2017_adjustcfl_众滤边青.tif"

shapefile_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\2017JSybmerge.shp"
# 读取影像数据
with rasterio.open(raster_path) as src:
    raster_image = src.read(1)  # 读取第一波段
    raster_meta = src.meta  # 获取元数据

# 读取矢量样本数据
gdf = gpd.read_file(shapefile_path)

# 确保投影一致
if gdf.crs != raster_meta['crs']:
    gdf = gdf.to_crs(raster_meta['crs'])

# 将矢量样本数据栅格化
# 使用样本数据中的类别字段 'class'
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['class']))

# 创建与影像尺寸相同的空数组
label_raster = np.zeros((raster_meta['height'], raster_meta['width']), dtype=raster_meta['dtype'])

# 栅格化样本数据
label_raster = rasterize(
    shapes=shapes,
    out_shape=(raster_meta['height'], raster_meta['width']),
    transform=raster_meta['transform'],
    fill=0,  # 背景值为0
    dtype=raster_meta['dtype']
)

# 创建掩膜，过滤背景值（只保留类别为1到5的像素）
# 样本数据的有效类别为1到5
sample_mask = (label_raster >= 1) & (label_raster <= 5)
# 影像数据的有效类别为1到5，背景值为6或其他值
raster_mask = (raster_image >= 1) & (raster_image <= 5)

# 综合掩膜，只保留两者都为有效类别的像素
mask = sample_mask & raster_mask

# 提取有效的真实值和预测值
true_labels = label_raster[mask].flatten()
pred_labels = raster_image[mask].flatten()

# 定义类别标签
labels = [1, 2, 3, 4, 5]

# 计算混淆矩阵
conf_mat = confusion_matrix(true_labels, pred_labels, labels=labels)
print('混淆矩阵:')
print(conf_mat)

# 计算Kappa系数
kappa = cohen_kappa_score(true_labels, pred_labels, labels=labels)
print('Kappa系数:', kappa)

# 计算生产者精度和用户精度
producer_accuracy = np.diag(conf_mat) / conf_mat.sum(axis=1)
user_accuracy = np.diag(conf_mat) / conf_mat.sum(axis=0)

print('生产者精度:', producer_accuracy)
print('用户精度:', user_accuracy)

# 计算总体精度
overall_accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
print('总体精度:', overall_accuracy)

# 输出分类报告
report = classification_report(true_labels, pred_labels, labels=labels)
print('分类报告:')
print(report)
