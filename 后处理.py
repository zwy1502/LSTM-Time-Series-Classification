# 明白了，具体来说，你的需求是：
#
# 读取最初的TIF分类结果。
# 对于每个像素，如果其值为0，则检查该像素是否位于后处理的Shapefile内。
# 如果在Shapefile内，保持其值为0。
# 如果不在Shapefile内，将其值改为4。
# 对于所有其他像素（值不是0的），保持不变。



import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask

# 读取分类结果的TIF影像
with rasterio.open(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult原始.tif") as src:
    image = src.read(1)  # 读取第一波段
    transform = src.transform
    profile = src.profile  # 获取影像的配置

# 读取后处理的Shapefile
shp = gpd.read_file(r'D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S2S1预测结果hhmc.shp')

# 创建掩膜，标识出Shapefile内的区域
mask = geometry_mask([geom for geom in shp.geometry], transform=transform, invert=True, out_shape=image.shape)

# 处理分类结果
# 只对值为0的像素进行处理，保留值为0的像素，并将不在Shapefile内的改为4
output_image = image.copy()  # 创建一个输出影像副本
output_image[(image == 0) & mask] = 4  # 仅在值为0且不在mask中的像素位置更新为4

# 保存处理后的影像
with rasterio.open(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult修改hhmc.tif", 'w', **profile) as dst:
    dst.write(output_image.astype(profile['dtype']), 1)
