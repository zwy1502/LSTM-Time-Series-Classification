import rasterio

# 读取TIF影像
with rasterio.open(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult.tif") as src:
    # 获取元数据
    metadata = src.meta
    # 获取分辨率
    resolution = (metadata['transform'][0], -metadata['transform'][4])  # x和y方向的分辨率

# 打印元数据和分辨率
print("元数据:", metadata)
print("分辨率 (x, y):", resolution)

