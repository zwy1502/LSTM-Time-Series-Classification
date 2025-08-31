#
# 你好，我想让你用代码帮我完成一个需求，主要是合并两个分类后的tif影像结果，首先我有一个光学影像分类的结果，
# 地址是"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S2_2020flresult.tif"，
# 一个是SAR分类后的结果地址是："D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S12020LSTM.tif"，
# 我再讲一下这两个数据的基本情况，两个数据的投影是一致的，行列数也是一致的，（这一点你可以在代码中添加打印检查），
# 两个影像的投影是4326，行列数是33713, 39029，光学分类影像结果的像素值有这些，-9999，0，1，2，3，4.分别代表缺失值，
# 第一类，第二类依次类推总共五类。SAR的像素值为0，1，2，3，4，5代表缺失值，第一类到第五类。现在需要你帮我写一段python代码帮我实现以下需求，
# 我想对光学影像的未分类（缺失值）的部分用SAR数据填充，也就是你看到光学为-9999且对应SAR的像素不为0，则用SAR的值填充，
# 但是由于SAR的值比光学多1，所以你填充SAR还要对像素值减一，下面我举个例子说明过程，比如发现某一个像素光学为缺失值-9999，
# 然后对应SAR不是0（是非缺失值），那么用SAR这个像素的结果填充但是类别要-1因为SAR与光学的类别值并不对应。然后如果两个都缺失比如-9999光学，
# SAR是0，那么不做任何改变，维持光学-9999，
# 可以做到的话，请发给我代码，还有请你用一个进度条让我监测进度，最后请你生成一个新的tif，过程不要对原两个文件修改而是生成一个新的tif文件。

import numpy as np
import rasterio
from tqdm import tqdm

# 输入文件路径
optical_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\S2S1预测结果\S2_2017flresult.tif"
sar_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\S2S1预测结果\S1yuce.tif"
output_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\S2S1预测结果\S1S2MergedResult2017.tif"

# 读取光学影像
with rasterio.open(optical_path) as optical_src:
    optical_data = optical_src.read(1)
    optical_meta = optical_src.meta

# 读取SAR影像
with rasterio.open(sar_path) as sar_src:
    sar_data = sar_src.read(1)


print(f"光学影像投影: {optical_meta['crs']}")
print(f"SAR影像投影: {sar_src.meta['crs']}")
print(f"光学影像形状: {optical_data.shape}")
print(f"SAR影像形状: {sar_data.shape}")
# 检查投影和行列数
assert optical_meta['crs'] == sar_src.meta['crs'], "投影不一致"
assert optical_data.shape == sar_data.shape, "行列数不一致"

# 创建合并影像数组
merged_data = optical_data.copy()

# 进度条
total_pixels = optical_data.size
for idx in tqdm(range(total_pixels), desc="合并进度"):
    if optical_data.flat[idx] == -9999 and sar_data.flat[idx] != 0:
        merged_data.flat[idx] = sar_data.flat[idx] - 1

# 更新元数据
optical_meta.update({
    'dtype': 'int32',
    'count': 1,
    # 'compress': 'lzw'  # 可选择压缩方式
})

# 写入合并结果到新的TIF文件
with rasterio.open(output_path, 'w', **optical_meta) as dst:
    dst.write(merged_data, 1)

print("合并完成，结果已保存到:", output_path)
