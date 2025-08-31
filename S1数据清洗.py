# 请你发送一段代码给我，我的要求是这一段代码会对这个数据进行数据清洗，
# 我的要求是凡是VV或者VH小于-35的数据这一行就去掉，凡是有缺失值的行也去掉，
# 然后输出一个新的CSV文件，然后我在本地跑完你发的代码之后我再发给你帮我检查一下。是不是达到了我目前的需求
# 下面代码功能是
# 删除VV或VH列中小于-35的行。
# 删除不在[-1, 1]范围内的SAR_NDVI列的行。
# 删除包含缺失值的行。

# import pandas as pd
#
# # 读取数据
# file_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S1_2023时序训练数据.csv"  # 请替换为您的文件路径
# # file_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\江苏省S1和S2训练数据\S1_2020时序训练数据原始.csv"  # 请替换为您的文件路径
# data = pd.read_csv(file_path)
#
# # 删除 VV 或 VH 列中小于 -35 的行
# vv_vh_columns = [col for col in data.columns if '_VV' in col or '_VH' in col]
# data_cleaned = data[(data[vv_vh_columns] > -35).all(axis=1)]
#
# # 删除 SAR_NDVI 列中不在 [-1, 1] 范围内的行
# sar_ndvi_columns = [col for col in data.columns if 'SAR_NDVI' in col]
# data_cleaned = data_cleaned[(data_cleaned[sar_ndvi_columns] >= -1).all(axis=1) &
#                             (data_cleaned[sar_ndvi_columns] <= 1).all(axis=1)]
#
# # 删除包含缺失值的行
# data_cleaned = data_cleaned.dropna()
#
# # 输出清洗后的数据到新的 CSV 文件
# output_file_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023S1S2训练样本\S1_2023时序训练数据清洗后.csv"
# data_cleaned.to_csv(output_file_path, index=False)
#
# print(f"数据清洗完成，结果已保存到 {output_file_path}")





import pandas as pd


# 读取数据
file_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\2017S1S2训练样本\S1_2017时序训练数据月度.csv"
data = pd.read_csv(file_path)

print(f"原始数据行数: {len(data)}")

# 删除 VV 或 VH 列中小于 -35 的行
vv_vh_columns = [col for col in data.columns if '_VV' in col or '_VH' in col]
mask_vv_vh = (data[vv_vh_columns] > -35).all(axis=1)
rows_removed_vv_vh = len(data) - mask_vv_vh.sum()
data_cleaned = data[mask_vv_vh]

print(f"删除 VV 或 VH < -35 的行数: {rows_removed_vv_vh}")

# 删除 SAR_NDVI 列中不在 [-1, 1] 范围内的行
sar_ndvi_columns = [col for col in data.columns if 'SAR_NDVI' in col]
mask_sar_ndvi = (data_cleaned[sar_ndvi_columns] >= -1).all(axis=1) & (data_cleaned[sar_ndvi_columns] <= 1).all(axis=1)
rows_removed_sar_ndvi = len(data_cleaned) - mask_sar_ndvi.sum()
data_cleaned = data_cleaned[mask_sar_ndvi]

print(f"删除 SAR_NDVI 不在 [-1, 1] 范围内的行数: {rows_removed_sar_ndvi}")

# 删除包含缺失值的行
rows_with_na = data_cleaned.isna().any(axis=1).sum()
data_cleaned = data_cleaned.dropna()

print(f"删除包含缺失值的行数: {rows_with_na}")

# 删除完全相同的时间序列
time_series_columns = vv_vh_columns + sar_ndvi_columns
data_cleaned['time_series_hash'] = data_cleaned[time_series_columns].apply(lambda x: hash(tuple(x)), axis=1)
duplicate_rows = data_cleaned.duplicated(subset=['time_series_hash'], keep='first')
rows_removed_duplicates = duplicate_rows.sum()
data_cleaned = data_cleaned[~duplicate_rows].drop('time_series_hash', axis=1)

print(f"删除完全相同时间序列的行数: {rows_removed_duplicates}")

# 输出清洗后的数据到新的 CSV 文件
output_file_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\2017S1S2训练样本\S1_2017时序训练数据月度清洗后.csv"
data_cleaned.to_csv(output_file_path, index=False)

print(f"\n数据清洗完成，结果已保存到 {output_file_path}")
print(f"清洗后的数据行数: {len(data_cleaned)}")
print(f"总共删除的行数: {len(data) - len(data_cleaned)}")














