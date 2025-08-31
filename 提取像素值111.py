import os
import glob
import geopandas as gpd
import rasterio
import pandas as pd
from shapely.geometry import Point, box
from tqdm import tqdm  # 导入tqdm库

def extract_pixel_values(shp_path, image_folder, output_csv):
    # 读取点矢量数据
    points_gdf = gpd.read_file(shp_path)
    points_gdf = points_gdf.to_crs('EPSG:4326')  # 确保投影一致

    # 创建一个列表来保存结果
    data_list = []

    # 获取所有影像文件的路径
    tif_files = glob.glob(os.path.join(image_folder, '**', '*.tif'), recursive=True)

    # 初始化外部进度条
    with tqdm(total=len(tif_files), desc="Processing images") as pbar:
        # 遍历所有影像文件
        for tif_file in tif_files:
            # 更新进度条
            pbar.set_postfix_str(f"Current image: {os.path.basename(tif_file)}")
            with rasterio.open(tif_file) as src:

                # 检查影像投影是否为EPSG:4326
                if src.crs != 'EPSG:4326':
                    print(f"警告：影像 {tif_file} 的投影不是 EPSG:4326，当前投影为 {src.crs}")
                # 获取影像的边界
                bbox = src.bounds
                bbox_geom = box(*bbox)
                bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs=src.crs)

                # 将点的坐标系转换为影像的坐标系
                points_in_crs = points_gdf.to_crs(src.crs)

                # 找到位于当前影像范围内的点
                points_in_image = gpd.sjoin(points_in_crs, bbox_gdf, how='inner', predicate='intersects')

                if points_in_image.empty:
                    pbar.update(1)
                    continue  # 如果没有点位于当前影像，跳过

                # 获取影像的波段名称
                band_names = src.descriptions
                if not any(band_names):
                    # 如果没有波段名称，生成默认的波段名称
                    band_names = [f'Band_{i+1}' for i in range(src.count)]
                else:
                    # 如果部分波段没有名称，填充默认名称
                    band_names = [name if name else f'Band_{i+1}' for i, name in enumerate(band_names)]

                # 初始化内部进度条
                with tqdm(total=len(points_in_image), desc=f"Processing points in {os.path.basename(tif_file)}") as inner_pbar:
                    # 提取像素值
                    for idx, row in points_in_image.iterrows():
                        point = row['geometry']
                        class_label = row['class']

                        # 获取点的像素行列号
                        try:
                            row_col = src.index(point.x, point.y)
                        except ValueError:
                            # 如果点在影像范围外，跳过
                            inner_pbar.update(1)
                            continue

                        # 检查像素行列号是否在影像范围内
                        if not (0 <= row_col[0] < src.height and 0 <= row_col[1] < src.width):
                            inner_pbar.update(1)
                            continue

                        # 读取该像素的所有波段值
                        pixel_values = src.read(window=((row_col[0], row_col[0]+1), (row_col[1], row_col[1]+1)))
                        pixel_values = pixel_values.flatten()

                        # 创建一个包含像素值和类标签的字典
                        data = {'class': class_label}
                        for band_idx, value in enumerate(pixel_values):
                            band_name = band_names[band_idx]
                            data[band_name] = value

                        # 将数据添加到列表
                        data_list.append(data)

                        # 更新内部进度条
                        inner_pbar.update(1)

                # 更新外部进度条
                pbar.update(1)

    # 将列表转换为DataFrame
    results_df = pd.DataFrame(data_list)

    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    # 保存结果到CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# 使用示例
# shp_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\2020JSybmerge.shp"
# shp_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\2023JSybmerge.shp"
shp_path = r"D:\research\fenlei\dalunwen\JS\JS_2017\2017JSybmerge.shp"
# image_folder = r'F:\ZWY\JSS12020_120bands'
# image_folder = r'F:\ZWY\JS2023_2'
image_folder = r'F:\影像'
output_csv = r'D:\research\fenlei\dalunwen\JS\JS_2017\2017S1S2训练样本\S1_2017时序训练数据月度.csv'

extract_pixel_values(shp_path, image_folder, output_csv)



