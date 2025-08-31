import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

def reproject_raster(input_raster, output_raster, dst_crs):
    """重投影栅格数据"""
    with rasterio.open(input_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def calculate_area_per_class(raster_path, shapefile, pixel_area):
    """计算每个类别的面积"""
    with rasterio.open(raster_path) as src:
        # 使用shapefile裁剪栅格数据
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        data = out_image[0]

        # 过滤掉NoData值和背景值(0)
        valid_data = data[(data != src.nodata) & (data != 0)]

        # 统计每个类别的像素数量
        unique, counts = np.unique(valid_data, return_counts=True)

        # 计算每个类别的面积（平方公里）
        class_areas = dict(zip(unique, counts * pixel_area / 1_000_000))  # 转换为平方公里
        return class_areas

def main():
    # 输入文件路径
    # lj= r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult众数滤波边界清理0_5最终.tif"
    # lj=r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\S2S1预测结果\S1S2MergedResult_edit_众滤_边清_last.tif"
    # lj=r"D:\research\fenlei\dalunwen\JS\JS2020_9_15_2023\S2S1预测结果\重新编辑结果重分类众滤边清lastnew.tif"
    lj=r"D:\research\fenlei\dalunwen\JS\JS_2017\S2S1预测结果\S1S2MergedResult2017_adjustcfl_众滤边青.tif"
    input_raster = lj
    shapefile_paths = [
        r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\连云港市计算面积.shp",
        r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\南通市计算面积.shp",
        r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\盐城市计算面积.shp"
    ]

    # 定义目标坐标系
    dst_crs = 'EPSG:32651'  # WGS_1984_UTM_Zone_51N

    # 定义类别名称
    class_names = {
        1: "互花米草",
        2: "碱蓬",
        3: "芦苇",
        4: "水体",
        5: "非湿地"
    }

    # 输出重投影后的栅格路径
    output_raster = os.path.splitext(input_raster)[0] + '_reprojected.tif'

    # 检查重投影后的栅格是否存在
    if not os.path.exists(output_raster):
        print("正在重投影栅格数据...")
        reproject_raster(input_raster, output_raster, dst_crs)
    else:
        print("重投影后的栅格数据已存在。")

    # 打开重投影后的栅格数据
    with rasterio.open(output_raster) as src:
        # 计算像素面积（单位：平方米）
        pixel_size_x = src.transform[0]
        pixel_size_y = -src.transform[4]
        pixel_area = pixel_size_x * pixel_size_y
        total_area = []#
        # 逐个处理每个shapefile
        for shp_path in shapefile_paths:
            print(f"正在处理 {shp_path}...")
            shapefile = gpd.read_file(shp_path)
            # 重投影shapefile
            shapefile = shapefile.to_crs(dst_crs)
            # 计算每个类别的面积
            class_areas = calculate_area_per_class(output_raster, shapefile, pixel_area)
            # 输出结果
            print(f"{os.path.basename(shp_path)} 的分类面积（平方公里）：")
            for cls, area in class_areas.items():
                if cls in class_names:
                    print(f"{class_names[cls]}: {area:.4f} 平方公里")
                    if cls==1:
                        total_area.append(area)
            print("-----------------------------")
        print(f'互花米草总面积为{sum(total_area)}平方公里')

if __name__ == "__main__":
    main()