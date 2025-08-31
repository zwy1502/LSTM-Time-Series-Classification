import geopandas as gpd
import rasterio

# 读取shapefile
shapefile_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\2020JSybmerge.shp" # 请替换为您的shapefile路径
shapefile = gpd.read_file(shapefile_path)
shapefile_crs = shapefile.crs

# 读取影像
image_path = r"F:\ZWY\JSS12020_120bands\ROI_4_0\S12020_ROI_4_0-0000000000-0000000000-001.tif"  # 请替换为您的影像文件路径
with rasterio.open(image_path) as src:
    image_crs = src.crs

# 比较CRS
if shapefile_crs == image_crs:
    print("shapefile和影像具有相同的投影。")
else:
    print("shapefile和影像的投影不同。")
    print("shapefile的CRS:", shapefile_crs)
    print("影像的CRS:", image_crs)
