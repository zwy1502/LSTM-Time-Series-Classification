import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

# 读取TIF影像
with rasterio.open(r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S1S2MergedResult众数滤波边界清理.tif") as src:
    image = src.read(1)  # 读取第一波段
    transform = src.transform
    crs = src.crs  # 获取原影像的CRS

# 创建掩膜，找出像素值为0的区域
mask = image == 0

# 矢量化处理
shapes = rasterio.features.shapes(image, mask=mask, transform=transform)

# 创建GeoDataFrame，保留像素值为0的形状
geometries = [shape(geom) for geom, value in shapes if value == 0]  # 只保留值为0的形状
gdf = gpd.GeoDataFrame({'geometry': geometries})

# 设置CRS
gdf.crs = crs

# 保存为Shapefile
gdf.to_file(r'D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S2S1预测结果hhmc.shp',
            driver='ESRI Shapefile')
