import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.mask import mask
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, f1_score
import warnings

warnings.filterwarnings('ignore')


def calculate_area_statistics(raster_image, raster_transform):
    """
    计算区域面积统计
    """
    # 计算每个像素的面积（平方米）
    pixel_width = abs(raster_transform[0])
    pixel_height = abs(raster_transform[4])
    pixel_area = pixel_width * pixel_height

    # 计算总像素数和缺失值像素数
    total_pixels = raster_image.size
    missing_pixels = np.sum(raster_image == -9999)

    # 计算面积
    total_area = total_pixels * pixel_area
    missing_area = missing_pixels * pixel_area

    return total_area, missing_area


def calculate_metrics(true_labels, pred_labels, class_labels, region_name=""):
    """
    计算并打印各种分类精度指标
    """
    # 计算混淆矩阵
    conf_mat = confusion_matrix(true_labels, pred_labels, labels=class_labels)

    # 计算Kappa系数
    kappa = cohen_kappa_score(true_labels, pred_labels, labels=class_labels)

    # 计算生产者精度和用户精度
    producer_accuracy = np.diag(conf_mat) / conf_mat.sum(axis=1)
    user_accuracy = np.diag(conf_mat) / conf_mat.sum(axis=0)

    # 计算总体精度
    overall_accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    # 计算每个类别的F1分数
    f1_scores = f1_score(true_labels, pred_labels, labels=class_labels, average=None)

    # 计算宏平均F1分数
    macro_f1 = f1_score(true_labels, pred_labels, labels=class_labels, average='macro')

    # 打印结果
    print(f"\n{'=' * 20} {region_name} 精度评价 {'=' * 20}")
    print('\n混淆矩阵:')
    print(conf_mat)
    print('\nKappa系数:', kappa)

    class_names = ['互花米草', '碱蓬', '芦苇', '水体', '非湿地']
    print('\n各类别精度:')
    for i, name in enumerate(class_names):
        print(f"{name}:")
        print(f"  生产者精度: {producer_accuracy[i]:.4f}")
        print(f"  用户精度: {user_accuracy[i]:.4f}")
        print(f"  F1分数: {f1_scores[i]:.4f}")

    print('\n总体精度:', overall_accuracy)
    print('宏平均F1分数:', macro_f1)

    return conf_mat, kappa, overall_accuracy, macro_f1, true_labels, pred_labels


def process_region(raster_path, truth_shp_path, region_shp_path, region_name):
    """
    处理单个区域的分类精度评估
    """
    # 读取栅格数据
    with rasterio.open(raster_path) as src:
        # 读取区域shp并裁剪栅格
        region_gdf = gpd.read_file(region_shp_path)
        region_geom = region_gdf.geometry.values
        raster_crop, raster_transform = mask(src, region_geom, crop=True)
        raster_meta = src.meta.copy()
        raster_meta.update({
            "height": raster_crop.shape[1],
            "width": raster_crop.shape[2],
            "transform": raster_transform
        })
        raster_image = raster_crop[0]  # 获取第一个波段

        # 计算面积统计
        total_area, missing_area = calculate_area_statistics(raster_image, raster_transform)

        print(f"\n{'-' * 20} {region_name} 面积统计 {'-' * 20}")
        print(f"总面积: {total_area / 1000000:.2f} 平方公里")
        print(f"缺失值面积: {missing_area / 1000000:.2f} 平方公里")
        print(f"缺失值面积占比: {(missing_area / total_area) * 100:.2f}%")

    # 读取真实标签数据
    truth_gdf = gpd.read_file(truth_shp_path)
    truth_gdf = truth_gdf.clip(region_gdf.geometry.iloc[0])

    # 栅格化真实标签
    shapes = ((geom, value) for geom, value in zip(truth_gdf.geometry, truth_gdf['class']))
    label_raster = rasterize(
        shapes=shapes,
        out_shape=(raster_meta['height'], raster_meta['width']),
        transform=raster_meta['transform'],
        fill=-9999,
        dtype=raster_meta['dtype']
    )

    # 统计有效样本中的缺失值
    valid_samples_mask = (label_raster >= 1) & (label_raster <= 5)  # 有效样本掩膜
    missing_value_mask = (raster_image == -9999)  # 缺失值掩膜
    missing_count = np.sum(missing_value_mask & valid_samples_mask)  # 有效样本中的缺失值数量
    total_valid_samples = np.sum(valid_samples_mask)  # 总有效样本数

    print(f"\n{'-' * 20} {region_name} 缺失值统计 {'-' * 20}")
    print(f"总有效样本数: {total_valid_samples}")
    print(f"其中缺失值数量: {missing_count}")
    print(f"缺失值占比: {missing_count / total_valid_samples * 100:.2f}%")

    # 创建掩膜（排除无效值和背景）
    valid_mask = (label_raster >= 1) & (label_raster <= 5) & (raster_image >= 0) & (raster_image <= 4)

    # 提取有效的真实值和预测值
    true_labels = label_raster[valid_mask].flatten()
    pred_labels = raster_image[valid_mask].flatten()

    # 将预测标签加1以匹配真实标签的编码
    pred_labels = pred_labels + 1

    # 定义类别标签
    class_labels = [1, 2, 3, 4, 5]

    # 计算并返回精度指标
    return calculate_metrics(true_labels, pred_labels, class_labels, region_name)

def main():
    # 设置文件路径
    raster_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\S2S1预测结果\S2_2020flresult.tif"
    truth_shp_path = r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\2020JSybmerge.shp"

    # 区域shp文件路径
    region_paths = {
        "连云港": r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\lygstuarea.shp",
        "南通": r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\ntstuarea.shp",
        "盐城": r"D:\research\fenlei\dalunwen\JS\JS2020_9_15\YCstuarea.shp"
    }

    # 存储所有区域的结果
    all_true_labels = []
    all_pred_labels = []

    # 处理每个区域
    for region_name, region_path in region_paths.items():
        try:
            # 处理单个区域并保存结果
            conf_mat, kappa, oa, f1, true_labels, pred_labels = process_region(
                raster_path, truth_shp_path, region_path, region_name
            )

            # 将该区域的标签添加到总体评估中
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

        except Exception as e:
            print(f"\n处理{region_name}区域时发生错误: {str(e)}")

    # 计算总体精度评价
    if all_true_labels and all_pred_labels:
        print("\n\n" + "=" * 20 + " 三个区域总体精度评价 " + "=" * 20)
        print("注：这是将三个区域作为一个整体进行的精度评价")
        calculate_metrics(
            np.array(all_true_labels),
            np.array(all_pred_labels),
            [1, 2, 3, 4, 5],
            ""
        )


if __name__ == "__main__":
    main()