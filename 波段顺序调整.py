import os
import re
from osgeo import gdal
from tqdm import tqdm


def sort_key(filename):
    match = re.search(r'S1_2020_(\d{2})_(\d)', filename)
    if match:
        month, half = map(int, match.groups())
        return month * 2 + (half - 1)
    return 0


def get_band_names(file_path):
    ds = gdal.Open(file_path)
    band_names = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        band_names.append(band.GetDescription())
    return band_names


def merge_tiffs(input_folder, output_file):
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('.tif.ovr')]
    tif_files.sort(key=sort_key)

    first_file = os.path.join(input_folder, tif_files[0])
    band_names = get_band_names(first_file)

    ds = gdal.Open(first_file)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, ds.RasterXSize, ds.RasterYSize, len(tif_files) * len(band_names),
                           gdal.GDT_Float32)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    total_bands = len(band_names) * len(tif_files)
    with tqdm(total=total_bands, desc="Processing bands", unit="band") as pbar:
        for band_index, band_name in enumerate(band_names):
            for i, file in enumerate(tif_files):
                ds = gdal.Open(os.path.join(input_folder, file))
                band = ds.GetRasterBand(band_index + 1)
                data = band.ReadAsArray()
                out_band_index = band_index * len(tif_files) + i + 1
                out_band = out_ds.GetRasterBand(out_band_index)
                out_band.WriteArray(data)
                out_band.FlushCache()

                file_parts = file.split('_')
                description = f"S1_{file_parts[1]}_{file_parts[2]}_{file_parts[3]}_{band_name}"
                out_band.SetDescription(description)

                pbar.update(1)

    out_ds = None
    print("\n合成完成，输出文件：", output_file)


# 使用示例
input_folder =r"F:\ZWY\JSS12020\ROI_7_0"  # 替换为您的输入文件夹路径
output_file = r"F:\ZWY\JSS12020\S1merged_image_7_0.tif"  # 替换为您想要的输出文件路径

print("开始处理影像...")
merge_tiffs(input_folder, output_file)
print("处理完成！")