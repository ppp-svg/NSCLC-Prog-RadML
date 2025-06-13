import os
import glob
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pydicom import dcmread
from tqdm import tqdm
from pathlib import Path
import logging
# 禁用 radiomics 的日志输出
logger = logging.getLogger('radiomics')
logger.setLevel(logging.WARNING)  # 只显示 WARNING 及以上级别的日志

# 设置特征提取参数
settings = {}
settings['binWidth'] = 25  # 灰度离散化 bin 宽度
settings['sigma'] = [1, 3, 5]  # LoG 滤波器使用的 sigma 值
settings['resampledPixelSpacing'] = None  # 不进行重采样
settings['interpolator'] = 'sitkBSpline'  # 插值方式
settings['verbose'] = False  # 不显示调试信息
settings['normalize'] = True  # 标准化图像强度
settings['normalizeScale'] = 1  # 标准化缩放因子
settings['voxelArrayShift'] = 0  # 强度偏移量

# ROI 名称匹配正则表达式
TARGET_ROI_PATTERN = re.compile(r'\b\w*gtv\w*\b', re.IGNORECASE)

# 数据路径和输出路径
DATA_DIR = '../manifest-1603198545583/NSCLC-Radiomics/'
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'radiomic_features.csv')

# 实例化 Radiomics 特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

# 启用图像滤波器
extractor.enableImageTypeByName('Original')
extractor.enableImageTypeByName('Wavelet')
extractor.enableImageTypeByName('LoG')

# 启用特征类别
extractor.disableAllFeatures()  # 先禁用所有默认启用的特征
extractor.enableFeatureClassByName('firstorder')  # 一阶统计量
extractor.enableFeatureClassByName('shape')  # 形态学特征
extractor.enableFeatureClassByName('glcm')  # 灰度共生矩阵

def extract_roi_contours_from_rtstruct(rtstruct_path: str, pattern: re.Pattern) -> list:
    """
    加载 RTSTRUCT 并提取符合正则表达式的 ROI 轮廓点
    """
    rtstruct = dcmread(rtstruct_path)
    contours = []
    for roi_info, contour_info in zip(rtstruct.StructureSetROISequence, rtstruct.ROIContourSequence):
        if pattern.match(roi_info.ROIName):
            print(f"发现 ROI: {roi_info.ROIName}")
            for sequence in contour_info.ContourSequence:
                points = np.array(sequence.ContourData).reshape(-1, 3)
                contours.append(points)
    return contours


def generate_binary_mask(contours: list, ct_image: sitk.Image) -> sitk.Image:
    """
    根据 CT 图像信息创建一个与之对齐的 3D 二值掩码图像
    """
    origin = np.array(ct_image.GetOrigin())
    spacing = np.array(ct_image.GetSpacing())
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)

    mask_shape = sitk.GetArrayFromImage(ct_image).shape
    mask_array = np.zeros(mask_shape, dtype=np.uint8)

    for point_list in contours:
        for point in point_list:
            physical_point = np.array(point)
            voxel_index = np.round(np.linalg.inv(direction).dot((physical_point - origin) / spacing)).astype(int)
            x, y, z = voxel_index

            if 0 <= z < mask_array.shape[0] and 0 <= y < mask_array.shape[1] and 0 <= x < mask_array.shape[2]:
                mask_array[z, y, x] = 1

    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.SetOrigin(ct_image.GetOrigin())
    mask_image.SetSpacing(ct_image.GetSpacing())
    mask_image.SetDirection(ct_image.GetDirection())

    return mask_image

if __name__ == "__main__":
    import logging
    logger = logging.getLogger('radiomics')
    logger.setLevel(logging.WARNING)

    print("开始批量处理病人数据...")

    patient_dirs = glob.glob(os.path.join(DATA_DIR, '*/'))

    # 准备输出 CSV 文件
    output_df_path = OUTPUT_CSV_PATH
    file_exists = os.path.isfile(output_df_path)

    with open(output_df_path, mode='a', newline='') as f:
        for idx, patient_dir in enumerate(tqdm(patient_dirs, total=len(patient_dirs))):
            patient_id = os.path.basename(patient_dir.rstrip('/'))
            print(f"[Processing] Patient: {patient_id}")

            try:
                file_list = glob.glob(f"{patient_dir}/*/*")
                sorted_files = sorted(
                    [file for file in file_list if (file.split('/')[-1][0].isdigit()) and not ('Segmentation' in file)],
                    key=lambda x: float(x.split('/')[-1].split('.')[0])
                )

                if len(sorted_files) < 2:
                    raise FileNotFoundError(f"CT or RTSTRUCT 文件缺失：{patient_dir}")

                ct_series_path = sorted_files[0]
                rtstruct_path = glob.glob(f"{sorted_files[1]}/*")[0]

                output_folder = Path(ct_series_path).parent
                ct_output_path = os.path.join(output_folder, "CT_image.nii.gz")
                roi_output_path = os.path.join(output_folder, "binary_mask.nii.gz")

                # 检查是否已经存在 nii.gz 文件
                if os.path.exists(ct_output_path) and os.path.exists(roi_output_path):
                    print(f"[已存在] 跳过转换与掩码生成，使用现有文件：{patient_id}")
                else:
                    # Step 1: 将 DICOM CT 序列保存为 NIfTI 文件
                    reader = sitk.ImageSeriesReader()
                    dicom_files = reader.GetGDCMSeriesFileNames(ct_series_path)
                    reader.SetFileNames(dicom_files)
                    ct_image = reader.Execute()
                    sitk.WriteImage(ct_image, ct_output_path)

                    # Step 2: 提取 RTSTRUCT 轮廓
                    contours = extract_roi_contours_from_rtstruct(rtstruct_path, TARGET_ROI_PATTERN)

                    if not contours:
                        print(f"警告：{patient_id} 没有找到匹配的 ROI，使用全 0 特征。")
                        features = {key: 0 for key in extractor.featureExtractorClasses}
                        features['Patient'] = patient_id
                    else:
                        # Step 3: 创建二值掩码
                        binary_mask = generate_binary_mask(contours, ct_image)
                        sitk.WriteImage(binary_mask, roi_output_path)

                # Step 4: 提取放射组学特征（无论是否跳过前面步骤，都执行）
                features = extractor.execute(ct_output_path, roi_output_path)

                # 过滤掉'diagnostics'开头的特征名称，只保留实际计算的特征
                filtered_features = {
                    key: float(value) if isinstance(value, (int, float)) else str(value)
                    for key, value in features.items()
                    if not key.startswith("diagnostics_")
                }

                # 添加图像标识符
                filtered_features['Patient'] = patient_id

                # 构造 DataFrame 行，并强制 'Patient' 列排在最前面
                cols = ['Patient'] + [col for col in filtered_features if col != 'Patient']
                df_row = pd.DataFrame([filtered_features])[cols]
                if idx == 0 and not file_exists:
                    df_row.to_csv(f, index=False, header=True)
                else:
                    df_row.to_csv(f, index=False, header=False)

            except Exception as e:
                print(f"[Error] 处理病人失败：{patient_id}, 错误：{e}")
                continue

    print(f"所有病人特征提取完成，并已追加保存至：{output_df_path}")