import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import swin_t, Swin_T_Weights
from monai.data import NibabelReader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# 设备设置
device = torch.device("cpu")
print(f"Using device: {device}")

class CTScanDataset(Dataset):
    def __init__(self, ct_paths, mask_paths, transform=None):
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.reader = NibabelReader()

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        ct_path = self.ct_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载数据
        ct_img = self.reader.read(ct_path)
        mask_img = self.reader.read(mask_path)

        ct_data = ct_img.get_fdata()
        mask_data = mask_img.get_fdata()

        # 获取有肿瘤的 slice 索引
        tumor_slices = np.where(np.any(mask_data > 0, axis=(0, 1)))[0]

        # 获取患者文件夹名
        patient_name = os.path.basename(os.path.dirname(ct_path))

        return ct_data, mask_data, tumor_slices, patient_name


def preprocess_slice(slice_data):
    """将单个 slice 缩放为 224x224，并转换为 RGB"""
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-6)
    slice_data = (slice_data * 255).astype(np.uint8)

    img = Image.fromarray(slice_data)
    img = img.resize((224, 224), Image.BILINEAR)

    img_tensor = torch.tensor(np.array(img)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [C, H, W]

    return img_tensor


def extract_features(model, dataset, output_dir="features/"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tumor_features.csv")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting Features"):
            ct_data, mask_data, tumor_slices, name = dataset[i]
            patient_full_path = dataset.ct_paths[i]  # 完整路径
            patient_dir = os.path.basename(os.path.dirname(patient_full_path))  # 只取文件夹名

            if len(tumor_slices) == 0:
                continue  # 跳过没有肿瘤的患者

            features_list = []

            for sl in tumor_slices:
                slice_2d = ct_data[:, :, sl]
                img_tensor = preprocess_slice(slice_2d)
                img_tensor = img_tensor.unsqueeze(0).to(device)

                features = model(img_tensor).squeeze().cpu().numpy()
                features_list.append(features)

            # 对所有肿瘤切片的特征取平均
            avg_features = np.mean(features_list, axis=0)

            # 构造 DataFrame 行：只保留 patient 文件夹名
            feature_dict = {
                "patient": patient_dir,  # <-- 这里改了！只保留文件夹名
                **{f"feat_{j}": avg_features[j] for j in range(avg_features.shape[0])}
            }

            df = pd.DataFrame([feature_dict])

            # 写入 CSV：第一次创建带 header，之后追加不带 header
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False, mode='w', header=True)
            else:
                df.to_csv(csv_path, index=False, mode='a', header=False)

            print(f"Features saved for patient: {patient_dir}")


def load_data(data_dir):
    data_dir = Path(data_dir)
    ct_paths = []
    mask_paths = []

    for patient in sorted(data_dir.iterdir()):
        ct_file = os.path.join(patient, "CT_image.nii.gz")
        mask_file = os.path.join(patient, "binary_mask.nii.gz")
        if os.path.exists(ct_file) and os.path.exists(mask_file):
            ct_paths.append(ct_file)
            mask_paths.append(mask_file)

    print(f"Found {len(ct_paths)} patients with both CT and mask.")
    return CTScanDataset(ct_paths, mask_paths)


def main(data_dir="/root/lanyun-tmp/processed/", output_dir="features/"):
    # 加载数据集
    dataset = load_data(data_dir)

    # 加载预训练 Swin Transformer 模型
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类头
    model = model.to(device)

    # 开始提取特征并保存
    extract_features(model, dataset, output_dir)


if __name__ == "__main__":
    main()