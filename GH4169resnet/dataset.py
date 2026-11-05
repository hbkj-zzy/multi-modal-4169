from torchvision import transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MultiModalDataset(Dataset):
    # 修改 __init__，接收 DataFrame 切片
    def __init__(self, image_dir, dataframe_slice, v_delta_column_index, transform=None, scaler=None, label_min=None, label_max=None): # 新增 v_delta_column_index 参数
        self.image_dir = image_dir
        self.df = dataframe_slice # 直接使用传入的 DataFrame 切片
        if not self.df.empty and self.df.columns.size > 0:
            self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        self.transform = transform # 这个 transform 现在将包含所有图片处理和增强逻辑
        self.scaler = scaler # 接收训练集计算出的 scaler
        self.label_min = label_min # 接收训练集计算出的 label_min
        self.label_max = label_max # 接收训练集计算出的 label_max
        self.v_delta_column_index = v_delta_column_index # 保存 V_delta 列索引

        # 提取并保存原始 V_delta 数据 (未标准化)
        if not self.df.empty:
             # 确保 V_delta 列存在且是数字类型
             if self.v_delta_column_index is not None and self.v_delta_column_index < self.df.shape[1]:
                 v_delta_data = self.df.iloc[:, self.v_delta_column_index].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float32')
                 self.raw_v_delta_data = torch.tensor(v_delta_data, dtype=torch.float32)
             else:
                 print(f"Warning: V_delta column index {self.v_delta_column_index} is invalid or not provided. Cannot extract raw V_delta.")
                 self.raw_v_delta_data = None
        else:
             self.raw_v_delta_data = None

        # 表格数据标准化 (如果 scaler 提供了的话)
        if self.scaler is not None:
            numeric_df = self.df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
            tabular_data = numeric_df.fillna(0).values.astype('float32')
            tabular_data = self.scaler.transform(tabular_data)  # 使用传入的 scaler 进行 transform
            self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        else:
            # 如果没有提供 scaler (例如在训练集创建时)，则进行 fit_transform
            numeric_df = self.df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
            tabular_data = numeric_df.fillna(0).values.astype('float32')
            self.scaler = StandardScaler()
            tabular_data = self.scaler.fit_transform(tabular_data)
            self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
            # 保存归一化参数 (仅在训练集创建时有效)
            self.scaler_mean = self.scaler.mean_
            self.scaler_scale = self.scaler.scale_

        # 标签归一化参数 (在 main 函数中计算并传入)
        # self.label_min 和 self.label_max 现在从外部传入

    def __len__(self):
        return len(self.df)

    # 修改 __getitem__，返回 transform 函数的结果
    def __getitem__(self, idx):
        # 注意：因为我们直接在划分后的 DataFrame 上创建 Dataset，
        # 这里不再需要处理 random_split 的 indices

        img_name = self.df.iloc[idx, 0]
        # 检查图片文件名是否已有扩展名 (这部分保持不变)
        base, ext = os.path.splitext(img_name)
        if ext:
            img_path = os.path.join(self.image_dir, img_name).replace("\\", "/")
            if not os.path.exists(img_path):
                # 添加常见扩展名尝试
                 img_path_with_ext = None
                 for possible_ext in ['.jpg', '.jpeg', '.png']:
                     temp_path = img_path + possible_ext
                     if os.path.exists(temp_path):
                         img_path_with_ext = temp_path
                         break
                 if img_path_with_ext:
                     img_path = img_path_with_ext
                 else:
                    raise FileNotFoundError(f"Image file not found: {img_path}")
        else:
             img_path = os.path.join(self.image_dir, img_name).replace("\\", "/")
             img_found = False
             for ext in ['.jpg', '.jpeg', '.png']:
                 full_img_path = img_path + ext
                 if os.path.exists(full_img_path):
                     img_path = full_img_path
                     img_found = True
                     break
             if not img_found:
                 raise FileNotFoundError(f"Image file not found with common extensions: {img_path}")
        # 加载图片 (先加载为 PIL Image)
        image = Image.open(img_path) # 不再在这里转灰度，交给 transform

        # 表格数据
        tabular_data = self.tabular_data[idx] # 使用已经标准化好的 tabular_data

        # 原始 V_delta 数据
        raw_v_delta = self.raw_v_delta_data[idx] if self.raw_v_delta_data is not None else torch.tensor(0.0,dtype=torch.float32)  # 如果没有 V_delta 列，返回 0 或其他默认值

        # 标签 (获取原始标签)
        label = self.df.iloc[idx, -1].astype('float32')
        # 标签归一化将在 transform 函数中完成，或者由外部函数完成

        # 应用 transform 函数
        # transform 函数现在接收 (image, tabular_data, label, raw_v_delta, label_min, label_max) 并返回处理后的结果 (可能是列表)
        processed_data = self.transform(image, tabular_data, label, raw_v_delta, self.label_min, self.label_max)

        return processed_data


