import os
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns
import torchvision.transforms.functional as F

# 引入真实的模型和数据集类
from model import build_resnet as build_resnet_bn
from dataset import MultiModalDataset


# ==========================================
# 数据增强与预处理函数
# ==========================================

def train_transform_complex(image, tabular_data, label, raw_v_delta, label_min, label_max):
    augmented_samples = []
    label_normalized = (label - label_min) / (label_max - label_min)
    label_tensor = torch.tensor([label_normalized], dtype=torch.float32)

    image = image.convert("L")

    post_crop_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    for _ in range(5):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(336, 336))
        cropped_image = F.crop(image, i, j, h, w)
        img_rotated = F.rotate(cropped_image, 180)
        img_mirrored = F.hflip(cropped_image)

        augmented_samples.append((post_crop_transform(cropped_image), tabular_data, label_tensor, raw_v_delta))
        augmented_samples.append((post_crop_transform(img_rotated), tabular_data, label_tensor, raw_v_delta))
        augmented_samples.append((post_crop_transform(img_mirrored), tabular_data, label_tensor, raw_v_delta))

    return augmented_samples


def val_test_transform_simple(image, tabular_data, label, raw_v_delta, label_min, label_max):
    label_normalized = (label - label_min) / (label_max - label_min)
    label_tensor = torch.tensor([label_normalized], dtype=torch.float32)

    image = image.convert("L")
    image = transforms.Resize((336, 336))(image)

    image_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(image)

    return [(image_tensor, tabular_data, label_tensor, raw_v_delta)]


def custom_collate_fn(batch):
    flattened_batch = [item for sublist in batch for item in sublist]
    images, tabular_data, labels, raw_v_deltas = zip(*flattened_batch)
    images = torch.stack(images)
    tabular_data = torch.stack(tabular_data)
    labels = torch.stack(labels)
    raw_v_deltas = torch.stack(raw_v_deltas)

    return images, tabular_data, labels, raw_v_deltas


def train_val_test_data_process(image_dir, csv_file, v_delta_col_name='δcontent(after)', batch_size=4, random_seed=42):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"无法找到 CSV 文件，请确认文件路径是否正确: {csv_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"无法找到图像文件夹，请确认文件夹路径是否正确: {image_dir}")

    df = pd.read_csv(csv_file)

    if v_delta_col_name in df.columns:
        v_delta_column_index = df.columns.get_loc(v_delta_col_name)
        print(f"成功找到 V_delta 列 '{v_delta_col_name}'，索引为: {v_delta_column_index}")
    else:
        v_delta_column_index = None
        print(f"警告: 未在 CSV 文件中找到名为 '{v_delta_col_name}' 的列。")

    label_min = df.iloc[:, -1].min()
    label_max = df.iloc[:, -1].max()

    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_size = int(0.7 * len(df_shuffled))
    val_size = int(0.15 * len(df_shuffled))

    train_df = df_shuffled.iloc[:train_size].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df_shuffled.iloc[train_size + val_size:].reset_index(drop=True)

    train_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=train_df,
                                      v_delta_column_index=v_delta_column_index,
                                      transform=train_transform_complex,
                                      label_min=label_min, label_max=label_max)

    scaler = train_dataset.scaler

    val_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=val_df,
                                    v_delta_column_index=v_delta_column_index,
                                    transform=val_test_transform_simple, scaler=scaler,
                                    label_min=label_min, label_max=label_max)

    test_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=test_df,
                                     v_delta_column_index=v_delta_column_index,
                                     transform=val_test_transform_simple, scaler=scaler,
                                     label_min=label_min, label_max=label_max)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, label_min, label_max, v_delta_column_index


# ==========================================
# 训练过程与评估
# ==========================================

def train_model_process(model, train_dataloader, val_dataloader, test_dataloader,
                        label_min, label_max, v_delta_column_index,
                        L_max, alpha, v_delta_opt, lambda_reg,
                        num_epoch, avg_true):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion_task = nn.SmoothL1Loss()
    criterion_phys = nn.SmoothL1Loss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # 记录训练集 Loss
    train_loss_all = []
    train_task_loss_all = []
    train_phys_loss_all = []

    # 记录验证集 Loss
    val_loss_all = []
    val_task_loss_all = []
    val_phys_loss_all = []

    # 新增：记录测试集每轮 Loss (供 Origin 绘图)
    test_loss_all = []
    test_task_loss_all = []
    test_phys_loss_all = []

    train_mae_all = []
    val_mae_all = []
    train_r2_all = []
    val_r2_all = []
    train_outputs_all_epochs = []
    train_labels_all_epochs = []
    val_outputs_all_epochs = []
    val_labels_all_epochs = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')
        print('-' * 10)

        # ------------------ 1. 训练阶段 ------------------
        model.train()
        epoch_train_total_loss = 0.0
        epoch_train_task_loss = 0.0
        epoch_train_phys_loss = 0.0
        epoch_train_mae = 0.0
        train_samples = 0
        all_train_outputs = []
        all_train_labels = []

        for images, tabular_data, labels, raw_v_deltas in train_dataloader:
            images = images.to(device).float()
            tabular_data = tabular_data.to(device).float()
            labels = labels.to(device).float().squeeze(-1)
            raw_v_deltas = raw_v_deltas.to(device).float()

            optimizer.zero_grad()
            L_model_normalized, L_phys_normalized, attention_weights, _, _ = model(
                images, tabular_data, raw_v_deltas,
                L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                label_min=label_min, label_max=label_max
            )

            labels_normalized = labels.unsqueeze(-1)

            task_loss = criterion_task(L_model_normalized, labels_normalized)
            phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)
            total_loss = task_loss + lambda_reg * phys_loss

            total_loss.backward()
            optimizer.step()

            epoch_train_total_loss += total_loss.item() * labels.size(0)
            epoch_train_task_loss += task_loss.item() * labels.size(0)
            epoch_train_phys_loss += phys_loss.item() * labels.size(0)

            outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min
            labels_original = labels * (label_max - label_min) + label_min

            epoch_train_mae += mean_absolute_error(labels_original.cpu().numpy(),
                                                   outputs_original.cpu().detach().numpy()) * labels.size(0)
            all_train_outputs.append(outputs_original.cpu().detach().numpy())
            all_train_labels.append(labels_original.cpu().numpy())
            train_samples += labels.size(0)

        train_epoch_outputs = np.concatenate(all_train_outputs, axis=0)
        train_epoch_labels = np.concatenate(all_train_labels, axis=0)
        train_r2 = r2_score(train_epoch_labels, train_epoch_outputs)

        train_loss_all.append(epoch_train_total_loss / train_samples)
        train_task_loss_all.append(epoch_train_task_loss / train_samples)
        train_phys_loss_all.append(epoch_train_phys_loss / train_samples)
        train_mae_all.append(epoch_train_mae / train_samples)
        train_r2_all.append(train_r2)
        train_outputs_all_epochs.append(train_epoch_outputs)
        train_labels_all_epochs.append(train_epoch_labels)

        # ------------------ 2. 验证阶段 ------------------
        model.eval()
        epoch_val_total_loss = 0.0
        epoch_val_task_loss = 0.0
        epoch_val_phys_loss = 0.0
        epoch_val_mae = 0.0
        val_samples = 0
        all_val_outputs = []
        all_val_labels = []

        with torch.no_grad():
            for images, tabular_data, labels, raw_v_deltas in val_dataloader:
                images = images.to(device).float()
                tabular_data = tabular_data.to(device).float()
                labels = labels.to(device).float().squeeze(-1)
                raw_v_deltas = raw_v_deltas.to(device).float()

                L_model_normalized, L_phys_normalized, _, _, _ = model(
                    images, tabular_data, raw_v_deltas,
                    L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                    label_min=label_min, label_max=label_max
                )

                labels_normalized = labels.unsqueeze(-1)

                task_loss = criterion_task(L_model_normalized, labels_normalized)
                phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)
                total_loss = task_loss + lambda_reg * phys_loss

                epoch_val_total_loss += total_loss.item() * labels.size(0)
                epoch_val_task_loss += task_loss.item() * labels.size(0)
                epoch_val_phys_loss += phys_loss.item() * labels.size(0)

                outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min
                labels_original = labels * (label_max - label_min) + label_min

                epoch_val_mae += mean_absolute_error(labels_original.cpu().numpy(),
                                                     outputs_original.cpu().detach().numpy()) * labels.size(0)
                all_val_outputs.append(outputs_original.cpu().detach().numpy())
                all_val_labels.append(labels_original.cpu().numpy())
                val_samples += labels.size(0)

        val_epoch_outputs = np.concatenate(all_val_outputs, axis=0)
        val_epoch_labels = np.concatenate(all_val_labels, axis=0)
        val_r2 = r2_score(val_epoch_labels, val_epoch_outputs)
        val_total_loss = epoch_val_total_loss / val_samples if val_samples > 0 else 0

        val_mae = epoch_val_mae / val_samples if val_samples > 0 else np.nan
        val_mae = np.nan_to_num(val_mae, nan=0.0)

        val_loss_all.append(val_total_loss)
        val_task_loss_all.append(epoch_val_task_loss / val_samples)
        val_phys_loss_all.append(epoch_val_phys_loss / val_samples)
        val_mae_all.append(val_mae)
        val_r2_all.append(val_r2)
        val_outputs_all_epochs.append(val_epoch_outputs)
        val_labels_all_epochs.append(val_epoch_labels)

        # ------------------ 3. 测试阶段 (每轮计算 Loss 以供 Origin 导出) ------------------
        epoch_test_total_loss = 0.0
        epoch_test_task_loss = 0.0
        epoch_test_phys_loss = 0.0
        test_samples = 0

        with torch.no_grad():
            for images, tabular_data, labels, raw_v_deltas in test_dataloader:
                images = images.to(device).float()
                tabular_data = tabular_data.to(device).float()
                labels = labels.to(device).float().squeeze(-1)
                raw_v_deltas = raw_v_deltas.to(device).float()

                L_model_normalized, L_phys_normalized, _, _, _ = model(
                    images, tabular_data, raw_v_deltas,
                    L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                    label_min=label_min, label_max=label_max
                )

                labels_normalized = labels.unsqueeze(-1)
                task_loss = criterion_task(L_model_normalized, labels_normalized)
                phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)
                total_loss = task_loss + lambda_reg * phys_loss

                epoch_test_total_loss += total_loss.item() * labels.size(0)
                epoch_test_task_loss += task_loss.item() * labels.size(0)
                epoch_test_phys_loss += phys_loss.item() * labels.size(0)
                test_samples += labels.size(0)

        test_loss_all.append(epoch_test_total_loss / test_samples)
        test_task_loss_all.append(epoch_test_task_loss / test_samples)
        test_phys_loss_all.append(epoch_test_phys_loss / test_samples)

        print(
            f'Train Loss: Total={train_loss_all[-1]:.4f}, Task={train_task_loss_all[-1]:.4f}, Phys={train_phys_loss_all[-1]:.4f}')
        print(
            f'Val Loss:   Total={val_loss_all[-1]:.4f}, Task={val_task_loss_all[-1]:.4f}, Phys={val_phys_loss_all[-1]:.4f}')
        print(
            f'Test Loss:  Total={test_loss_all[-1]:.4f}, Task={test_task_loss_all[-1]:.4f}, Phys={test_phys_loss_all[-1]:.4f}')

        scheduler.step(val_total_loss)

        if val_total_loss < best_loss:
            best_loss = val_total_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    # 最佳模型的最终测试集中真实值与预测值导出
    model.eval()
    all_test_outputs = []
    all_test_labels = []

    with torch.no_grad():
        for images, tabular_data, labels, raw_v_deltas in test_dataloader:
            images = images.to(device).float()
            tabular_data = tabular_data.to(device).float()
            labels = labels.to(device).float().squeeze(-1)
            raw_v_deltas = raw_v_deltas.to(device).float()

            L_model_normalized, L_phys_normalized, _, _, _ = model(
                images, tabular_data, raw_v_deltas,
                L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                label_min=label_min, label_max=label_max
            )

            outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min
            labels_original = labels * (label_max - label_min) + label_min

            all_test_outputs.append(outputs_original.cpu().detach().numpy())
            all_test_labels.append(labels_original.cpu().numpy())

    test_outputs = np.concatenate(all_test_outputs, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)

    return {
        'train_loss': train_loss_all,
        'train_task_loss': train_task_loss_all,
        'train_phys_loss': train_phys_loss_all,
        'val_loss': val_loss_all,
        'val_task_loss': val_task_loss_all,
        'val_phys_loss': val_phys_loss_all,
        'test_loss': test_loss_all,
        'test_task_loss': test_task_loss_all,
        'test_phys_loss': test_phys_loss_all,
        'train_outputs_all_epochs': train_outputs_all_epochs,
        'train_labels_all_epochs': train_labels_all_epochs,
        'val_outputs_all_epochs': val_outputs_all_epochs,
        'val_labels_all_epochs': val_labels_all_epochs,
        'test_outputs': test_outputs,
        'test_labels': test_labels
    }


# ==========================================
# 主程序
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)

    csv_file = os.path.join(parent_dir, 'data.csv')
    image_dir = os.path.join(parent_dir, 'GHdata')
    export_path = os.path.join(parent_dir, 'origin_data')
    model_save_path = os.path.join(parent_dir, 'best_model.pth')

    if not os.path.exists(csv_file):
        csv_file = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\data.csv'
    if not os.path.exists(image_dir):
        image_dir = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\GHdata'

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    batch_size = 12
    num_epoch = 30

    L_max_physics = 70
    alpha_physics = 0.001
    v_delta_opt_physics = 0.05
    lambda_regularization = 0.1

    v_delta_column_name = 'δcontent(after)'

    train_loader, val_loader, test_loader, label_min, label_max, v_delta_col_index = train_val_test_data_process(
        image_dir, csv_file, v_delta_col_name=v_delta_column_name, batch_size=batch_size)

    df_full = pd.read_csv(csv_file)
    avg_true = np.mean(df_full.iloc[:, -1].values.astype('float32'))

    model_bn = build_resnet_bn(tabular_input_dim=30).to(device)
    models = {'G-MAPN': model_bn}

    for model_name, model in models.items():
        print(f"\n--- Training and Evaluating {model_name} Model ---")

        train_process = train_model_process(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            label_min=label_min,
            label_max=label_max,
            v_delta_column_index=v_delta_col_index,
            L_max=L_max_physics,
            alpha=alpha_physics,
            v_delta_opt=v_delta_opt_physics,
            lambda_reg=lambda_regularization,
            num_epoch=num_epoch,
            avg_true=avg_true
        )

        # ----------------------------------------------------
        # 导出符合 Origin 画图标准的 10 列完整数据 CSV
        # ----------------------------------------------------
        origin_loss_df = pd.DataFrame({
            'Epoch': range(1, num_epoch + 1),
            'Train_Task_Loss': train_process['train_task_loss'],
            'Train_Phys_Loss': train_process['train_phys_loss'],
            'Train_Total_Loss': train_process['train_loss'],
            'Val_Task_Loss': train_process['val_task_loss'],
            'Val_Phys_Loss': train_process['val_phys_loss'],
            'Val_Total_Loss': train_process['val_loss'],
            'Test_Task_Loss': train_process['test_task_loss'],
            'Test_Phys_Loss': train_process['test_phys_loss'],
            'Test_Total_Loss': train_process['test_loss']
        })

        loss_csv_file = os.path.join(export_path, f'{model_name}_loss_curves_all_epochs.csv')
        origin_loss_df.to_csv(loss_csv_file, index=False)
        print(f"\n>>> 导出成功！Origin 画图专用的全 Epoch Loss 数据已存入:\n    {loss_csv_file}")

        # 保存散点图数据与模型权重
        best_val_epoch_idx = np.argmin(train_process['val_loss'])

        pd.DataFrame({
            'True_Life': train_process['train_labels_all_epochs'][best_val_epoch_idx],
            'Pred_Life': train_process['train_outputs_all_epochs'][best_val_epoch_idx]
        }).to_csv(os.path.join(export_path, f'{model_name}_scatter_train.csv'), index=False)

        pd.DataFrame({
            'True_Life': train_process['val_labels_all_epochs'][best_val_epoch_idx],
            'Pred_Life': train_process['val_outputs_all_epochs'][best_val_epoch_idx]
        }).to_csv(os.path.join(export_path, f'{model_name}_scatter_val.csv'), index=False)

        pd.DataFrame({
            'True_Life': train_process['test_labels'],
            'Pred_Life': train_process['test_outputs']
        }).to_csv(os.path.join(export_path, f'{model_name}_scatter_test.csv'), index=False)

        torch.save(model_bn.state_dict(), model_save_path)
        print(f">>> 最佳模型权重已保存至: {model_save_path}")


if __name__ == '__main__':
    main()