import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import build_resnet
from sklearn.preprocessing import StandardScaler
import os

# 1. 基础配置：设置全局衬线字体为 Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块


def run_pdp_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\best_model.pth'
    csv_file = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\data.csv'

    # 1. 加载模型
    model = build_resnet(tabular_input_dim=29).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 准备数据
    df = pd.read_csv(csv_file)
    numeric_df = df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = numeric_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df.values)

    # 3. 选定要求的 10 个关键特征
    target_features = [
        "Temp", "δsize(after)", "gs(after)", "δsize",
        "δcontent(after)", "gs(before)", "C", "Cr", "Mn", "YS"
    ]

    # 固定一张代表性的图像背景
    test_img = torch.randn(1, 1, 336, 336).to(device)

    # 4. PDP 计算核心函数
    def get_pdp_values(feature_name, grid_points=50):
        feat_idx = feature_names.index(feature_name)

        # 在该特征的真实取值范围内生成均匀的网格点（基于归一化数据）
        feat_min, feat_max = X_scaled[:, feat_idx].min(), X_scaled[:, feat_idx].max()
        grid = np.linspace(feat_min, feat_max, grid_points)

        pdp_means = []

        # 使用前100个样本进行扰动平均
        X_ref = X_scaled[:100].copy()

        for val in grid:
            X_temp = X_ref.copy()
            X_temp[:, feat_idx] = val

            tabular_tensor = torch.tensor(X_temp, dtype=torch.float32).to(device)
            with torch.no_grad():
                images = test_img.repeat(tabular_tensor.size(0), 1, 1, 1)
                outputs, _, _, _, _ = model(
                    images, tabular_tensor,
                    raw_v_delta=torch.zeros(tabular_tensor.size(0)).to(device),
                    L_max=70, alpha=0.001, v_delta_opt=0.05,
                    label_min=0, label_max=100
                )
                pdp_means.append(outputs.mean().item())

        # 逆归一化回真实物理量纲
        dummy = np.zeros((grid_points, len(feature_names)))
        dummy[:, feat_idx] = grid
        real_grid = scaler.inverse_transform(dummy)[:, feat_idx]

        return real_grid, pdp_means

    # 5. 逐个特征生成单张图片
    for feat in target_features:
        if feat not in feature_names:
            print(f"警告: 特征 '{feat}' 不在数据的列名中，跳过计算。")
            continue

        print(f"正在计算并生成特征 【{feat}】 的独立 PDP 图...")
        x, y = get_pdp_values(feat)

        # 创建更大的画布以完美容纳极大号加粗字体
        fig, ax = plt.subplots(figsize=(10.0, 8.0), dpi=300)

        # 保持高对比度颜色方案：深海军蓝主曲线 + 高透亮浅蓝填充
        ax.plot(x, y, color='#1e3a8a', linewidth=4.5, zorder=2)
        ax.fill_between(x, y, min(y), color='#93c5fd', alpha=0.35, zorder=1)

        # ---------------------------------------------------------
        # 深度定制：SCI 封闭式四边框与极大加粗字体控制
        # ---------------------------------------------------------
        # 1. 强制显示并加粗四条边框 (线宽设为 2.5 pt)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color('#000000')

        # 2. 精确控制刻度表现：刻度线朝内 (in)，线宽加粗至 2.5
        ax.get_xaxis().set_tick_params(which='both', top=False, bottom=True, direction='in', width=2.5, length=7)
        ax.get_yaxis().set_tick_params(which='both', left=True, right=False, direction='in', width=2.5, length=7)

        # 3. 严格设置标题与坐标轴标签（标签提升至 28 号字，标题 32 号字，全部 Times New Roman 且加粗）
        ax.set_title(f"Partial Dependence of {feat}", fontsize=31, fontname="Times New Roman", fontweight="bold",
                     pad=22)
        ax.set_xlabel(f"{feat}", fontsize=27, fontname="Times New Roman", fontweight="bold", labelpad=18)
        ax.set_ylabel("Predicted log(Creep Life)", fontsize=27, fontname="Times New Roman", fontweight="bold",
                      labelpad=18)

        # 4. 坐标轴刻度数字（数值标签）字体设置提升至 26 号字，加粗
        for label in ax.get_xticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(25)
            label.set_weight("bold")

        for label in ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(25)
            label.set_weight("bold")

        # 5. 网格线美化
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # 保存图片
        safe_filename = feat.replace("(", "_").replace(")", "_")
        plt.tight_layout()
        plt.savefig(f"pdp_{safe_filename}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)  # 释放内存

    print(f"\n全部 10 张极放大字体、高清晰度 PDP 图已生成完毕！结果已保存至：{os.getcwd()}")


if __name__ == "__main__":
    run_pdp_analysis()