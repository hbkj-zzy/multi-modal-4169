import torch
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from model import build_resnet


def run_shap_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\best_model.pth'
    csv_file = r'D:\desktop\Desktop\GH4169resnet7（标尺统一）\data.csv'

    model = build_resnet(tabular_input_dim=29).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(csv_file)
    numeric_df = df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = numeric_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df.values)
    test_img = torch.randn(1, 1, 336, 336).to(device)

    def predict_func(tabular_np):
        tabular_tensor = torch.tensor(tabular_np, dtype=torch.float32).to(device)
        batch_size = tabular_tensor.size(0)
        with torch.no_grad():
            images = test_img.repeat(batch_size, 1, 1, 1)
            l_model, _, _, _, _ = model(
                images,
                tabular_tensor,
                raw_v_delta=torch.zeros(batch_size).to(device),
                L_max=70,
                alpha=0.001,
                v_delta_opt=0.05,
                label_min=0,
                label_max=100,
            )
        return l_model.cpu().numpy().flatten()

    print('正在初始化 SHAP Explainer...')
    background = shap.kmeans(X_scaled, 10)
    explainer = shap.KernelExplainer(predict_func, background)
    print('开始计算 SHAP Values (100 samples)...')
    shap_values = explainer.shap_values(X_scaled[:100], nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values_arr = np.array(shap_values)
    if shap_values_arr.ndim == 3:
        shap_values_2d = shap_values_arr.reshape(
            shap_values_arr.shape[0], shap_values_arr.shape[1]
        )
    else:
        shap_values_2d = shap_values_arr

    # ---------------------------------------------------------
    # 6. 可视化与导出 (四边框、无刻度、所有字体放大加粗为Times New Roman)
    # ---------------------------------------------------------
    print('正在生成筛选后的高级 SHAP 图表...')

    target_features = [
        'Temp',
        'δsize(after)',
        'gs(after)',
        'δsize',
        'δcontent(after)',
        'gs(before)',
        'C',
        'Cr',
        'Mn',
        'YS',
    ]
    indices = [feature_names.index(f) for f in target_features if f in feature_names]

    filtered_shap_values = shap_values_2d[:, indices]
    filtered_X = X_scaled[:100, indices]
    filtered_names = [feature_names[i] for i in indices]

    # A. 配置全局 Times New Roman 字体环境
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # B. 构建散点图所需的特定渐变色盘
    colors = ['#BAC9E3', '#FDDBDA']
    custom_cmap = LinearSegmentedColormap.from_list('custom_shap_cmap', colors)

    # ==========================================
    # 核心绘图 1: SHAP Summary Plot (点图)
    # ==========================================
    fig1 = plt.figure(figsize=(9.5, 6.5), dpi=300)

    shap.summary_plot(
        filtered_shap_values,
        filtered_X,
        feature_names=filtered_names,
        plot_type='dot',
        show=False,
    )

    ax1 = plt.gca()

    for collection in ax1.collections:
        collection.set_cmap(custom_cmap)

    # 边框加粗至 2.0
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color('#000000')

    # 仅保留左侧和底部的刻度线，加粗至 2.0
    ax1.get_xaxis().set_tick_params(
        which='both', top=False, bottom=True, direction='in', width=2.0, length=6
    )
    ax1.get_yaxis().set_tick_params(
        which='both', left=True, right=False, direction='in', width=2.0, length=6
    )

    # 特征名字体大小提升至 16 号字，加粗
    ax1.set_yticklabels(
        ax1.get_yticklabels(),
        fontname='Times New Roman',
        fontsize=16,
        fontweight='bold',
    )

    # 横坐标标签字体大小提升至 17 号字，加粗
    ax1.set_xlabel(
        'SHAP value (impact on model output)',
        fontsize=17,
        fontname='Times New Roman',
        fontweight='bold',
    )
    for label in ax1.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')

    # 接管右侧 Colorbar 的美化
    for child in fig1.get_children():
        if isinstance(child, plt.Axes) and child != ax1:
            colorbar_ax = child
            colorbar_ax.clear()
            norm = plt.Normalize(vmin=0, vmax=1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
            cbar = fig1.colorbar(mappable, cax=colorbar_ax, ticks=[0, 1])
            cbar.set_ticklabels(['Low', 'High'])
            cbar.set_label(
                'Feature value',
                labelpad=6,
                fontname='Times New Roman',
                fontsize=17,
                fontweight='bold',
            )
            for spine in colorbar_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('#000000')
            for text in colorbar_ax.get_yticklabels():
                text.set_fontname('Times New Roman')
                text.set_fontsize(16)
                text.set_weight('bold')

    plt.tight_layout()
    plt.savefig('shap_summary_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 核心绘图 2: SHAP 特征重要性条形图 (Bar Plot)
    # ==========================================
    print('正在生成匹配的四边封闭型 SHAP 条形图...')
    global_importance = np.abs(filtered_shap_values).mean(0)
    sort_inds = np.argsort(global_importance)
    sorted_importance = global_importance[sort_inds]
    sorted_names = [filtered_names[i] for i in sort_inds]

    fig2, ax_bar = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    # 条形图的边框线加粗至 1.5
    bars = ax_bar.barh(
        sorted_names,
        sorted_importance,
        color='#d1ead1',
        height=0.6,
        edgecolor='#333333',
        linewidth=1.5,
    )

    # 数值标签字体大小提升至 15 号字，加粗
    for bar in bars:
        width = bar.get_width()
        text_obj = ax_bar.text(
            width + (max(sorted_importance) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}',
            va='center',
            ha='left',
            fontsize=15,
            fontname='Times New Roman',
        )
        text_obj.set_weight('bold')

    ax_bar.set_xlim(0, max(sorted_importance) * 1.25)

    # 规范化条形图的四条边框，加粗至 2.0
    for spine in ax_bar.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color('#000000')

    # 设置刻度朝内，上右无刻度，线宽加粗至 2.0
    ax_bar.get_xaxis().set_tick_params(
        which='both', top=False, bottom=True, direction='in', width=2.0, length=6
    )
    ax_bar.get_yaxis().set_tick_params(
        which='both', left=True, right=False, direction='in', width=2.0, length=6
    )

    # 横坐标标签字体大小 17 号字，纵坐标与刻度标签 16 号字，全部加粗及 Times New Roman
    ax_bar.set_xlabel(
        'mean(|SHAP value|) (average impact on model output magnitude)',
        fontsize=17,
        fontname='Times New Roman',
        fontweight='bold',
    )
    for label in ax_bar.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')
    for label in ax_bar.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')

    plt.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 导出 CSV 备份
    shap_df = pd.DataFrame(filtered_shap_values, columns=filtered_names)
    original_data_df = pd.DataFrame(
        filtered_X, columns=[f'Value_{n}' for n in filtered_names]
    )
    final_export = pd.concat([shap_df, original_data_df], axis=1)
    final_export.to_csv('shap_values_filtered_analysis.csv', index=False)

    print('🎉 绘图完成！所有字体已按要求大幅放大加粗为新罗马体，边框与线条也已加粗。')


if __name__ == '__main__':
    run_shap_analysis()