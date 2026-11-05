import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap
from model import build_resnet
from sklearn.preprocessing import StandardScaler


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
    # 6. 可视化部分 A: 渐变 Summary Plot (特征散点分布图)
    # ---------------------------------------------------------
    print('正在生成高保真 SHAP 散点分布图 (Summary Plot)...')

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

    colors = ['#2563eb', '#dc2626']
    custom_cmap = LinearSegmentedColormap.from_list('custom_shap_cmap', colors)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(9, 6.5), dpi=300)

    shap.summary_plot(
        filtered_shap_values,
        filtered_X,
        feature_names=filtered_names,
        plot_type='dot',
        show=False,
    )

    ax = plt.gca()

    for collection in ax.collections:
        collection.set_cmap(custom_cmap)

    for child in fig.get_children():
        if isinstance(child, plt.Axes) and child != ax:
            colorbar_ax = child
            colorbar_ax.clear()

            norm = plt.Normalize(vmin=0, vmax=1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
            cbar = fig.colorbar(mappable, cax=colorbar_ax, ticks=[0, 1])

            cbar.set_ticklabels(['Low', 'High'])
            cbar.set_label('Feature value', labelpad=6, fontsize=17, fontweight='bold')

            for spine in colorbar_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.8)
                spine.set_color('#000000')

            for text in colorbar_ax.get_yticklabels():
                text.set_fontname('Times New Roman')
                text.set_fontsize(16)
                text.set_weight('bold')
            colorbar_ax.yaxis.label.set_fontname('Times New Roman')
            colorbar_ax.yaxis.label.set_fontsize(17)
            colorbar_ax.yaxis.label.set_weight('bold')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color('#000000')

    ax.tick_params(direction='in', top=False, right=False, width=1.8, length=5)

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontname='Times New Roman',
        fontsize=16,
        fontweight='bold',
    )
    ax.set_xlabel(
        'SHAP value (impact on model output)',
        fontsize=17,
        fontname='Times New Roman',
        fontweight='bold',
    )
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')

    plt.tight_layout()
    plt.savefig('shap_summary_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 6. 可视化部分 B: 淡绿色带数值标签的条形图 (Bar Plot)
    # ---------------------------------------------------------
    print('正在生成高度契合样稿样式的 SHAP 条形图 (Bar Plot)...')

    global_importance = np.abs(filtered_shap_values).mean(0)
    sort_inds = np.argsort(global_importance)
    sorted_importance = global_importance[sort_inds]
    sorted_names = [filtered_names[i] for i in sort_inds]

    plt.figure(figsize=(8.5, 6.0), dpi=300)

    bars = plt.barh(
        sorted_names,
        sorted_importance,
        color='#D1EAD1',
        height=0.65,
        edgecolor='#333333',
        linewidth=1.2,
    )

    plt.xlabel(
        'mean(|SHAP value|) (average impact on model output magnitude)',
        fontsize=17,
        fontname='Times New Roman',
        fontweight='bold',
        labelpad=10,
    )

    ax_bar = plt.gca()

    for spine in ax_bar.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color('#000000')

    ax_bar.tick_params(direction='in', top=False, right=False, width=1.8, length=5)

    bar_labels = ax_bar.bar_label(bars, padding=5, fmt='%.4f', fontsize=15)
    for label in bar_labels:
        label.set_fontname('Times New Roman')
        label.set_weight('bold')

    for label in ax_bar.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')

    for label in ax_bar.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)
        label.set_weight('bold')

    ax_bar.set_xlim(0, max(sorted_importance) * 1.25)

    plt.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    shap_df = pd.DataFrame(filtered_shap_values, columns=filtered_names)
    original_data_df = pd.DataFrame(
        filtered_X, columns=[f'Value_{n}' for n in filtered_names]
    )
    final_export = pd.concat([shap_df, original_data_df], axis=1)
    final_export.to_csv('shap_values_filtered_analysis.csv', index=False)

    print(
        '🎉 完美！所有图表的字体大小已进一步全面放大（坐标标签提升至16-17号字，数值标签提升至15号字），Times New Roman字体、左/下侧单边刻度与加粗边框完美保留。'
    )


if __name__ == '__main__':
    run_shap_analysis()