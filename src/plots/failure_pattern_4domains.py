import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= 1. 配置与映射 =========
MODELS = ['gpt55', "gpt5", "gpt4o", 'claude46', "claude4", "medgemma", 'qwen35', "qwen3", "baichuan", "huatuo", 'deepseekv4', "deepseek"]
MODEL_DISPLAY_NAMES = {
    'gpt55': "GPT-5.5", "gpt5": "GPT-5", "gpt4o": "GPT-4o", 'claude46': "Claude 4.6", "claude4": "Claude 4.0", "deepseek": "DS-V3", 'deepseekv4': "DS-V4",
    'qwen35': "Qwen3.5", "qwen3": "Qwen3", "medgemma": "MedGem", "baichuan": "BC-M2", "huatuo": "HT-o1"
}

DOMAIN_MAP = {
    "Domain 1": "Medical interview questioning skills",
    "Domain 2": "Information coverage",
    "Domain 3": "Handling of ambiguous patient responses",
    "Domain 4": "Ethical and professional conduct",
    "Domain 5": "Clarity and transparency of clinical explanations"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 中间数据目录（阶段1已产出，直接读取绘图）
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data/failure_pattern")
PANEL_PLOT_DATA_PATTERN = os.path.join(INTERMEDIATE_DIR, "panel_{panel_label}_plot_data.csv")
OUT_DIR = "./outputs/failre_pattern_4domains"
os.makedirs(OUT_DIR, exist_ok=True)

JAMA_COLORS = {
    "panel_a": "#87AAB9",
    "panel_b": "#2F5763",
    "panel_c": "#C4D4DA",
    "panel_d": "#5B7C8A"
}
SCATTER_COLOR = "#555555"


# ================================================================
# 读取中间数据 CSV -> 画图
# ================================================================
def format_error_label(err):
    formatted_err = err.replace("_", " ").capitalize()
    if formatted_err == "Uncertain answer handling":
        return "Unaddressed uncertainty"
    elif formatted_err == "Unclear expression handling":
        return "Clarification failure"
    elif formatted_err == "Logical jumps":
        return "Logical gap"
    elif formatted_err == "Repetitive questions":
        return "Repetitive question"
    elif formatted_err == "Unclear questions":
        return "Unclear question"
    elif formatted_err == "Suggestive questions":
        return "Suggestive question"
    elif formatted_err == "Confrontational questions":
        return "Confrontational question"
    return formatted_err


def load_intermediate_data():
    """读取"倒数第二步"的 panel 绘图数据 CSV。"""
    panel_labels = ['a', 'b', 'c', 'd']
    panel_data_list = []
    for p in panel_labels:
        csv_path = PANEL_PLOT_DATA_PATTERN.format(panel_label=p)
        if not os.path.exists(csv_path):
            print(f"⚠️ 文件不存在，跳过: {csv_path}")
            continue
        pdf = pd.read_csv(csv_path)
        # 按 RowOrder 升序（与图上从左到右一致）
        pdf = pdf.sort_values("RowOrder").reset_index(drop=True)
        title = DOMAIN_MAP.get(f"Domain {panel_labels.index(p)+1}", "")
        panel_data_list.append((p, title, pdf))
        print(f"📂 读取 panel {p} 倒数第二步绘图数据 -> {csv_path}（{len(pdf)} 行，5 个箱位）")
    return panel_data_list


def draw_jama_style_panels(panel_iter, out_dir):
    model_display_order = [MODEL_DISPLAY_NAMES[m] for m in MODELS]
    panel_labels = ['a', 'b', 'c', 'd']
    dark_panels = {'panel_b', 'panel_d'}
    generated_files = []

    for panel_label, title, pdf in panel_iter:
        prob_cols = [f"{c}_Prob" for c in model_display_order if f"{c}_Prob" in pdf.columns]
        box_data = [pdf[prob_cols].iloc[i].dropna().values for i in range(len(pdf))]
        labels = pdf["ErrorLabel"].tolist()
        idx = panel_labels.index(panel_label) if panel_label in panel_labels else 0

        fig, ax = plt.subplots(figsize=(8, 6))

        panel_color_key = f"panel_{panel_labels[idx]}"
        panel_color = JAMA_COLORS.get(panel_color_key, "#87AAB9")
        is_dark = panel_color_key in dark_panels
        median_color = "white" if is_dark else "#2F5763"
        line_color = "#2F5763"

        bp = ax.boxplot(
            box_data,
            vert=True,
            patch_artist=True,
            widths=0.45,
            showfliers=True,
            whis=1.5,
            flierprops=dict(
                marker='o',
                markerfacecolor='none',
                markeredgecolor="#2F5763",
                markersize=5,
                alpha=0.7
            )
        )

        for patch in bp['boxes']:
            patch.set_facecolor(panel_color)
            patch.set_edgecolor(line_color)
            patch.set_linewidth(0.8)

        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(1.3)

        for whisker in bp['whiskers']:
            whisker.set_color(line_color)
            whisker.set_linewidth(0.8)

        for cap in bp['caps']:
            cap.set_color(line_color)
            cap.set_linewidth(0.8)

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')

        base_axes_width = 0.7
        max_labels = 5
        current_axes_width = base_axes_width * (len(labels) / max_labels)
        ax.set_position([0.20, 0.22, current_axes_width, 0.70])

        ax.set_xlim(0.5, len(labels) + 0.5)

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel("Failure probability", fontsize=11)

        ax.grid(axis="y", color="#ECECEC", linewidth=0.6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        for ext, fmt, kw in [
            ("svg", "svg", {}),
            ("png", "png", {"dpi": 300}),
            ("eps", "eps", {}),
        ]:
            panel_save_path = os.path.join(out_dir, f"panel_{panel_labels[idx]}.{ext}")
            plt.savefig(panel_save_path, format=fmt, bbox_inches='tight', **kw)
            generated_files.append(panel_save_path)
            print(f"✅ 保存子图 -> {panel_save_path}")
        plt.close()

    return generated_files


# ========= 运行：从中间数据画图 =========
if __name__ == "__main__":
    panel_iter = load_intermediate_data()
    generated_files = draw_jama_style_panels(panel_iter, OUT_DIR)
    print("\nGenerated files:")
    for f in generated_files:
        print(f"  - {f}")
