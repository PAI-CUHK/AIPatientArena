from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 配置 =========
MODEL_NAMES = ["gpt55", "gpt5", "gpt4o", "claude46", "claude4", "deepseekv4", "deepseek", "qwen35", "qwen3", "medgemma", "huatuo", "baichuan"]
MODEL_LABELS = [f"LLM{i}" for i in range(1, 13)]

# 中间数据目录（阶段1已产出，直接读取绘图）
BASE_DIR = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = BASE_DIR / "data" / "dx_ms"
INTERSECTION_CSV = INTERMEDIATE_DIR / "dx_ms_intersection.csv"
ALL_CSV = INTERMEDIATE_DIR / "dx_ms_all.csv"

OUTPUT_DIR = Path("./outputs/diagnosis_accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 读取中间数据 CSV -> 画图
# ================================================================
def load_intermediate_data():
    """读取中间数据 CSV，并按 MODEL_NAMES 顺序排序，保证 LLM1-LLM12 标签对应正确。"""
    df_int = pd.read_csv(INTERSECTION_CSV)
    df_all = pd.read_csv(ALL_CSV)

    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    df_int["_order"] = df_int["model"].map(order)
    df_all["_order"] = df_all["model"].map(order)
    df_int = df_int.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df_all = df_all.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    print(f"📂 读取交集病例中间数据 -> {INTERSECTION_CSV}（{len(df_int)} 行）")
    print(f"📂 读取所有病例中间数据 -> {ALL_CSV}（{len(df_all)} 行）")
    return df_int, df_all


def plot_radar_chart(df_int, df_all):
    from matplotlib.lines import Line2D

    models = MODEL_LABELS
    num_vars = len(models)

    r_min = 2
    r_max = 4
    r_step = 0.5
    r_range = r_max - r_min

    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, num_vars, endpoint=False)
    angles = np.append(angles, angles[0])  # 闭合

    def draw_single_radar(ax, all_vals, intersection_vals):
        all_offset = np.append(np.array(all_vals) - r_min, (np.array(all_vals) - r_min)[0])
        intersection_offset = np.append(np.array(intersection_vals) - r_min, (np.array(intersection_vals) - r_min)[0])

        for r in np.arange(0, r_range + 0.01, r_step):
            x = (r / r_range) * np.cos(angles)
            y = (r / r_range) * np.sin(angles)
            ax.plot(x, y, color="#D9D9D9", linestyle='--', linewidth=0.5)

            for i in range(num_vars):
                mid_angle = (angles[i] + angles[i+1])/2
                x_text = (r / r_range) * np.cos(mid_angle) * 1.05
                y_text = (r / r_range) * np.sin(mid_angle) * 1.05
                if i == 0:
                    ax.text(x_text, y_text, f"{r + r_min:.1f}", fontsize=11, ha='center', va='center', color="#333333")

        for a in angles[:-1]:
            ax.plot([0, np.cos(a)], [0, np.sin(a)], color="#D9D9D9", linestyle='--', linewidth=0.5)

        x_all = (all_offset / r_range) * np.cos(angles)
        y_all = (all_offset / r_range) * np.sin(angles)
        ax.plot(x_all, y_all, 'o-', color="#426A9E", linewidth=2, label="All cases")
        ax.fill(x_all, y_all, color="#426A9E", alpha=0.06)

        x_int = (intersection_offset / r_range) * np.cos(angles)
        y_int = (intersection_offset / r_range) * np.sin(angles)
        ax.plot(x_int, y_int, 's-', color="#A64B4B", linewidth=2, label="Dialogue-Sufficient")
        ax.fill(x_int, y_int, color="#A64B4B", alpha=0.06)

        for i, label in enumerate(models):
            x_text = 1.15 * np.cos(angles[i])
            y_text = 1.15 * np.sin(angles[i])
            ax.text(x_text, y_text, label, ha='center', va='center', fontsize=12, color="#333333")

        ax.set_facecolor('white')
        ax.axis('off')
        ax.set_aspect('equal')

    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='-', color='#426A9E', linewidth=2, label='All cases'),
        Line2D([0], [0], marker='s', linestyle='-', color='#A64B4B', linewidth=2, label='Dialogue-Sufficient')
    ]

    panels = [
        ("diagnostic_reasoning_radar",
         df_all["diagnostic_reasoning_avg"].tolist(),
         df_int["diagnostic_reasoning_avg"].tolist()),
        ("medication_safety_radar",
         df_all["medication_safety_avg"].tolist(),
         df_int["medication_safety_avg"].tolist()),
    ]

    for name, all_vals, intersection_vals in panels:
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_single_radar(ax, all_vals, intersection_vals)

        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=13)
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.12, top=0.95, left=0.05, right=0.95)

        for ext, fmt, kw in [
            ("svg", "svg", {}),
            ("png", "png", {"dpi": 300}),
            ("eps", "eps", {}),
        ]:
            out = OUTPUT_DIR / f"{name}.{ext}"
            plt.savefig(out, format=fmt, bbox_inches='tight', **kw)
            print(f"✅ 保存雷达图 -> {out}")

        plt.close(fig)


def plot_diagnostic_accuracy(df_int):
    model_labels = [f"LLM{i}" for i in range(1, len(df_int) + 1)]
    accuracy = (df_int["exact_match"] + df_int["partial_match"]).tolist()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_labels, accuracy, color="#87CEEA")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.3f}",
            ha='center', va='bottom', fontsize=10, color="#333333"
        )

    plt.xticks(rotation=45, ha='right', color="#333333")
    plt.yticks(color="#333333")
    plt.ylabel("Diagnostic Accuracy", color="#333333")

    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#333333")
    ax.spines['bottom'].set_color("#333333")
    ax.tick_params(colors="#333333")
    plt.gcf().patch.set_facecolor('white')

    plt.tight_layout()

    for ext, fmt, kw in [
        ("svg", "svg", {}),
        ("png", "png", {"dpi": 300}),
        ("eps", "eps", {}),
    ]:
        out = OUTPUT_DIR / f"diagnostic_accuracy.{ext}"
        plt.savefig(out, format=fmt, bbox_inches='tight', **kw)
        print(f"✅ 保存诊断正确率柱状图 -> {out}")

    plt.close()


# ========= 运行：从中间数据画图 =========
if __name__ == "__main__":
    df_int, df_all = load_intermediate_data()
    plot_radar_chart(df_int, df_all)
    plot_diagnostic_accuracy(df_int)
