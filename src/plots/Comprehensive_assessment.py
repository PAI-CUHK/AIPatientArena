import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ===== 配置 =====
EVAL_DIR = Path("/media/disk1/niujiahui/aipatient_eval/data/eval")
MODEL_NAMES = ["gpt5", "gpt4o", "claude4", "medgemma", "qwen3", "baichuan", "huatuo", "deepseek"]
MODEL_LABELS = ["GPT-5", "GPT-4o", "Claude 4.0 Sonnet", "Medgemma", "Qwen3", "Baichuan-M2", "HuatuoGPT-o1", "Deepseek-V3"]

# ===== 维度映射 =====
dimension_map = {
    "QS": "questioning_skills_score",
    "IC": "self_awareness_score",
    "HR": "robustness_score",
    "ET": "ethics_score",
    "EX": "explainability_score",
    "II": "information_summary_score",
    "DR": "diagnostic_reasoning_score",
    "MS": "medication_safety_score"
}

labels = ["QS","IC","HR","ET","EX","II","DR","MS"]
num_vars = len(labels)

# ===== 从eval文件读取并计算得分 =====
def read_and_compute_scores(model):
    input_path = EVAL_DIR / f"{model}_all_0117.jsonl"
    scores = {dim: [] for dim in dimension_map.values()}
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            
            for dim_key, dim_name in dimension_map.items():
                dim_data = case.get(dim_name, {})
                if isinstance(dim_data, dict) and "score" in dim_data:
                    scores[dim_name].append(dim_data["score"])
    
    # 计算每个维度的平均分
    avg_scores = []
    for dim_name in dimension_map.values():
        if scores[dim_name]:
            avg_scores.append(sum(scores[dim_name]) / len(scores[dim_name]))
        else:
            avg_scores.append(0)
    
    return avg_scores

# 构建数据字典
data = {}
for model, label in zip(MODEL_NAMES, MODEL_LABELS):
    data[label] = read_and_compute_scores(model)
    print(f"{label} scores: {data[label]}")

# ===== 自动配色（关键改动）=====
# 使用 matplotlib 内置调色盘，自动生成 8 种颜色
import matplotlib as mpl
# cmap = mpl.colormaps.get_cmap("Set2").resampled(len(data))

# line_colors = {model: cmap(i) for i, model in enumerate(data)}
# fill_colors = {
#     model: (cmap(i)[0], cmap(i)[1], cmap(i)[2], 0.25)  # 降低透明度
#     for i, model in enumerate(data)
# }

# cmap = plt.get_cmap("tab10").resampled(len(data))

# line_colors = {model: cmap(i) for i, model in enumerate(data)}
# fill_colors = {
#     model: (cmap(i)[0], cmap(i)[1], cmap(i)[2], 0.25)
#     for i, model in enumerate(data)
# }

base_cmap = plt.get_cmap("tab10")
colors = base_cmap.colors  # 10个固定颜色

# 找到红色在 tab10 中的位置（通常 index=3）
# tab10 顺序: blue, orange, green, red, ...
colors = colors[3:] + colors[:3]

cmap_colors = colors  # 已经重排好的颜色列表

line_colors = {model: cmap_colors[i % len(cmap_colors)]
                for i, model in enumerate(data)}

fill_colors = {
    model: (*cmap_colors[i % len(cmap_colors)][:3], 0.25)
    for i, model in enumerate(data)
}

# ===== 角度 =====
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# ===== 字体 =====
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams['svg.fonttype'] = 'none'

# ===== 画布 =====
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

# ===== 方向 =====
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ===== 画模型 =====
for model, values in data.items():
    values = values + values[:1]

    # ax.fill(
    #     angles, values,
    #     color=fill_colors[model],
    #     zorder=1
    # )

    ax.plot(
        angles, values,
        linewidth=2,
        color=line_colors[model],
        label=model,
        zorder=2
    )

# ===== 最外圈 =====
outer_circle = [5] * len(angles)
ax.plot(
    angles,
    outer_circle,
    color="#888888",
    linewidth=2,
    alpha=0.7,
    zorder=0
)

# ===== 坐标 =====
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

ax.set_ylim(2,5)

ticks = [2,3,4,5]
ax.set_yticks(ticks)
ax.set_yticklabels([str(t) for t in ticks], fontsize=9)

ax.set_rlabel_position(360/(2*num_vars))

# ===== 网格 =====
ax.yaxis.grid(True, linewidth=0.8, alpha=0.4, color="#999999")
ax.xaxis.grid(True, linewidth=0.5, alpha=0.2)

ax.spines['polar'].set_visible(False)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.0, -0.15, 1.0, 0.1),  # ⭐ 关键：给一个“宽度=1”的盒子
    ncol=4,
    mode="expand",                         # ⭐ 关键：强制拉伸
    frameon=False,
    fontsize=10
)

# ===== 布局 =====
plt.tight_layout()

# ===== 导出 =====
import os

# 创建output目录
os.makedirs("./outputs", exist_ok=True)

plt.savefig("./outputs/radar_8models_full.png", dpi=300, bbox_inches="tight")

plt.close()