import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from pathlib import Path

# ===== 配置 =====
EVAL_DIR = Path("/media/disk1/njh/aipatient_eval/aipatient_eval_5880/data/eval")
MODEL_NAMES = ["gpt55", "gpt5", "gpt4o", "claude46", "claude4", "medgemma", "qwen35", "qwen3", "baichuan", "huatuo", "deepseekv4", "deepseek"]
MODEL_LABELS = ["GPT-5.5", "GPT-5", "GPT-4o", "Claude 4.6 Sonnet", "Claude 4.0 Sonnet", "Medgemma", "Qwen3.5", "Qwen3", "Baichuan-M2", "HuatuoGPT-o1", "Deepseek-V4-Pro", "Deepseek-V3"]

# ===== 维度映射 =====
dimension_map = {
    "QS": "questioning_skills_score",
    "IC": "self_awareness_score",
    "HR": "robustness_score",
    "ET": "ethics_score",
    "EX": "explainability_score",
    "II": "information_summary_score",
    "Dx": "diagnostic_reasoning_score",
    "MS": "medication_safety_score"
}

labels = ["QS","IC","HR","ET","EX","II","Dx","MS"]
num_vars = len(labels)

# ===== 按模型类型分组 =====
GROUP1_LABELS = ["GPT-5.5", "GPT-5", "GPT-4o", "Claude 4.6 Sonnet", "Claude 4.0 Sonnet"]  # 闭源大模型
GROUP2_LABELS = ["Qwen3.5", "Qwen3", "Deepseek-V4-Pro", "Deepseek-V3"]  # 开源通用模型
GROUP3_LABELS = ["Medgemma", "Baichuan-M2", "HuatuoGPT-o1"]  # 医疗/开源医疗模型

GROUP_CONFIGS = [
    {"name": "closed_source", "title": "(a) Closed-source", "models": GROUP1_LABELS},
    {"name": "open_source", "title": "(b) Open-source", "models": GROUP2_LABELS},
    {"name": "medical", "title": "(c) Medical-specialized", "models": GROUP3_LABELS},
]

# ===== 从eval文件读取并计算得分 =====
def read_and_compute_scores(model):
    input_path = EVAL_DIR / f"{model}_all_0117.jsonl"
    scores = {dim: [] for dim in dimension_map.values()}
    
    if not input_path.exists():
        print(f"⚠️ 文件不存在，跳过: {input_path}")
        return None
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            
            for dim_key, dim_name in dimension_map.items():
                dim_data = case.get(dim_name, {})
                if isinstance(dim_data, dict) and "score" in dim_data:
                    scores[dim_name].append(dim_data["score"])
    
    avg_scores = []
    for dim_name in dimension_map.values():
        if scores[dim_name]:
            avg_scores.append(sum(scores[dim_name]) / len(scores[dim_name]))
        else:
            avg_scores.append(0)
    
    return avg_scores

# ===== 固定配色 =====
base_cmap = plt.get_cmap("tab10")
colors = base_cmap.colors
colors = colors[3:] + colors[:3]

model_name_to_label = dict(zip(MODEL_NAMES, MODEL_LABELS))

fixed_colors = {}
for i, model_name in enumerate(MODEL_NAMES):
    fixed_colors[model_name] = colors[i % len(colors)]

line_colors = {}
for model_name in MODEL_NAMES:
    line_colors[model_name_to_label[model_name]] = fixed_colors[model_name]

# ===== 构建数据字典 =====
data = {}
for model, label in zip(MODEL_NAMES, MODEL_LABELS):
    scores = read_and_compute_scores(model)
    if scores is None:
        continue
    data[label] = scores
    print(f"{label} scores: {data[label]}")

# ===== 角度 =====
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# ===== 字体 =====
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams['svg.fonttype'] = 'none'

# ===== 创建画布和子图 =====
fig, axes = plt.subplots(1, 3, figsize=(20, 8), subplot_kw=dict(polar=True),
                        gridspec_kw={'wspace': 0.15})

# ===== 按子图顺序收集所有模型的图例元素 =====
legend_elements = []
for group in GROUP_CONFIGS:
    for model in group["models"]:
        if model in data:
            legend_elements.append(Line2D([0], [0], linewidth=2, color=line_colors[model], label=model))

# ===== 绘制雷达图 =====
for idx, (ax, group) in enumerate(zip(axes, GROUP_CONFIGS)):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    for model in group["models"]:
        if model not in data:
            continue
        values = data[model]
        values = values + values[:1]
        
        ax.plot(
            angles, values,
            linewidth=2,
            color=line_colors[model],
            label=model,
            zorder=2
        )
    
    outer_circle = [5] * len(angles)
    ax.plot(
        angles,
        outer_circle,
        color="#888888",
        linewidth=2,
        alpha=0.7,
        zorder=0
    )
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    
    ax.set_ylim(2, 5)
    ticks = [2, 3, 4, 5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_rlabel_position(360 / (2 * num_vars))
    
    ax.yaxis.grid(True, linewidth=0.8, alpha=0.4, color="#999999")
    ax.xaxis.grid(True, linewidth=0.5, alpha=0.2)
    ax.spines['polar'].set_visible(False)
    
    # ax.set_title(group["title"], fontsize=12, pad=20)

# ===== 底部统一图例 =====
fig.legend(handles=legend_elements, 
           loc='lower center',
           bbox_to_anchor=(0.5, 0.08),
           ncol=6,
           frameon=False,
           fontsize=11)

# ===== 调整布局 =====
plt.subplots_adjust(bottom=0.18)

# ===== 导出 =====
import os
os.makedirs("./outputs", exist_ok=True)

plt.savefig("./outputs/radar_three_groups_combined.png", dpi=300, bbox_inches="tight")
print("✅ 已保存: ./outputs/radar_three_groups_combined.png")

plt.close()

print("\n🎉 合成雷达图绘制完成！")