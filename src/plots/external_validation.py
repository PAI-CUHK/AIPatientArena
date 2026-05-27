import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from pathlib import Path

# ===== 配置 =====
BASE_DIR = Path(__file__).parent.parent.parent / "data"

DATASET_CONFIGS = [
    {"name": "eval", "dir": "eval", "filename_pattern": "{model}_all_0117.jsonl", "title": "(a) CCQA dataset"},
    {"name": "pmc_eval", "dir": "pmc_eval", "filename_pattern": "{model}.jsonl", "title": "(b) PMC dataset"},
    {"name": "shengli_eval", "dir": "shengli_eval", "filename_pattern": "{model}.jsonl", "title": "(c) PCI dataset"},
]

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
    "DR": "diagnostic_reasoning_score",
    "MS": "medication_safety_score"
}

labels = ["QS","IC","HR","ET","EX","II","DR","MS"]
num_vars = len(labels)

# ===== 从不同数据源读取并计算得分 =====
def read_and_compute_scores(model, dataset_config):
    eval_dir = BASE_DIR / dataset_config["dir"]
    filename = dataset_config["filename_pattern"].format(model=model)
    input_path = eval_dir / filename
    
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

# ===== 读取所有数据集 =====
all_data = {}
for dataset in DATASET_CONFIGS:
    dataset_name = dataset["name"]
    print(f"\n📂 读取 {dataset_name} 数据集...")
    all_data[dataset_name] = {}
    
    for model, label in zip(MODEL_NAMES, MODEL_LABELS):
        scores = read_and_compute_scores(model, dataset)
        if scores is None:
            continue
        all_data[dataset_name][label] = scores
        print(f"  {label} scores: {[round(s, 2) for s in scores]}")

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

# ===== 角度 =====
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# ===== 字体 =====
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams['svg.fonttype'] = 'none'

# ===== 创建画布和子图 =====
fig, axes = plt.subplots(1, 3, figsize=(20, 8), subplot_kw=dict(polar=True),
                        gridspec_kw={'wspace': 0.15})

# ===== 收集所有模型的图例元素 =====
legend_elements = []
for model in MODEL_LABELS:
    legend_elements.append(Line2D([0], [0], linewidth=2, color=line_colors[model], label=model))

# ===== 绘制雷达图 =====
for idx, (ax, dataset) in enumerate(zip(axes, DATASET_CONFIGS)):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    dataset_data = all_data[dataset["name"]]
    
    for model in MODEL_LABELS:
        if model not in dataset_data:
            continue
        values = dataset_data[model]
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
    
    # ax.set_title(dataset["title"], fontsize=12, pad=20)

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

plt.savefig("./outputs/radar_3datasets_combined.png", dpi=300, bbox_inches="tight")
print("\n✅ 已保存: ./outputs/radar_3datasets_combined.png")

plt.close()

print("\n🎉 合成雷达图绘制完成！")
