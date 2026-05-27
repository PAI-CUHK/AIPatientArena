import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ========= 1. 数据配置与映射 =========
MODELS = ['gpt55',"gpt5", "gpt4o", 'claude46', "claude4", "medgemma", 'qwen35', "qwen3", "baichuan", "huatuo", 'deepseekv4', "deepseek"]
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

JSONL_TEMPLATE = "../../data/eval_52/{model}.jsonl"
OUT_DIR = "./outputs/error_type_5domains"
os.makedirs(OUT_DIR, exist_ok=True)

NATURE_PALETTE = ["#247BA0", "#70C1B3", "#B2DBBF", "#F3FFBD", "#FF1654", "#9575CD"]

# ========= 2. 数据处理逻辑 =========
def read_cases(path):
    cases = {}
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            case = json.loads(line)
            key = (case.get("SubjectID"), case.get("AdmissionID"))
            if None not in key: cases[key] = case
    return cases

all_cases = {m: read_cases(JSONL_TEMPLATE.format(model=m)) for m in MODELS}
intersection_keys = set.intersection(*(set(c.keys()) for c in all_cases.values()))
N = len(intersection_keys)

rows = []
raw_domains_found = []
for model in MODELS:
    cases = all_cases[model]
    counters = {}
    for key in intersection_keys:
        case = cases[key]
        for k, v in case.items():
            if not k.endswith("_score"): continue
            raw_dom = k.replace("_score", "").replace("_", " ").title().strip()
            if raw_dom not in raw_domains_found: raw_domains_found.append(raw_dom)
            if raw_dom not in counters: counters[raw_dom] = Counter()

            if "diagnostic_reasoning" in k.lower():
                for err, cnt in v.get("reasoning_errors", {}).items():
                    if cnt > 0: counters[raw_dom][err] += 1
            elif "information_coverage" in k.lower():
                for aspect, info in v.get("explanation", {}).items():
                    if isinstance(info, dict) and info.get("status", "").startswith("missing"):
                        counters[raw_dom][aspect] += 1
            elif "self_awareness" in k.lower():
                for aspect, info in v.get("explanation", {}).items():
                    if isinstance(info, dict) and info.get("status", "").startswith("missing"):
                        counters[raw_dom][aspect] += 1
            else:
                for err, cnt in v.get("details", {}).items():
                    if cnt > 0: counters[raw_dom][err] += 1
    for dom, counter in counters.items():
        for err, cnt in counter.items():
            rows.append({"Model": MODEL_DISPLAY_NAMES[model], "RawDomain": dom, "ErrorType": err, "Prob": cnt/N})

df_all = pd.DataFrame(rows)

# ========= 3. 绘图引擎 =========
def draw_nature_radial_tight(df_dom, display_title, save_path):
    error_types = (df_dom.groupby("ErrorType")["Prob"].mean()
                   .sort_values(ascending=False).head(6).index.tolist())

    model_list = ["HT-o1", "BC-M2", "MedGem", "Qwen3", "Qwen3.5", "DS-V3", "DS-V4", "Claude 4.0", "Claude 4.6", "GPT-4o", "GPT-5", "GPT-5.5"]

    n_groups = len(error_types)
    n_models = len(model_list)

    group_gap = 0.2
    total_bars = n_groups * n_models
    available_whisker = 2 * np.pi - (n_groups * group_gap)
    bar_width = available_whisker / total_bars

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rorigin(-0.4)

    for r in [0.2, 0.4, 0.6, 0.8]:
        ax.add_patch(plt.Circle((0, 0), r, transform=ax.transData._b, fill=False,
                                edgecolor='#CCCCCC', linestyle='--', linewidth=0.7, zorder=1))

    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.text(0, r, f"{r}", color="#888888", fontsize=11, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    current_angle = 0

    for i, err in enumerate(error_types):
        color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
        err_df = df_dom[df_dom["ErrorType"] == err].set_index("Model").reindex(model_list).fillna(0)

        group_start_angle = current_angle

        for j, m_name in enumerate(model_list):
            val = err_df.loc[m_name, 'Prob']

            ax.bar(current_angle + bar_width/2, val, width=bar_width*0.9, color=color,
                   alpha=0.8, edgecolor='none', zorder=3)

            angle_deg = np.rad2deg(current_angle + bar_width/2)
            rot = -angle_deg + 90

            if 90 < angle_deg < 270:
                rot += 180
                va = 'top'
            else:
                va = 'bottom'

            ax.text(current_angle + bar_width/2, 1.05, m_name,
                    rotation=rot, rotation_mode='anchor',
                    ha='center', va=va, fontsize=13, fontweight='bold')

            current_angle += bar_width

        group_end_angle = current_angle - bar_width

        r_bracket = 1.35
        bracket_angles = np.linspace(group_start_angle, group_end_angle, 100)
        ax.plot(bracket_angles, [r_bracket]*100, color='black', linewidth=2)
        ax.plot([group_start_angle, group_start_angle], [r_bracket, r_bracket-0.05], color='black', linewidth=2)
        ax.plot([group_end_angle, group_end_angle], [r_bracket, r_bracket-0.05], color='black', linewidth=2)

        mid_angle = (group_start_angle + group_end_angle) / 2
        mid_deg = np.rad2deg(mid_angle)
        t_rot = -mid_deg
        if 90 < mid_deg < 270: t_rot += 180

        formatted_err = err.replace("_", " ").capitalize()
        
        ax.text(mid_angle, r_bracket + 0.08, formatted_err,
                rotation=t_rot, rotation_mode='anchor',
                ha='center', va='center', fontsize=18, fontweight='bold')

        current_angle += group_gap

    ax.set_xticks([]); ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    plt.title(display_title, pad=60, fontsize=24, fontweight='bold')
    plt.savefig(save_path, format="svg", bbox_inches='tight', dpi=300)
    plt.close()

# ========= 4. 运行 =========
for i in range(1, 6):
    raw_key = f"Domain {i}"
    new_title = DOMAIN_MAP.get(raw_key)
    if i-1 < len(raw_domains_found):
        actual_raw_dom = raw_domains_found[i-1]
        df_sub = df_all[df_all["RawDomain"] == actual_raw_dom]
        if not df_sub.empty:
            save_path = os.path.join(OUT_DIR, f"radial_{raw_key.replace(' ', '')}.svg")
            draw_nature_radial_tight(df_sub, new_title, save_path)
            print(f"Generated: {new_title}")

# ========= 5. 输出详细表格 =========
print("\n" + "="*80)
print("📊 各模型在各Domain下的Error Type错误概率详细表格")
print("="*80)

for i in range(1, 6):
    raw_key = f"Domain {i}"
    new_title = DOMAIN_MAP.get(raw_key)

    if i-1 < len(raw_domains_found):
        actual_raw_dom = raw_domains_found[i-1]
        df_sub = df_all[df_all["RawDomain"] == actual_raw_dom]

        if not df_sub.empty:
            print(f"\n【{new_title}】")
            pivot_df = df_sub.pivot_table(index="Model", columns="ErrorType", values="Prob", aggfunc="mean")
            pivot_df = pivot_df.round(4)
            print(pivot_df.to_string())

pivot_all = df_all.pivot_table(index="Model", columns=["RawDomain", "ErrorType"], values="Prob", aggfunc="mean")
pivot_all = pivot_all.round(4)

csv_path = os.path.join(OUT_DIR, "all_models_error_probs.csv")
pivot_all.to_csv(csv_path)
print(f"\n✅ 完整表格已保存至: {csv_path}")