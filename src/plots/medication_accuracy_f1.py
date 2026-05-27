import json
import os
import pandas as pd
import numpy as np
from scipy import stats

MODELS = [
    "gpt55", "gpt5", "gpt4o", "claude46", "claude4", "medgemma",
    "qwen35", "qwen3", "baichuan", "huatuo", "deepseekv4", "deepseek"
]

JSONL_TEMPLATE = "../../data/eval/{model}.jsonl"
CORRECT_JSONL = "../../data/dialogue/baichuan.jsonl"
DIALOGUE_DIR = "../../data/dialogue"
DIALOGUE_FILES = {
    "gpt55": "gpt55.jsonl",
    "gpt5": "gpt5.jsonl",
    "gpt4o": "gpt4o.jsonl",
    "claude46": "claude46.jsonl",
    "claude4": "claude4.jsonl",
    "deepseekv4": "deepseekv4.jsonl",
    "deepseek": "deepseek.jsonl",
    "baichuan": "baichuan.jsonl",
    "huatuo": "huatuo.jsonl",
    "medgemma": "medgemma.jsonl",
    "qwen35": "qwen35.jsonl",
    "qwen3": "qwen3.jsonl",
}
OUT_DIR = "./outputs/medication_safety"
os.makedirs(OUT_DIR, exist_ok=True)

def read_cases(file_path):
    cases = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            key = (case.get("SubjectID"), case.get("AdmissionID"))
            if None not in key:
                cases[key] = case
    return cases

def read_recommended_drugs(dialogue_jsonl):
    rec_map = {}
    with open(dialogue_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)

            sid = case.get("id", {}).get("SubjectID")
            aid = case.get("id", {}).get("AdmissionID")
            if sid is None or aid is None:
                continue

            key = (sid, aid)

            drugs = (
                case.get("interactive_system", {})
                    .get("final_answer", {})
                    .get("recommended_drugs", [])
            )

            drugs = [x.strip().upper() for x in drugs if x.strip()]
            rec_map[key] = drugs

    return rec_map


def read_correct_answers(file_path):
    correct_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            key = (case["id"]["SubjectID"], case["id"]["AdmissionID"])
            correct_ans = case["info"].get("correct_answer", [])
            
            if isinstance(correct_ans, str):
                drugs = [x.strip().upper() for x in correct_ans.split(",") if x.strip()]
            elif isinstance(correct_ans, list):
                drugs = [x.strip().upper() for x in correct_ans if x.strip()]
            else:
                drugs = []
            
            correct_dict[key] = drugs
    return correct_dict

def compute_case_f1_from_eval(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


all_cases = {model: read_cases(JSONL_TEMPLATE.format(model=model)) for model in MODELS}
correct_answers = read_correct_answers(CORRECT_JSONL)



intersection_keys = set.intersection(*(set(cases.keys()) for cases in all_cases.values()))
print(f"交集病例数: {len(intersection_keys)}")

VALID_MATCH_LEVELS = {"exact_match", "partial_match"}
good_keys = set()

for key in intersection_keys:
    is_good = True
    for model, cases in all_cases.items():
        acc = cases[key].get("diagnostic_reasoning_score", {}) \
                        .get("diagnosis_accuracy", {})
        match_level = acc.get("match_level", "none").lower()

        if match_level not in VALID_MATCH_LEVELS:
            is_good = False
            break

    if is_good:
        good_keys.add(key)

results = []

for model in MODELS:
    cases = all_cases[model]
    dialogue_path = os.path.join(DIALOGUE_DIR, DIALOGUE_FILES[model])
    rec_drugs = read_recommended_drugs(dialogue_path)

    unsafe_count = 0
    missing_numerator = 0
    missing_denominator = 0
    total_cases = len(intersection_keys)
    ref_rate_list = []
    f1_list = []
    for key in intersection_keys:
        case = cases[key]
        med_score = case.get("medication_safety_score", {})
        details = med_score.get("details", {})
        
        unsafe_drugs = details.get("unsafe_drug_use", [])
        if unsafe_drugs > 0:
            unsafe_count += 1
        
        correct_drugs = correct_answers.get(key, [])
        
        
        recommended_drugs = rec_drugs[key]
        
        
        fp = details.get("reference_drug_deviation", 0)
        fn = details.get("missing_important_drugs", 0)
        tp = max(len(recommended_drugs) - fp, 0)
        f1 = compute_case_f1_from_eval(tp, fp, fn)
        f1_list.append(f1)

        dia_dev = details.get("drug_diagnosis_mismatch", 0)
        ref_rate = dia_dev / (len(recommended_drugs) if len(recommended_drugs) > 0 else 1)
        ref_rate_list.append(ref_rate)

    
    unsafe_rate = unsafe_count / total_cases if total_cases > 0 else 0
    avg_f1 = sum(f1_list) / len(f1_list)
    avg_error_rate = sum(ref_rate_list) / len(ref_rate_list) if ref_rate_list else 0
    
    f1_std = np.std(f1_list) if f1_list else 0
    f1_ci = stats.t.interval(0.95, len(f1_list)-1, loc=avg_f1, scale=stats.sem(f1_list)) if len(f1_list) > 1 else (0, 0)
    
    error_std = np.std(ref_rate_list) if ref_rate_list else 0
    error_ci = stats.t.interval(0.95, len(ref_rate_list)-1, loc=avg_error_rate, scale=stats.sem(ref_rate_list)) if len(ref_rate_list) > 1 else (0, 0)
    
    results.append({
        "model": model,
        "avg_f1": round(avg_f1, 3),
        "f1_std": round(f1_std, 3),
        "f1_ci_lower": round(f1_ci[0], 3),
        "f1_ci_upper": round(f1_ci[1], 3),
        "unsafe_drug_use_rate": round(unsafe_rate, 3),
        "drug_diagnosis_mismatch_error_rate": round(avg_error_rate, 3),
        "error_std": round(error_std, 3),
        "error_ci_lower": round(error_ci[0], 3),
        "error_ci_upper": round(error_ci[1], 3),
        "total_cases": total_cases
    })

df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, "medication_safety_intersection.csv")
df.to_csv(out_path, index=False)
print(f"✅ 保存 Medication Safety 指标 -> {out_path}")
print(df)

import matplotlib.pyplot as plt

model_name_map = {
    "claude4": "Claude 4.0 Sonnet",
    "claude46": "Claude 4.6",
    "gpt4o": "GPT-4o",
    "gpt5": "GPT-5",
    'gpt55': "GPT-5.5",
    "deepseekv4": "DeepSeek-V4-Pro",
    "deepseek": "DeepSeek-V3",
    "qwen35": "Qwen3.5",
    "qwen3": "Qwen3",
    "medgemma": "Medgemma",
    "baichuan": "Baichuan-M2",
    "huatuo": "HuatuoGPT-o1"
}

df["model"] = df["model"].map(model_name_map)

models_order = ["Claude 4.0 Sonnet", "Claude 4.6", "GPT-4o", "GPT-5", "GPT-5.5", "DeepSeek-V4-Pro", "DeepSeek-V3", "Qwen3.5", "Qwen3", "Medgemma", "Baichuan-M2", "HuatuoGPT-o1"]
df = df.set_index("model").reindex(models_order).reset_index()

models = df["model"].tolist()
f1 = df["avg_f1"].tolist()
f1_ci_lower = df["f1_ci_lower"].tolist()
f1_ci_upper = df["f1_ci_upper"].tolist()
unsafe = df["unsafe_drug_use_rate"].tolist()
mismatch = df["drug_diagnosis_mismatch_error_rate"].tolist()
error_ci_lower = df["error_ci_lower"].tolist()
error_ci_upper = df["error_ci_upper"].tolist()

safety = [1 - x for x in unsafe]
accuracy = [1 - x for x in mismatch]

accuracy_ci_lower = [1 - x for x in error_ci_upper]
accuracy_ci_upper = [1 - x for x in error_ci_lower]

x = np.arange(len(models))
width = 0.6

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))

bars1 = ax1.bar(x, f1, width, color='#1B9C72')
for i, (xi, y, ymin, ymax) in enumerate(zip(x, f1, f1_ci_lower, f1_ci_upper)):
    ax1.errorbar(xi, y, yerr=[[y - ymin], [ymax - y]], 
                 fmt='none', c='black', capsize=3, elinewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30, fontsize=15)
# ax1.set_title('Medication F1 Score', fontsize=16)
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='y', labelsize=13)
ax1.bar_label(bars1, fmt='%.3f', padding=8, fontsize=13)

bars2 = ax2.bar(x, safety, width, color='#D85F06')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, fontsize=15)
# ax2.set_title('Medication Safety (1 - Unsafe Rate)', fontsize=16)
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='y', labelsize=13)
ax2.bar_label(bars2, fmt='%.3f', padding=8, fontsize=13)

bars3 = ax3.bar(x, accuracy, width, color='#766FB1')
for i, (xi, y, ymin, ymax) in enumerate(zip(x, accuracy, accuracy_ci_lower, accuracy_ci_upper)):
    ax3.errorbar(xi, y, yerr=[[y - ymin], [ymax - y]], 
                 fmt='none', c='black', capsize=3, elinewidth=1)
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=30, fontsize=15)
# ax3.set_title('Drug-Diagnosis Agreement (1 - Mismatch Rate)', fontsize=16)
ax3.set_ylim(0, 1.1)
ax3.tick_params(axis='y', labelsize=13)
ax3.bar_label(bars3, fmt='%.3f', padding=8, fontsize=13)

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar_chart.svg"), format="svg", bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar_chart.png"), dpi=300, bbox_inches='tight')
print(f"✅ 保存模型比较图表 -> {os.path.join(OUT_DIR, 'model_comparison_bar_chart.svg')}")