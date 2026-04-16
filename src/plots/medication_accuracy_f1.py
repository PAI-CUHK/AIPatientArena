import json
import os
import pandas as pd
import numpy as np
from scipy import stats

# ========== 配置 ==========
MODELS = [
    "gpt5", "gpt4o", "claude4", "medgemma",
    "qwen3", "baichuan", "huatuo", "deepseek"
]

JSONL_TEMPLATE = "../../data/eval/{model}.jsonl"
CORRECT_JSONL = "../../data/dialogue/baichuan.jsonl"
DIALOGUE_DIR = "../../data/dialogue"
DIALOGUE_FILES = {
    "gpt5": "gpt5.jsonl",
    "gpt4o": "gpt4o.jsonl",
    "claude4": "claude4.jsonl",
    "deepseek": "deepseek.jsonl",
    "baichuan": "baichuan.jsonl",
    "huatuo": "huatuo.jsonl",
    "medgemma": "medgemma.jsonl",
    "qwen3": "qwen3.jsonl",
}
OUT_DIR = "./outputs/medication_safety"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 辅助函数 ==========
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

            # 统一大写 + strip
            drugs = [x.strip().upper() for x in drugs if x.strip()]
            rec_map[key] = drugs

    return rec_map


def read_correct_answers(file_path):
    """返回 dict: key=(SubjectID, AdmissionID), value=list of correct drugs"""
    correct_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            key = (case["id"]["SubjectID"], case["id"]["AdmissionID"])
            correct_ans = case["info"].get("correct_answer", [])
            
            # 判断类型
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


# ========== 读取数据 ==========
all_cases = {model: read_cases(JSONL_TEMPLATE.format(model=model)) for model in MODELS}
correct_answers = read_correct_answers(CORRECT_JSONL)



# 找交集病例（所有模型都包含的 SubjectID+AdmissionID）
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

# intersection_keys = good_keys

# ========== 计算指标 ==========
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
    # print(intersection_keys)
    for key in intersection_keys:
        case = cases[key]
        med_score = case.get("medication_safety_score", {})
        details = med_score.get("details", {})
        
        # 是否存在 unsafe drug use
        unsafe_drugs = details.get("unsafe_drug_use", [])
        if unsafe_drugs > 0:
            unsafe_count += 1
        
        # correct answer 从单独文件里获取
        correct_drugs = correct_answers.get(key, [])
        
        
        # recommended drugs
        recommended_drugs = rec_drugs[key]
        # print(recommended_drugs)
        
        # missing drugs
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
    
    # 计算分布和置信度区间
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

# ========== 保存结果 ==========
df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, "medication_safety_intersection.csv")
df.to_csv(out_path, index=False)
print(f"✅ 保存 Medication Safety 指标 -> {out_path}")
print(df)

# ========== 绘制图表 ==========
import matplotlib.pyplot as plt

# 模型名称映射
model_name_map = {
    "claude4": "Claude 4.0 Sonnet",
    "gpt4o": "GPT-4o",
    "gpt5": "GPT-5",
    "deepseek": "DeepSeek-V3",
    "qwen3": "Qwen3",
    "medgemma": "Medgemma",
    "baichuan": "Baichuan-M2",
    "huatuo": "HuatuoGPT-o1"
}

# 应用模型名称映射
df["model"] = df["model"].map(model_name_map)

# 按指定顺序排列模型
models_order = ["Claude 4.0 Sonnet", "GPT-4o", "GPT-5", "DeepSeek-V3", "Qwen3", "Medgemma", "Baichuan-M2", "HuatuoGPT-o1"]
df = df.set_index("model").reindex(models_order).reset_index()

# 提取数据
models = df["model"].tolist()
f1 = df["avg_f1"].tolist()
f1_std = df["f1_std"].tolist()
f1_ci_lower = df["f1_ci_lower"].tolist()
f1_ci_upper = df["f1_ci_upper"].tolist()
unsafe = df["unsafe_drug_use_rate"].tolist()
mismatch = df["drug_diagnosis_mismatch_error_rate"].tolist()
error_std = df["error_std"].tolist()
error_ci_lower = df["error_ci_lower"].tolist()
error_ci_upper = df["error_ci_upper"].tolist()

# Convert to "higher is better"
safety = [1 - x for x in unsafe]
accuracy = [1 - x for x in mismatch]

# 计算安全和准确性的置信区间
safety_ci_lower = [1 - x for x in unsafe]
safety_ci_upper = [1 - x for x in unsafe]
accuracy_ci_lower = [1 - x for x in error_ci_upper]
accuracy_ci_upper = [1 - x for x in error_ci_lower]

x = np.arange(len(models)) * 1.5  # 增加模型之间的间距
width = 0.45  # 保持柱子宽度
offset = width + 0.01

# 计算误差范围
f1_error = [(y - ymin) for y, ymin in zip(f1, f1_ci_lower)]
accuracy_error = [(y - ymin) for y, ymin in zip(accuracy, accuracy_ci_lower)]

# 调整图表大小，为标签提供更多空间
plt.figure(figsize=(20, 6))

# 绘制 F1 柱状图并添加置信区间
bars1 = plt.bar(x - offset, f1, width, 
                label='Avg F1', 
                color='#1B9C72')

# 添加 F1 置信区间
for i, (x_pos, y, ymin, ymax) in enumerate(zip(x - offset, f1, f1_ci_lower, f1_ci_upper)):
    plt.errorbar(x_pos, y, yerr=[[y - ymin], [ymax - y]], 
                 fmt='none', c='black', capsize=3, elinewidth=1)

# 绘制安全柱状图
bars2 = plt.bar(x, safety, width, 
                label='Safety (1 - Unsafe Rate)', 
                color='#D85F06')

# 绘制准确性柱状图并添加置信区间
bars3 = plt.bar(x + offset, accuracy, width, 
                label='Agreement (1 - Mismatch Rate)', 
                color='#766FB1')

# 添加准确性置信区间
for i, (x_pos, y, ymin, ymax) in enumerate(zip(x + offset, accuracy, accuracy_ci_lower, accuracy_ci_upper)):
    plt.errorbar(x_pos, y, yerr=[[y - ymin], [ymax - y]], 
                 fmt='none', c='black', capsize=3, elinewidth=1)

plt.xticks(x, models, rotation=30)

# 调整数据标签位置和格式，避免与置信度区间重叠
# F1 分数标签：平均值 ± 误差（使用三位小数）
f1_labels = [f"{y:.3f}±{err:.3f}" for y, err in zip(f1, f1_error)]
plt.bar_label(bars1, labels=f1_labels, padding=10, fontsize=8)  # 减小字体大小，增加padding

# 安全分数标签：直接显示数值
plt.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)  # 减小字体大小

# 准确率标签：平均值 ± 误差（使用三位小数）
accuracy_labels = [f"{y:.3f}±{err:.3f}" for y, err in zip(accuracy, accuracy_error)]
plt.bar_label(bars3, labels=accuracy_labels, padding=10, fontsize=8)  # 减小字体大小，增加padding

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar_chart.svg"), format="svg", bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar_chart.png"), dpi=300, bbox_inches='tight')
print(f"✅ 保存模型比较图表 -> {os.path.join(OUT_DIR, 'model_comparison_bar_chart.svg')}")
