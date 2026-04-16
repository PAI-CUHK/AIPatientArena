import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 8 个模型文件路径
MODEL_NAMES = ["gpt5", "gpt4o", "claude4", "medgemma", "qwen3", "baichuan", "huatuo", "deepseek"]
MODEL_LABELS = ["GPT-5", "GPT-4o", "Claude 4", "MedGemma", "Qwen3", "Baichuan", "HuatuoGPT", "DeepSeek"]
EVAL_DIR = Path("../../data/eval")

OUTPUT_DIR = Path("./outputs/diagnosis_accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取并筛选 JSONL 文件，只保留 determinable == True 的记录
def read_and_filter_cases(model):
    input_path = EVAL_DIR / f"{model}.jsonl"
    cases = {}
    all_cases = {}
    
    total_records = 0
    determinable_true_count = 0
    missing_diagnostic_reasoning = 0
    
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total_records += 1
            record = json.loads(line)
            
            # 保存所有病例
            key = (record.get("SubjectID"), record.get("AdmissionID"))
            if None not in key:
                all_cases[key] = record
            
            diagnostic_reasoning = record.get("diagnostic_reasoning_score")
            
            if not isinstance(diagnostic_reasoning, dict):
                missing_diagnostic_reasoning += 1
                continue
            
            determinable = diagnostic_reasoning.get("determinable", False)
            
            if determinable is True:
                determinable_true_count += 1
                key = (record.get("SubjectID"), record.get("AdmissionID"))
                if None not in key:
                    cases[key] = record
    
    print(f"{model}: 总记录数={total_records}, 可确定记录数={determinable_true_count}, 缺失诊断推理={missing_diagnostic_reasoning}")
    return cases, all_cases

# 读取所有模型的筛选后数据
all_cases = {}
all_cases_all = {}
for model in MODEL_NAMES:
    cases, all_case = read_and_filter_cases(model)
    all_cases[model] = cases
    all_cases_all[model] = all_case

# 找交集病例（所有模型都包含的 SubjectID+AdmissionID）
intersection_keys = set.intersection(*(set(cases.keys()) for cases in all_cases.values()))
print(f"\n交集病例数: {len(intersection_keys)}")

# 计算诊断准确率和评分平均分
results = []
all_case_results = []

for model, cases in all_cases.items():
    exact_match_count = 0
    partial_match_count = 0
    diagnostic_reasoning_scores = []
    medication_safety_scores = []
    
    for key in intersection_keys:
        case = cases[key]
        # 计算诊断准确率
        acc = case.get("diagnostic_reasoning_score", {}).get("diagnosis_accuracy", {})
        match_level = acc.get("match_level", "none").lower()
        
        if match_level == "exact_match":
            exact_match_count += 1
        elif match_level == "partial_match":
            partial_match_count += 1
        
        # 收集评分数据
        diagnostic_reasoning = case.get("diagnostic_reasoning_score", {})
        if isinstance(diagnostic_reasoning, dict) and "score" in diagnostic_reasoning:
            diagnostic_reasoning_scores.append(diagnostic_reasoning["score"])
        
        medication_safety = case.get("medication_safety_score", {})
        if isinstance(medication_safety, dict) and "score" in medication_safety:
            medication_safety_scores.append(medication_safety["score"])
    
    total = len(intersection_keys)
    
    # 计算平均分
    diagnostic_reasoning_avg = sum(diagnostic_reasoning_scores) / len(diagnostic_reasoning_scores) if diagnostic_reasoning_scores else 0
    medication_safety_avg = sum(medication_safety_scores) / len(medication_safety_scores) if medication_safety_scores else 0
    
    results.append({
        "model": model,
        "exact_match": exact_match_count / total,
        "partial_match": partial_match_count / total,
        "diagnostic_reasoning_avg": diagnostic_reasoning_avg,
        "medication_safety_avg": medication_safety_avg,
        "total_cases": total
    })

# 计算所有病例的得分
for model, cases in all_cases_all.items():
    diagnostic_reasoning_scores = []
    medication_safety_scores = []
    
    for case in cases.values():
        # 收集评分数据
        diagnostic_reasoning = case.get("diagnostic_reasoning_score", {})
        if isinstance(diagnostic_reasoning, dict) and "score" in diagnostic_reasoning:
            diagnostic_reasoning_scores.append(diagnostic_reasoning["score"])
        
        medication_safety = case.get("medication_safety_score", {})
        if isinstance(medication_safety, dict) and "score" in medication_safety:
            medication_safety_scores.append(medication_safety["score"])
    
    # 计算平均分
    diagnostic_reasoning_avg = sum(diagnostic_reasoning_scores) / len(diagnostic_reasoning_scores) if diagnostic_reasoning_scores else 0
    medication_safety_avg = sum(medication_safety_scores) / len(medication_safety_scores) if medication_safety_scores else 0
    
    all_case_results.append({
        "model": model,
        "diagnostic_reasoning_avg": diagnostic_reasoning_avg,
        "medication_safety_avg": medication_safety_avg,
        "total_cases": len(cases)
    })

# 保存结果表格
df = pd.DataFrame(results)
out_csv = OUTPUT_DIR / "diagnosis_accuracy_intersection.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ 保存诊断准确率结果 -> {out_csv}")
print(df)

# 保存所有病例的结果
df_all = pd.DataFrame(all_case_results)
out_csv_all = OUTPUT_DIR / "diagnosis_accuracy_all.csv"
df_all.to_csv(out_csv_all, index=False)
print(f"\n✅ 保存所有病例结果 -> {out_csv_all}")
print(df_all)

# 绘制雷达图
def plot_radar_chart():
    # 准备数据
    models = MODEL_LABELS
    num_vars = len(models)
    
    # 获取数据
    intersection_dr = [r["diagnostic_reasoning_avg"] for r in results]
    intersection_ms = [r["medication_safety_avg"] for r in results]
    all_dr = [r["diagnostic_reasoning_avg"] for r in all_case_results]
    all_ms = [r["medication_safety_avg"] for r in all_case_results]
    
    # 雷达图范围
    r_min = 2
    r_max = 4
    r_step = 0.5
    r_range = r_max - r_min
    
    # 不排序，使用原始模型顺序
    sorted_models_dr = models
    sorted_all_dr = all_dr
    sorted_intersection_dr = intersection_dr
    
    sorted_models_ms = models
    sorted_all_ms = all_ms
    sorted_intersection_ms = intersection_ms
    
    # 数据偏移，使中心对应最小值
    all_dr_offset = np.array(all_dr) - r_min
    intersection_dr_offset = np.array(intersection_dr) - r_min
    all_ms_offset = np.array(all_ms) - r_min
    intersection_ms_offset = np.array(intersection_ms) - r_min
    
    # 八边形角度 - 从顶部开始逆时针排列
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, num_vars, endpoint=False)
    angles = np.append(angles, angles[0])  # 闭合
    all_dr_offset = np.append(all_dr_offset, all_dr_offset[0])
    intersection_dr_offset = np.append(intersection_dr_offset, intersection_dr_offset[0])
    all_ms_offset = np.append(all_ms_offset, all_ms_offset[0])
    intersection_ms_offset = np.append(intersection_ms_offset, intersection_ms_offset[0])
    
    # 创建画布 - 诊断推理雷达图
    fig1, ax1 = plt.subplots(figsize=(7,7))
    
    # 绘制八边形网格
    for r in np.arange(0, r_range + 0.01, r_step):
        x = (r / r_range) * np.cos(angles)
        y = (r / r_range) * np.sin(angles)
        ax1.plot(x, y, color='gray', linestyle='--', linewidth=0.5)
        
        # 刻度值显示在每条边中点，避开轴线
        for i in range(num_vars):
            # 中点角度
            mid_angle = (angles[i] + angles[i+1])/2
            x_text = (r / r_range) * np.cos(mid_angle) * 1.05
            y_text = (r / r_range) * np.sin(mid_angle) * 1.05
            if i == 0:  # 只标一次，避免重复
                ax1.text(x_text, y_text, f"{r + r_min:.1f}", fontsize=9, ha='center', va='center')
    
    # 绘制径向线
    for a in angles[:-1]:
        ax1.plot([0, np.cos(a)], [0, np.sin(a)], color='gray', linestyle='--', linewidth=0.5)
    
    # 绘制数据
    x_all_dr = (all_dr_offset / r_range) * np.cos(angles)
    y_all_dr = (all_dr_offset / r_range) * np.sin(angles)
    ax1.plot(x_all_dr, y_all_dr, 'o-', color="#91A8D5", linewidth=2)
    ax1.fill(x_all_dr, y_all_dr, color="#91A8D5", alpha=0.25)
    
    x_intersection_dr = (intersection_dr_offset / r_range) * np.cos(angles)
    y_intersection_dr = (intersection_dr_offset / r_range) * np.sin(angles)
    ax1.plot(x_intersection_dr, y_intersection_dr, 's-', color="#DA99AB", linewidth=2)
    ax1.fill(x_intersection_dr, y_intersection_dr, color="#DA99AB", alpha=0.25)
    
    # 模型标签
    for i, label in enumerate(sorted_models_dr):
        x_text = 1.1 * np.cos(angles[i])
        y_text = 1.1 * np.sin(angles[i])
        ax1.text(x_text, y_text, label, ha='center', va='center', fontsize=10)
    
    # 去掉坐标轴
    ax1.axis('off')
    ax1.set_aspect('equal')
    
    # 保存诊断推理雷达图
    output_file1 = OUTPUT_DIR / "diagnostic_reasoning_radar.svg"
    plt.savefig(output_file1, format="svg", bbox_inches='tight')
    print(f"\n✅ 保存诊断推理雷达图 -> {output_file1}")
    
    # 创建画布 - 药物安全性雷达图
    fig2, ax2 = plt.subplots(figsize=(7,7))
    
    # 绘制八边形网格
    for r in np.arange(0, r_range + 0.01, r_step):
        x = (r / r_range) * np.cos(angles)
        y = (r / r_range) * np.sin(angles)
        ax2.plot(x, y, color='gray', linestyle='--', linewidth=0.5)
        
        # 刻度值显示在每条边中点，避开轴线
        for i in range(num_vars):
            # 中点角度
            mid_angle = (angles[i] + angles[i+1])/2
            x_text = (r / r_range) * np.cos(mid_angle) * 1.05
            y_text = (r / r_range) * np.sin(mid_angle) * 1.05
            if i == 0:  # 只标一次，避免重复
                ax2.text(x_text, y_text, f"{r + r_min:.1f}", fontsize=9, ha='center', va='center')
    
    # 绘制径向线
    for a in angles[:-1]:
        ax2.plot([0, np.cos(a)], [0, np.sin(a)], color='gray', linestyle='--', linewidth=0.5)
    
    # 绘制数据
    x_all_ms = (all_ms_offset / r_range) * np.cos(angles)
    y_all_ms = (all_ms_offset / r_range) * np.sin(angles)
    ax2.plot(x_all_ms, y_all_ms, 'o-', color="#91A8D5", linewidth=2)
    ax2.fill(x_all_ms, y_all_ms, color="#91A8D5", alpha=0.25)
    
    x_intersection_ms = (intersection_ms_offset / r_range) * np.cos(angles)
    y_intersection_ms = (intersection_ms_offset / r_range) * np.sin(angles)
    ax2.plot(x_intersection_ms, y_intersection_ms, 's-', color="#DA99AB", linewidth=2)
    ax2.fill(x_intersection_ms, y_intersection_ms, color="#DA99AB", alpha=0.25)
    
    # 模型标签
    for i, label in enumerate(sorted_models_ms):
        x_text = 1.1 * np.cos(angles[i])
        y_text = 1.1 * np.sin(angles[i])
        ax2.text(x_text, y_text, label, ha='center', va='center', fontsize=10)
    
    # 去掉坐标轴
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    # 保存药物安全性雷达图
    output_file2 = OUTPUT_DIR / "medication_safety_radar.svg"
    plt.savefig(output_file2, format="svg", bbox_inches='tight')
    print(f"✅ 保存药物安全性雷达图 -> {output_file2}")

# 绘制诊断正确率柱状图
def plot_diagnostic_accuracy():
    # 计算每个模型的诊断正确率
    model_labels = []
    accuracy = []
    
    # 模型名称映射
    model_name_map = {
        "gpt5": "GPT-5",
        "gpt4o": "GPT-4o",
        "claude4": "Claude 4.0 Sonnet",
        "medgemma": "MedGemma",
        "qwen3": "Qwen3",
        "baichuan": "Baichuan-M2",
        "huatuo": "HuatuoGPT-o1",
        "deepseek": "DeepSeek-V3"
    }
    
    for r in results:
        model = r["model"]
        model_labels.append(model_name_map.get(model, model))
        # 计算诊断正确率：(exact_match + partial_match) / total
        acc = (r["exact_match"] + r["partial_match"])
        accuracy.append(acc)
    
    # 不排序，使用原始模型顺序
    # 绘制柱状图
    plt.figure(figsize=(10,6))
    
    # Nature风格颜色
    bars = plt.bar(model_labels, accuracy, color="#87CEEA")
    
    # 数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.3f}",
            ha='center', va='bottom', fontsize=10
        )
    
    # 坐标轴优化（Nature风格关键）
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Diagnostic Accuracy")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存为 SVG
    output_file = OUTPUT_DIR / "diagnostic_accuracy.svg"
    plt.savefig(output_file, format="svg", bbox_inches='tight')
    print(f"✅ 保存诊断正确率柱状图 -> {output_file}")

# 调用绘图函数
plot_radar_chart()
plot_diagnostic_accuracy()