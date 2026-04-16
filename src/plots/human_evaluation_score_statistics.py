#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算指定目录下所有子文件夹的平均分，并汇总总体平均分，同时生成箱线图
"""

import os
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_case_level_scores(folder_path):
    """
    加载病例级评分数据
    
    Returns:
        pd.DataFrame: 包含病例级评分数据的DataFrame
    """
    records = []

    print(f"🔍 扫描目录: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ 目录不存在: {folder_path}")
        return pd.DataFrame()

    for reviewer in os.listdir(folder_path):
        reviewer_dir = os.path.join(folder_path, reviewer)
        if not os.path.isdir(reviewer_dir):
            continue

        print(f"📂 处理评审员: {reviewer}")
        file_count = 0
        
        for fname in os.listdir(reviewer_dir):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue

            fpath = os.path.join(reviewer_dir, fname)
            
            try:
                # 处理 .jsonl 文件
                if fname.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    file_count += 1
                                    
                                    case_id = f"{data.get('SubjectID', 'unknown')}_{data.get('AdmissionID', 'unknown')}"

                                    for key, value in data.items():
                                        if not key.endswith("_review"):
                                            continue

                                        if not isinstance(value, dict) or "average_score" not in value:
                                            continue

                                        records.append({
                                            "case_id": case_id,
                                            "reviewer": reviewer,
                                            "dimension": key,
                                            "score": value["average_score"]
                                        })
                                except json.JSONDecodeError:
                                    continue
                
                # 处理 .json 文件
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        file_count += 1

                    case_id = f"{data.get('SubjectID', 'unknown')}_{data.get('AdmissionID', 'unknown')}"

                    for key, value in data.items():
                        if not key.endswith("_review"):
                            continue

                        if not isinstance(value, dict) or "average_score" not in value:
                            continue

                        records.append({
                            "case_id": case_id,
                            "reviewer": reviewer,
                            "dimension": key,
                            "score": value["average_score"]
                        })
            
            except Exception as e:
                print(f"⚠️  处理文件 {fname} 时出错: {e}")
                continue
        
        print(f"   ✅ 处理了 {file_count} 个文件")

    print(f"\n📊 总共收集了 {len(records)} 条记录")
    return pd.DataFrame(records)

def calculate_folder_average(folder_path):
    """
    计算单个文件夹的平均分
    
    Returns:
        dict: 包含各维度平均分的字典，如果没有数据则返回None
    """
    df = load_case_level_scores(folder_path)
    
    if df.empty:
        return None
    
    # 计算病例级平均分
    case_level = (
        df
        .groupby(["case_id", "dimension"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "case_mean_score"})
    )
    
    # 计算每个维度的平均分
    dimension_avg = {}
    for dim, group in case_level.groupby('dimension'):
        dimension_avg[dim] = round(group['case_mean_score'].mean(), 2)
    
    # 计算总体平均分
    overall_avg = 0
    if dimension_avg:
        overall_avg = round(sum(dimension_avg.values()) / len(dimension_avg), 2)
    
    return {
        "total_cases": len(case_level['case_id'].unique()),
        "dimension_averages": dimension_avg,
        "overall_average": overall_avg
    }

def calculate_all_folders(base_path):
    """
    计算基础路径下所有子文件夹的平均分
    
    Returns:
        dict: 包含所有子文件夹结果的字典
    """
    results = {}
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, f))]
    
    if not subfolders:
        print(f"❌ 在 {base_path} 下没有找到子文件夹")
        return None
    
    print(f"📁 找到 {len(subfolders)} 个子文件夹: {', '.join(subfolders)}\n")
    
    # 计算每个子文件夹的平均分
    for subfolder in sorted(subfolders):
        folder_path = os.path.join(base_path, subfolder)
        print(f"🔍 处理文件夹: {subfolder}")
        
        result = calculate_folder_average(folder_path)
        
        if result:
            results[subfolder] = result
            print(f"   ✅ {result['total_cases']} 个病例, 总体平均分: {result['overall_average']}")
        else:
            print(f"   ⚠️  没有找到有效数据")
        print()
    
    return results

def calculate_grand_average(results):
    """
    计算所有文件夹的总体平均分
    
    Returns:
        dict: 包含总体统计的字典
    """
    if not results:
        return None
    
    # 收集所有维度的分数
    all_dimension_scores = defaultdict(list)
    total_cases = 0
    
    for folder_name, folder_result in results.items():
        total_cases += folder_result["total_cases"]
        
        for dim, avg_score in folder_result["dimension_averages"].items():
            # 按病例数加权
            for _ in range(folder_result["total_cases"]):
                all_dimension_scores[dim].append(avg_score)
    
    # 计算每个维度的总体平均分
    grand_dimension_avg = {}
    for dim, scores in all_dimension_scores.items():
        if len(scores) > 0:
            grand_dimension_avg[dim] = round(sum(scores) / len(scores), 2)
    
    # 计算总体平均分
    grand_overall_avg = 0
    if grand_dimension_avg:
        grand_overall_avg = round(sum(grand_dimension_avg.values()) / len(grand_dimension_avg), 2)
    
    return {
        "total_cases": total_cases,
        "total_folders": len(results),
        "dimension_averages": grand_dimension_avg,
        "overall_average": grand_overall_avg
    }

def print_detailed_results(results, grand_average):
    """打印详细结果"""
    print("=" * 80)
    print("各文件夹详细结果:")
    print("=" * 80)
    
    for folder_name in sorted(results.keys()):
        folder_result = results[folder_name]
        print(f"\n📂 {folder_name}:")
        print(f"   病例数: {folder_result['total_cases']}")
        print(f"   总体平均分: {folder_result['overall_average']}")
        print(f"   各维度平均分:")
        
        for dim in sorted(folder_result["dimension_averages"].keys()):
            avg = folder_result["dimension_averages"][dim]
            print(f"      {dim:40s}: {avg:.2f}")
    
    print("\n" + "=" * 80)
    print("总体汇总:")
    print("=" * 80)
    print(f"总文件夹数: {grand_average['total_folders']}")
    print(f"总病例数: {grand_average['total_cases']}")
    print(f"\n各维度总体平均分:")
    
    for dim in sorted(grand_average["dimension_averages"].keys()):
        avg = grand_average["dimension_averages"][dim]
        print(f"   {dim:40s}: {avg:.2f}")
    
    print(f"\n{'总体平均分':40s}: {grand_average['overall_average']:.2f}")
    print("=" * 80)

def save_results(results, grand_average, output_file):
    """保存结果到JSON文件"""
    output_data = {
        "folder_results": results,
        "grand_average": grand_average
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="计算所有子文件夹的平均分并汇总")
    parser.add_argument("--base-path", "-b", type=str,
                       default="../../data/ai_review",
                       help="基础文件夹路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="输出JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    print(f"🚀 开始处理: {args.base_path}\n")
    
    # 计算所有子文件夹的平均分
    results = calculate_all_folders(args.base_path)
    
    if not results:
        print("❌ 没有找到有效数据")
        return
    
    # 计算总体平均分
    grand_average = calculate_grand_average(results)
    
    # 打印详细结果
    print_detailed_results(results, grand_average)
    
    # 保存结果（如果指定了输出文件）
    if args.output:
        save_results(results, grand_average, args.output)

def generate_boxplot(base_path, output_dir):
    """
    为每个模型生成单独的箱线图
    """
    # 固定维度顺序 + 简写标签
    dimension_order = [
        "questioning_skills_review",
        "self_awareness_review",
        "robustness_review",
        "ethics_review",
        "explainability_review",
        "information_summary_review",
        "diagnostic_reasoning_review",
        "medication_safety_review",
    ]

    dimension_labels = [
        "QS", "IC", "HR", "EX", "ET", "II", "DR", "MS"
    ]

    # 收集所有模型的数据
    models = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # 为每个模型生成箱线图
    for model in models:
        model_path = os.path.join(base_path, model)
        
        # 使用 load_case_level_scores 函数加载数据
        model_df = load_case_level_scores(model_path)
        
        if model_df.empty:
            print(f"❌ 没有找到 {model} 的有效数据用于绘制箱线图")
            continue
        
        # 计算病例级平均分，与 avg_all_folders.py 逻辑一致
        case_level = (
            model_df
            .groupby(["case_id", "dimension"])["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "case_mean_score"})
        )
        
        # 设置维度顺序
        case_level["dimension"] = pd.Categorical(
            case_level["dimension"],
            categories=dimension_order,
            ordered=True
        )

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成箱线图，使用 case_mean_score 列，与 avg_all_folders.py 生成的文件格式一致
        plt.figure(figsize=(9, 3.8))

        # 定义 Nature 风格配色变量
        fill_color = "#E8F0F5"      
        main_color = "#5A7D9A"      
        median_color = "#081D33"    
        dot_color = "#5A7D9A"       

        # 绘制箱线图
        sns.boxplot(
            data=case_level,
            x="dimension",
            y="case_mean_score",
            color=fill_color,           # 应用更明亮的填充
            width=0.6,
            linewidth=1.0,              # 线条稍微减细，显得更精致
            fliersize=0,
            medianprops=dict(color=median_color, linewidth=2.2), # 粗壮深色的中位数线
            whiskerprops=dict(color=main_color, linewidth=1.0),
            capprops=dict(color=main_color, linewidth=1.0),
            boxprops=dict(edgecolor=main_color, linewidth=1.1)
        )

        # 叠加病例散点（利用透明度制造密度感）
        sns.stripplot(
            data=case_level,
            x="dimension",
            y="case_mean_score",
            color=main_color,           # 散点与箱体边框同色
            size=4.0,                   # 稍微调整大小
            jitter=0.2,                 # 适中的抖动范围
            alpha=0.4,                  # 关键：设置 40% 透明度，重叠处颜色自然变深
            edgecolor="none"            # 去掉点边框，让点看起来更柔和
        )

        # 坐标轴与排版
        plt.ylim(3.5, 5.05)

        plt.ylabel("Human satisfaction score", fontsize=11)
        plt.xlabel("", fontsize=11)

        plt.xticks(
            ticks=range(len(dimension_labels)),
            labels=dimension_labels,
            fontsize=10
        )

        plt.yticks(fontsize=10)

        sns.despine(trim=True)

        plt.tight_layout()

        # 保存箱线图
        output_path = os.path.join(output_dir, f"human_evaluation_boxplot_{model}.png")
        plt.savefig(
            output_path,
            format='png',
            bbox_inches="tight",
            dpi=300
        )
        plt.close()

        print(f"✅ 箱线图已保存到: {output_path}")

        # 计算该模型的统计数据
        print(f"\n📊 --- {model} 的统计数据（用于 Results 写作） ---")
        model_stats = case_level.groupby("dimension")["case_mean_score"].agg(
            Median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Min='min',
            Max='max',
            Count='count'
        ).reindex(dimension_order)

        # 计算 IQR
        model_stats['IQR'] = model_stats['Q3'] - model_stats['Q1']

        # 将索引替换为简写标签
        model_stats.index = dimension_labels

        # 打印统计数据
        print(model_stats.round(3).to_string())

    return

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="计算所有子文件夹的平均分并汇总")
    parser.add_argument("--base-path", "-b", type=str,
                       default="../../data/ai_review",
                       help="基础文件夹路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="输出JSON文件路径（可选）")
    parser.add_argument("--output-dir", "-d", type=str, default="./outputs/human_evaluation",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 开始处理: {args.base_path}\n")
    
    # 计算所有子文件夹的平均分
    results = calculate_all_folders(args.base_path)
    
    if not results:
        print("❌ 没有找到有效数据")
        return
    
    # 计算总体平均分
    grand_average = calculate_grand_average(results)
    
    # 打印详细结果
    print_detailed_results(results, grand_average)
    
    # 保存结果（如果指定了输出文件）
    if args.output:
        # 如果输出文件是相对路径，则保存到输出目录中
        if not os.path.isabs(args.output):
            args.output = os.path.join(args.output_dir, args.output)
        save_results(results, grand_average, args.output)
    
    # 生成箱线图
    print("\n📊 生成箱线图...")
    generate_boxplot(args.base_path, args.output_dir)

if __name__ == "__main__":
    main()