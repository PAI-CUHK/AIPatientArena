import json
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 8 个评分维度
SCORE_DIMS = [
    "diagnostic_reasoning_score",
    "medication_safety_score",
    "questioning_skills_score",
    "robustness_score",
    "self_awareness_score",
    "information_summary_score",
    "explainability_score",
    "ethics_score"
]

# 维度缩写
DIM_ABBR = {
    "questioning_skills_score": "QS",
    "self_awareness_score": "IC",
    "robustness_score": "HR",
    "ethics_score": "ET",  
    "explainability_score": "EX",
    "information_summary_score": "II",
    "diagnostic_reasoning_score": "Dx",
    "medication_safety_score": "MS",
}

def load_one_model(jsonl_path, model_name):
    """读取单个模型的 JSONL，返回 rows"""
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ {model_name} 第 {i} 行 JSON 解析失败: {e}")
                continue

            row = {
                "model_name": model_name
            }

            for dim in SCORE_DIMS:
                row[dim] = case.get(dim, {}).get("score")

            rows.append(row)

    return rows

def calculate_correlation(input_csv, output_corr_csv):
    """计算维度间的相关性"""
    df = pd.read_csv(input_csv)
    
    # 计算 Spearman 相关性
    corr = df[SCORE_DIMS].corr(method="spearman")
    print("\n📊 维度间相关性矩阵：")
    print(corr)
    
    # 保存相关性矩阵
    corr.to_csv(output_corr_csv)
    print(f"\n💾 已保存相关性矩阵：{output_corr_csv}")
    
    return corr

def generate_heatmap(corr_matrix, output_png):
    """生成相关性热力图"""
    # 重命名索引和列名使用缩写
    df_corr = corr_matrix.rename(index=DIM_ABBR, columns=DIM_ABBR)
    
    # 构造 alpha 矩阵
    alpha = np.full(df_corr.shape, 0.35)
    np.fill_diagonal(alpha, 1.0)
    
    # 绘图
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(
        df_corr,
        cmap="coolwarm",
        center=0,
        vmin=-0.15,
        vmax=0.15,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        alpha=alpha
    )
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 保存为 PNG 格式
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n📈 已生成相关性热力图：{output_png}")

def main():
    parser = argparse.ArgumentParser(
        description="Dimension Correlation Analysis: Extract scores, calculate correlations, and generate heatmap"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/eval",
        help="Directory containing evaluation JSONL files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--heatmap_png",
        type=str,
        default="dimension_spearman_corr_heatmap.png",
        help="Output PNG file name for heatmap"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 提取评分数据
    print("\n📥 提取模型评分数据...")
    all_rows = []

    # 自动检测所有模型文件
    import glob
    jsonl_files = glob.glob(f"{args.data_dir}/*.jsonl")
    
    if not jsonl_files:
        print(f"❌ 在 {args.data_dir} 目录中未找到模型文件")
        return
    
    print(f"📦 找到 {len(jsonl_files)} 个模型文件")
    
    for jsonl_path in jsonl_files:
        # 从文件名中提取模型名称
        model_name = os.path.basename(jsonl_path).replace("_all_0117.jsonl", "")
        
        print(f"📦 处理模型：{model_name}")
        rows = load_one_model(jsonl_path, model_name)
        print(f"   → 读取 {len(rows)} 条记录")

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print("\n📊 合并后的数据概览：")
    print(df.head())
    print(f"\n📦 总样本数：{len(df)}")
    
    # 2. 计算相关性（直接从DataFrame计算，不保存CSV）
    print("\n🔍 计算维度间相关性...")
    # 计算 Spearman 相关性
    corr_matrix = df[SCORE_DIMS].corr(method="spearman")
    print("\n📊 维度间相关性矩阵：")
    print(corr_matrix)
    
    # 3. 生成热力图
    print("\n📈 生成相关性热力图...")
    heatmap_png_path = os.path.join(args.out_dir, args.heatmap_png)
    generate_heatmap(corr_matrix, heatmap_png_path)
    
    print("\n✅ 维度关联性分析完成！")
    print(f"\n输出文件：")
    print(f"  - 相关性热力图：{heatmap_png_path}")

if __name__ == "__main__":
    main()
