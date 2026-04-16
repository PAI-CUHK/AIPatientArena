# conversation_eval_agent.py
import os
import json
import argparse
import traceback
from eval_helper import *

# === 各种指标（并行版）===
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def compute_scores_parallel(dialogue, summaries, patient_info, diagnosis_info, drug_info, evaluator):
    results = {}

    # 各个任务函数
    tasks = {
        "questioning_skills_score": lambda: calculate_questioning_skills_score(questioning_skills_instruction(dialogue, evaluator.eval_model)),
        "information_coverage_score": lambda: calculate_information_coverage_score(information_coverage_instruction(dialogue, evaluator.eval_model)),
        "robustness_score": lambda: calculate_robustness_score(robustness_instruction(dialogue, evaluator.eval_model)),
        "ethics_score": lambda: calculate_ethics_score(ethics_instruction(dialogue, evaluator.eval_model)),
        "explainability_score": lambda: calculate_explainability_score(explainability_instruction(dialogue, evaluator.eval_model)),
        "information_summary_score": lambda: calculate_summary_accuracy_score(information_summary_instruction(dialogue, summaries, evaluator.eval_model)),
        "diagnostic_reasoning_score": lambda: calculate_diagnostic_reasoning_score(evaluate_diagnosis_case(patient_info, diagnosis_info, evaluator.eval_model)),
        "medication_safety_score": lambda: calculate_medication_safety_score(medication_safety_instruction(dialogue, drug_info, evaluator.eval_model)),
    }

    # 使用线程池
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {executor.submit(func): key for key, func in tasks.items()}

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            while True:
                try:
                    # 尝试获取结果
                    results[key] = future.result()
                    break  # 成功，跳出循环
                except Exception as e:
                    print(f"⚠️ {key} 出错，正在重试: {e}")
                    time.sleep(1)
                    # 重新提交该任务
                    future = executor.submit(tasks[key])

    return results


class DialogueEvaluator:
    def __init__(self, model_name="gpt5"):
        self.model_name = model_name
        self.eval_model = LLM_Models()


def main(input_file, output_file, start_index=0, end_index=442, max_retries=3, retry_delay=2):
    evaluator = DialogueEvaluator()

    with open(input_file, 'r', encoding='utf-8') as file, open(output_file, 'a', encoding='utf-8') as out_file:
        count = 0
        for idx, line in enumerate(file):
            if idx < start_index:
                continue
            if idx >= end_index:
                break

            record = json.loads(line)
            patient_id = record.get('id', {})
            subject_id = patient_id.get('SubjectID', 'Unknown')
            admission_id = patient_id.get('AdmissionID', 'Unknown')
            print(f"\n▶ Processing SubjectID={subject_id}, AdmissionID={admission_id}")

            # ===== 解析 answer =====
            answer_raw = record.get('interactive_system', {}).get('final_answer', {})
            if isinstance(answer_raw, str):
                try:
                    answer = json.loads(answer_raw)
                except json.JSONDecodeError as e:
                    error_entry = {
                        "SampleIndex": idx,
                        "SubjectID": subject_id,
                        "AdmissionID": admission_id,
                        "status": "error",
                        "error_type": "JSONDecodeError",
                        "error_msg": str(e),
                        "answer_raw": answer_raw
                    }
                    out_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    continue
            elif isinstance(answer_raw, dict):
                answer = answer_raw
            else:
                error_entry = {
                    "SampleIndex": idx,
                    "SubjectID": subject_id,
                    "AdmissionID": admission_id,
                    "status": "error",
                    "error_type": "InvalidType",
                    "error_msg": f"final_answer is {type(answer_raw)}",
                    "answer_raw": str(answer_raw)
                }
                out_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                continue

            success = False
            metric_order = [
                "questioning_skills_score",
                "information_coverage_score",
                "robustness_score",
                "ethics_score",
                "explainability_score",
                "information_summary_score",
                "diagnostic_reasoning_score",
                "medication_safety_score",
            ]
            while True:
                try:
                    dialogue = record.get('interactive_system', {}).get('conversation_history', [])
                    patient_info = {
                        "dialogue": dialogue,
                        "info": record.get("info", {}).get("initial_info", [])
                    }
                    summaries = answer.get('category_summaries', {})
                    diagnosis_info = {
                        "clinician_provided_diagnosis": answer.get("diagnosis", ""),
                        "diagnosis_reasoning": answer.get("diagnosis_reasoning", {}),
                        "correct_diagnoses": record.get("info", {}).get("correct_diagnoses", [])
                    }
                    drug_info = {
                        "clinician_prescribed_medications": answer.get('recommended_drugs', []),
                        "drug_reasoning": answer.get('drug_reasoning', {}),
                        "correct_diagnoses": record.get("info", {}).get("correct_diagnoses", []),
                        "correct_answer": record.get('info', {}).get('correct_answer', []),
                        "diagnosis_reasoning": answer.get("diagnosis_reasoning", {}),
                    }

                    # === 各种指标 ===
                    metric_scores = compute_scores_parallel(
                        dialogue, summaries, patient_info, diagnosis_info, drug_info, evaluator
                    )

                    # ✅ 先 SubjectID / AdmissionID，再按顺序加入指标
                    scores = {
                        "SubjectID": subject_id,
                        "AdmissionID": admission_id,
                    }
                    for key in metric_order:
                        scores[key] = metric_scores.get(key)
                    # ✅ 成功才写
                    out_file.write(json.dumps(scores, ensure_ascii=False) + "\n")
                    success = True
                    break  # 跳出 retry 循环


                except Exception as e:

                    print(f"⚠️ 出错，重新尝试 SubjectID={subject_id}, AdmissionID={admission_id}: {e}")

                    time.sleep(2)  # 等一会再试

            count += 1

        print(f"\n✅ Finished evaluating {count} records.")
    print(f"📄 Output saved to: {os.path.abspath(output_file)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dialogues and output JSONL results")
    parser.add_argument("--input", "-i", type=str, default='qwen3.jsonl', help="Input JSONL file path")
    parser.add_argument("--output", "-o",type=str, default='qwen3.jsonl', help="Output JSONL file path")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start index for processing (default=0)")
    parser.add_argument("--end", "-e", type=int, default=5, help="end index for processing (default=100)")
    args = parser.parse_args()
    input_file = '../data/dialogue/' + args.input
    output_file = '../data/eval/' + args.output
    main(input_file, output_file, args.start, args.end)
