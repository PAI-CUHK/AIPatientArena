import json
import os
import re
import time
import logging
import argparse
import importlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from args import get_args
from AIPatient import AIPatient
from Neo4j_functions import Neo4jDatabase
from AIdoctor import AIdoctor
from eval_helper import (
    Medical_interview_questioning_skills,
    self_awareness_counts_instruction,
    robustness_counts_instruction,
    ethics_counts_instruction,
    explainability_counts_instruction,
    information_summary_instruction,
    evaluate_diagnosis_case,
    medication_safety_counts_instruction,
    extract_reasoning_path_with_linkage,
    calculate_questioning_skills_score,
    calculate_self_awareness_score,
    calculate_robustness_score,
    calculate_ethics_score,
    calculate_explainability_score,
    calculate_summary_accuracy_score,
    calculate_diagnostic_reasoning_score,
    calculate_medication_safety_score,
    LLM_Models,
)


def setup_logger(name, file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]
    data = {item['id']: item for item in data}
    return data


def patientize_query(query):
    patterns = [
        r"\bTHE PATIENT HAVING AN ADMISSION ID \d+\b",
        r"\bTHE PATIENT WITH ADMISSION ID \d+\b"
    ]
    for pattern in patterns:
        query = re.sub(pattern, "ME", query, flags=re.IGNORECASE)
    return query.strip()


def compute_scores_parallel(diagnosis_path, dialogue, summaries, patient_info, diagnosis_info, drug_info, evaluator):
    results = {}

    tasks = {
        "questioning_skills_score": lambda: calculate_questioning_skills_score(Medical_interview_questioning_skills(diagnosis_path, evaluator.eval_model)),
        "self_awareness_score": lambda: calculate_self_awareness_score(self_awareness_counts_instruction(diagnosis_path, evaluator.eval_model)),
        "robustness_score": lambda: calculate_robustness_score(robustness_counts_instruction(diagnosis_path, evaluator.eval_model)),
        "ethics_score": lambda: calculate_ethics_score(ethics_counts_instruction(dialogue, evaluator.eval_model)),
        "explainability_score": lambda: calculate_explainability_score(explainability_counts_instruction(diagnosis_path, evaluator.eval_model)),
        "information_summary_score": lambda: calculate_summary_accuracy_score(information_summary_instruction(diagnosis_path, summaries, evaluator.eval_model)),
        "diagnostic_reasoning_score": lambda: calculate_diagnostic_reasoning_score(evaluate_diagnosis_case(patient_info, diagnosis_info, evaluator.eval_model)),
        "medication_safety_score": lambda: calculate_medication_safety_score(medication_safety_counts_instruction(dialogue, drug_info, evaluator.eval_model)),
    }

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {executor.submit(func): key for key, func in tasks.items()}

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            while True:
                try:
                    results[key] = future.result()
                    break
                except Exception as e:
                    print(f"⚠️ {key} 出错，正在重试: {e}")
                    time.sleep(1)
                    future = executor.submit(tasks[key])

    return results


class DialogueEvaluator:
    def __init__(self, model_name="gpt-oss:120b"):
        self.model_name = model_name
        self.eval_model = LLM_Models()


def evaluate_single_record(record, evaluator, eval_output_file, error_output_file):
    subject_id = record.get('id', {}).get('SubjectID', 'Unknown')
    admission_id = record.get('id', {}).get('AdmissionID', 'Unknown')
    print(f"\n▶ Evaluating SubjectID={subject_id}, AdmissionID={admission_id}")

    answer_raw = record.get('interactive_system', {}).get('final_answer', {})
    if isinstance(answer_raw, str):
        try:
            answer = json.loads(answer_raw)
        except json.JSONDecodeError as e:
            error_entry = {
                "SubjectID": subject_id,
                "AdmissionID": admission_id,
                "status": "error",
                "error_type": "JSONDecodeError",
                "error_msg": str(e),
                "answer_raw": answer_raw
            }
            error_output_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
            error_output_file.flush()
            return False, "JSON解析失败"
    elif isinstance(answer_raw, dict):
        answer = answer_raw
    else:
        error_entry = {
            "SubjectID": subject_id,
            "AdmissionID": admission_id,
            "status": "error",
            "error_type": "InvalidType",
            "error_msg": f"final_answer is {type(answer_raw)}",
            "answer_raw": str(answer_raw)
        }
        error_output_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        error_output_file.flush()
        return False, f"final_answer类型错误: {type(answer_raw)}"

    required_fields = ['category_summaries', 'diagnosis', 'diagnosis_reasoning', 'recommended_drugs', 'drug_reasoning']
    for field in required_fields:
        if field not in answer:
            error_entry = {
                "SubjectID": subject_id,
                "AdmissionID": admission_id,
                "status": "error",
                "error_type": "MissingField",
                "error_msg": f"缺少必需字段: {field}",
                "answer_keys": list(answer.keys())
            }
            error_output_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
            error_output_file.flush()
            return False, f"缺少必需字段: {field}"

    try:
        dialogue = record.get('interactive_system', {}).get('conversation_history', [])
        diagnosis_path = extract_reasoning_path_with_linkage(dialogue, evaluator.eval_model)
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

        metric_order = [
            "questioning_skills_score",
            "self_awareness_score",
            "robustness_score",
            "ethics_score",
            "explainability_score",
            "information_summary_score",
            "diagnostic_reasoning_score",
            "medication_safety_score",
        ]

        metric_scores = compute_scores_parallel(
            diagnosis_path, dialogue, summaries, patient_info, diagnosis_info, drug_info, evaluator
        )

        scores = {
            "SubjectID": subject_id,
            "AdmissionID": admission_id,
        }
        for key in metric_order:
            scores[key] = metric_scores.get(key)

        eval_output_file.write(json.dumps(scores, ensure_ascii=False) + "\n")
        eval_output_file.flush()
        return True, "success"

    except Exception as e:
        error_entry = {
            "SubjectID": subject_id,
            "AdmissionID": admission_id,
            "status": "error",
            "error_type": "EvaluationError",
            "error_msg": str(e)
        }
        error_output_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        error_output_file.flush()
        return False, f"评估错误: {str(e)}"


def run_patient_interaction(doctor_system, initial_question, db, patient_admission, secret_file, model_type, max_questions):
    personality_profile = random.choice(["Responsible", "Organized", "Analytical", "Terse"])

    patient_system = AIPatient(args,
                               initial_question, db,
                               patient_admission,
                               personality_profile,
                               secret_file=secret_file,
                               model_type=model_type)
    temp_answer_list = []
    temp_additional_info = []

    while len(patient_system.get_questions()) < max_questions:
        patient_state = patient_system.get_state()
        response_dict = doctor_system.respond(patient_state)

        response_type = response_dict.get("type", "unknown")
        response_content = response_dict.get("content", "[No content returned]")
        print(response_content)
        if response_type != 'answer':
            patient_system.update_interaction('doctor', response_content)
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "content"]})

        if response_type == "question":
            patient_response = patient_system.respond_aipatient(response_content)
            patient_system.update_interaction('patient', patient_response)
        elif response_type == "test":
            temp_answer_list.append(response_content)
        elif response_type == "answer":
            return response_content, patient_system.get_questions(), patient_system.get_answers(), temp_answer_list, temp_additional_info, patient_system.history

        else:
            print(response_dict)
            temp_answer_list.append(response_content)
            return response_content, patient_system.get_questions(), patient_system.get_answers(), temp_answer_list, temp_additional_info, patient_system.history

    stuck_response = 'I dont know'
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "answer"})

    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_answer_list + [
        stuck_response], temp_additional_info, patient_system.history


def validate_eval_requirements(final_answer, conversation_history):
    """验证对话生成结果是否满足评估所需的必要元素"""
    required_fields = ['category_summaries', 'diagnosis', 'diagnosis_reasoning', 'recommended_drugs', 'drug_reasoning']
    
    if not conversation_history or len(conversation_history) == 0:
        return False, "缺少对话历史 (conversation_history)"
    
    if not final_answer:
        return False, "缺少最终答案 (final_answer)"
    
    answer = None
    if isinstance(final_answer, str):
        try:
            answer = json.loads(final_answer)
        except json.JSONDecodeError as e:
            return False, f"final_answer 不是有效的 JSON: {str(e)}"
    elif isinstance(final_answer, dict):
        answer = final_answer
    else:
        return False, f"final_answer 类型错误: {type(final_answer)}"
    
    if not isinstance(answer, dict):
        return False, "final_answer 解析后不是字典类型"
    
    missing_fields = []
    for field in required_fields:
        if field not in answer:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"缺少必需字段: {', '.join(missing_fields)}"
    
    return True, "验证通过"


def load_processed_records(output_filename):
    """从已存在的输出文件中读取已处理的patient ID"""
    processed_ids = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            patient_id = record.get('id', {})
                            if isinstance(patient_id, dict):
                                key = (patient_id.get('SubjectID'), patient_id.get('AdmissionID'))
                                processed_ids.add(key)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ 读取已处理记录时出错: {e}")
    return processed_ids


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    module = importlib.import_module(args.doctor_module)
    doctor_class = getattr(module, args.doctor_class)

    # 从 secrets.txt 读取 Neo4j 配置
    secret_file = os.path.join(os.path.dirname(__file__), "..", "secrets.txt")
    uri, user, password, database = None, None, None, 'neo4j'  # 默认数据库为 neo4j
    
    with open(secret_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(',')
            key = parts[0].strip()
            if key == "uri":
                uri = parts[1].strip()
            elif key == "user":
                user = parts[1].strip()
            elif key == "password":
                password = parts[1].strip()
            elif key == "database":
                database = parts[1].strip()
    
    db = Neo4jDatabase(uri, user, password, database)

    history_logger = setup_logger('history_logger', args.history_log_filename)
    general_logger = setup_logger('general_logger', args.log_filename)

    num_processed = 0
    correct = []
    all_num = 0
    records = db.get_all_patient_admissions()
    doctor_system = doctor_class(args)
    model_type = "claude"

    evaluator = DialogueEvaluator()

    dialogue_output_file = open(args.output_filename, 'a+')
    eval_output_file = open(args.eval_output_filename, 'a+')
    error_output_file = open(args.error_filename, 'a+')

    processed_ids = load_processed_records(args.output_filename)
    if processed_ids:
        print(f"✅ 发现已处理 {len(processed_ids)} 条记录，将跳过这些记录")

    try:
        for patient_admission in records[args.start_id:args.end_id]:
            subject_id = patient_admission["SubjectID"]
            admission_id = patient_admission["AdmissionID"]
            
            if args.resume and (subject_id, admission_id) in processed_ids:
                print(f"⏭️ 跳过已处理的记录: SubjectID={subject_id}, AdmissionID={admission_id}")
                continue

            print(patient_admission)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            all_num += 1
            initial_question = 'What is the patient\'s medication list?'
            correct_answer = db.get_initial_question_answer(subject_id, admission_id)
            # answers_list = db.get_all_drug_answers_list(subject_id, admission_id)
            initial_info = db.get_patient_demographics(subject_id)
            initial_question = patientize_query(initial_question)
            diagnoses = db.get_patient_diagnoses(subject_id, admission_id)
            print(initial_question)
            print(initial_info)
            history_logger.info(
                f"\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nPATIENT #{all_num}")

            max_retries = 3
            retry_count = 0
            output_dict = None
            conversation_success = False

            while retry_count < max_retries and not conversation_success:
                try:
                    final_answer, questions, answers, temp_answer_list, temp_additional_info, conversation_history = run_patient_interaction(
                        doctor_system, initial_info, db, patient_admission, secret_file, model_type, args.max_questions)

                    output_dict = {
                        "id": patient_admission,
                        "info": {
                            "initial_info": initial_info,
                            "correct_answer": correct_answer,
                            "correct_diagnoses": diagnoses,
                            "question": initial_question,
                        },
                        "interactive_system": {
                            "final_answer": final_answer,
                            "num_questions": len(questions),
                            "conversation_history": conversation_history,
                        },
                    }

                    valid, msg = validate_eval_requirements(final_answer, conversation_history)
                    if valid:
                        conversation_success = True
                        print(f"✅ 对话生成验证通过")
                    else:
                        retry_count += 1
                        print(f"⚠️ 对话生成验证失败 (尝试 {retry_count}/{max_retries}): {msg}")
                        if retry_count < max_retries:
                            print(f"🔄 等待 5 秒后重新生成...")
                            time.sleep(5)

                except Exception as e:
                    retry_count += 1
                    print(f"⚠️ 对话生成失败 (尝试 {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        print(f"🔄 等待 5 秒后重试...")
                        time.sleep(5)

            if not conversation_success:
                error_entry = {
                    "SubjectID": subject_id,
                    "AdmissionID": admission_id,
                    "status": "error",
                    "error_type": "ConversationGenerationFailed",
                    "error_msg": f"对话生成失败，已重试 {max_retries} 次"
                }
                error_output_file.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                error_output_file.flush()
                
                history_logger.info(f"❌ Patient #{all_num} 对话生成失败，跳过")
                print(f"❌ Patient #{all_num} 对话生成失败，跳过")
                continue

            correct.append(final_answer == correct_answer)
            num_processed += 1
            history_logger.info(
                f"||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nInteraction ended for patient #{all_num}")

            dialogue_output_file.write(json.dumps(output_dict) + '\n')
            dialogue_output_file.flush()

            general_logger.info(
                f'Processed {num_processed}/{all_num} patients | Accuracy: {sum(correct) / len(correct)}')

            print(f"\n=== Evaluating dialogue for patient #{all_num} ===")
            eval_success, eval_msg = evaluate_single_record(output_dict, evaluator, eval_output_file, error_output_file)
            
            if eval_success:
                print(f"✅ Evaluation completed for patient #{all_num}")
            else:
                print(f"❌ Evaluation failed for patient #{all_num}: {eval_msg}")

        print(f"\nFinal Accuracy: {sum(correct) / len(correct)}")
        print(f"📄 Dialogue output saved to: {os.path.abspath(args.output_filename)}")
        print(f"📄 Evaluation output saved to: {os.path.abspath(args.eval_output_filename)}")
        print(f"📄 Error log saved to: {os.path.abspath(args.error_filename)}")

    finally:
        dialogue_output_file.close()
        eval_output_file.close()
        error_output_file.close()


if __name__ == "__main__":
    args = get_args()
    main()