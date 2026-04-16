import json
import re
import logging
import torch
from args import get_args
import importlib
import random
from Neo4j_functions import Neo4jDatabase
from AIPatient import AIPatient

def setup_logger(name, file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def patientize_query(query):
    """
    Replaces patient ID patterns with "ME"
    """
    patterns = [
        r"\bTHE PATIENT HAVING AN ADMISSION ID \d+\b",
        r"\bTHE PATIENT WITH ADMISSION ID \d+\b"
    ]
    for pattern in patterns:
        query = re.sub(pattern, "ME", query, flags=re.IGNORECASE)
    return query.strip()

def load_neo4j_credentials(secret_file):
    """
    Load Neo4j credentials from secret file
    """
    credentials = {}
    with open(secret_file) as f:
        for line in f:
            key, value = line.strip().split(',', 1)
            credentials[key.strip()] = value.strip()
    return credentials['uri'], credentials['user'], credentials['password']

def run_patient_interaction(expert_system, initial_info, answers_list, db, patient_admission):
    """
    Run patient interaction loop
    """
    personality = random.choice(["Responsible", "Organized", "Analytical", "Terse"])
    patient = AIPatient(args, initial_info, db, patient_admission, personality)
    temp_answers = []
    temp_info = []

    while len(patient.get_questions()) < args.max_questions:
        state = patient.get_state()
        response = expert_system.respond(state)
        resp_type = response.get("type", "unknown")
        resp_content = response.get("content", "[No content returned]")
        
        print(resp_content)
        if resp_type != 'answer':
            patient.update_interaction('doctor', resp_content)
        
        temp_info.append({k: v for k, v in response.items() if k not in ["type", "content"]})

        if resp_type == "question":
            patient_resp = patient.respond_aipatient(resp_content)
            patient.update_interaction('patient', patient_resp)
        elif resp_type == "answer":
            return resp_content, patient.get_questions(), patient.get_answers(), temp_answers, temp_info, patient.history
        else:
            print(response)
            temp_answers.append(resp_content)
            return resp_content, patient.get_questions(), patient.get_answers(), temp_answers, temp_info, patient.history

    # Max questions reached
    stuck_response = 'I dont know'
    temp_info.append({k: v for k, v in response.items() if k != "answer"})
    return stuck_response, patient.get_questions(), patient.get_answers(), temp_answers + [stuck_response], temp_info, patient.history

def main():
    """
    Main function
    """
    # Load doctor module
    module = importlib.import_module(args.doctor_module)
    doctor_class = getattr(module, args.doctor_class)
    expert_system = doctor_class(args)

    # Load database credentials
    secret_file = "../secrets.txt"
    uri, user, password = load_neo4j_credentials(secret_file)
    db = Neo4jDatabase(uri, user, password)

    # Setup logger
    general_logger = setup_logger('general_logger', args.log_filename)

    # Process patients
    correct = []
    records = db.get_all_patient_admissions()
    
    for i, patient_admission in enumerate(records[args.start_id:args.end_id], 1):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Get patient data
        initial_question, correct_answer = db.get_initial_question_answer(
            patient_admission["SubjectID"], patient_admission["AdmissionID"])
        answers_list = db.get_all_drug_answers_list(
            patient_admission["SubjectID"], patient_admission["AdmissionID"])
        initial_info = db.get_patient_demographics(patient_admission["SubjectID"])
        initial_question = patientize_query(initial_question)
        diagnoses = db.get_patient_diagnoses(
            patient_admission["SubjectID"], patient_admission["AdmissionID"])
        
        print(initial_question)
        print(initial_info)
        general_logger.info(f"\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nPATIENT #{i}")

        # Run interaction
        final_answer, questions, answers, temp_answers, temp_info, conversation = run_patient_interaction(
            expert_system, initial_info, answers_list, db, patient_admission)

        # Save output
        output = {
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
                "conversation_history": conversation,
            },
        }

        correct.append(final_answer == correct_answer)
        general_logger.info(f"||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nInteraction ended for patient #{i}")

        with open(args.output_filename, 'a+') as f:
            f.write(json.dumps(output) + '\n')

        general_logger.info(f'Processed {i}/{len(records[args.start_id:args.end_id])} patients | Accuracy: {sum(correct) / len(correct)}')
    
    print(f"Accuracy: {sum(correct) / len(correct)}")

if __name__ == "__main__":
    args = get_args()
    main()
