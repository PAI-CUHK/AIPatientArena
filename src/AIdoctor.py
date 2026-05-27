import random
import re
import time
import json
from llm_model_class import LLM_Models


class AIdoctor:
    def __init__(self, args, inquiry=None, answers_list=None):
        self.args = args
        self.llm_models = LLM_Models()

    def respond(self, patient_state):
        initial_info = patient_state['initial_info']
        full_context = patient_state['conversation_history']
        turn_number = patient_state['turn_number']



        system_prompt = """
        You will simulate an experienced professional clinician and conduct a structured, human-like medical consultation.
        The consultation is taking place in the outpatient clinic. 
        You and the patient are sitting face to face in the consultation room, engaging in a natural doctor-patient conversation. 
        Your task is to conduct the consultation in a structured, human-like way, as if you were the treating physician.  
        Your behavior must reflect the style of a real doctor, avoid letting patients know that you are AI, and communicate like a doctor in a real clinic. 
        Ask only one question at a time, do not ask multiple questions at a time or break down the questions. 
        The consultation must strictly end within 15 turns. If the 15th turn is reached, you must stop and give the final diagnosis and treatment plan, even if information is incomplete.      
        The answer should be concise, logical, and contain moderate information.
        """

        user_prompt = f"""
        The current situation is as follows:
        Patient Information:
        {json.dumps(initial_info, indent=2)}

        Conversation History:
        \"{full_context}\"

        You are currently at turn {turn_number} of a maximum of 15 turns.

        Your tasks:
        1. Read the conversation history carefully to determine whether more information is needed.
        2. If necessary, ask the next natural question to move the consultation forward.
        3. You must strictly end the consultation no later than the 15th turn.      
            - If sufficient information is available earlier, end before the 15th turn.  
            - If the 15th turn is reached, you must end immediately, even if information is incomplete.  
            Provide a clear and precise diagnosis, and issue a complete, justified **prescription list**.  
            The prescription must include specific medication names.    
            Each drug must be clearly linked to the patient's specific clinical issues—including the primary disease, complications, symptoms, and risk prevention.  
            Do not omit treatment for any relevant problem. Ensure the medication plan is comprehensive, safe, and clinically appropriate.              
        4. Record all clinical information about the patient mentioned in the conversation. For any missing information, explicitly state "Not available".

        Please respond using **only** one of the two JSON formats below. Do not include any additional text or markdown formatting.

        Next question format:
        {
        "type": "question",
        "content": "<Your next natural continuation or prompt in the interview>"
        }

        Final diagnosis and treatment plan answer format:
        {
            "type": "answer",
            "content": {{   
            "diagnosis": "<Final diagnosis>",
            "diagnosis_reasoning": "<Detailed step-by-step diagnostic reasoning, explaining how this diagnosis was reached>",
            "recommended_drugs": ["DRUG1", "DRUG2", ...],
            "drug_reasoning": {
                "DRUG1": "<Why this drug is appropriate for the patient>",
                "DRUG2": "<...>"
            }},
            "category_summaries": {{
                "symptoms": "<...>",
                "medical_history": "<...>",
                "family_history": "<...>",
                "vital_signs": "<...>",
                "allergies": "<...>"        
            }}
        }
        }
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.get_valid_response(messages)


    def get_valid_response(self, messages):
        """
        根据配置调用相应的模型
        """
        # 根据模型类型选择调用方式
        if hasattr(self.args, 'model_type') and self.args.model_type == 'vllm':
            # 调用 vLLM
            response_text = self.llm_models.run_vllm(
                messages=messages,
                vllm_port=self.args.vllm_port,
                vllm_model=self.args.vllm_model
            )
        else:
            # 调用 GPT
            response_text = self.llm_models.run_gpt(
                messages,
                model=self.args.doctor_model_name
            )

        # 尝试解析 JSON
        response_json = json.loads(response_text)

        return response_json