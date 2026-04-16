from Neo4j_functions import Neo4jDatabase
import logging
import os
import re
from anthropic import Anthropic
from anthropic import AnthropicBedrock
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
import ollama
from llm_model_class import LLM_Models
import json
import random

class AIPatient:
    def __init__(self, args, initial_question, db, patient_admission, personality_profile, secret_file="../secrets.txt", model_type="claude"):
        # 初始化基础参数
        self.args = args
        self.model_name = args.patient_model_name
        self.db = db
        self.patient_admission = patient_admission
        self.personality_profile = personality_profile
        self.initial_info = initial_question
        
        # 初始化对话历史
        self.history = []  # 详细对话历史
        self.summary_history = ' '  # 摘要对话历史
        self.facts = None  # 原子事实存储
        
        # 初始化模型相关
        self.model_type = model_type.lower()
        self.openai_key = self._load_openai_key(secret_file)
        self.gpt = LLM_Models()
        
        # 初始化知识图谱相关查询
        self.node_properties_query = """
            CALL apoc.meta.data()
            YIELD label, other, elementType, type, property
            WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
            WITH label AS nodeLabels, collect(property) AS properties
            RETURN {labels: nodeLabels, properties: properties} AS output
            """

        self.rel_properties_query = """
            CALL apoc.meta.data()
            YIELD label, other, elementType, type, property
            WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
            WITH label AS nodeLabels, collect(property) AS properties
            RETURN {type: nodeLabels, properties: properties} AS output
            """

        self.rel_query = """
            CALL apoc.meta.data()
            YIELD label, other, elementType, type, property
            WHERE type = "RELATIONSHIP" AND elementType = "node"
            RETURN {source: label, relationship: property, target: other} AS output
            """
    
    def _load_openai_key(self, secret_file):
        """
        从密钥文件加载OpenAI API密钥
        """
        with open(secret_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.split(',')[0].strip() == "open_ai_key":
                    open_ai_key = line.split(',')[1].strip()
        return open_ai_key
    
    def run_model(self, text_prompt, max_tokens_to_sample=1024, temperature=1.0):
        """
        运行模型生成响应
        """
        return self.gpt.run_gpt(messages=[
            {"role": "user", "content": text_prompt}
        ], model="gpt-5-mini-ca")
    
    def schema_text(self, node_props, rel_props, rels):
        """
        生成知识图谱 schema 文本
        """
        return f"""
            This is the schema representation of the Neo4j database.
            Node properties are the following:
            {node_props}

            Relationship properties are the following:
            {rel_props}

            Relationship point from source to target nodes
            {rels}

            Make sure to respect relationship types and directions
            """
    
    def generate_schema(self, exclude_nodes=("DrugQuestion", "Medication", "Diagnosis")):
        """
        生成知识图谱 schema
        """
        node_props = self.db.execute_cypher_query(self.node_properties_query)
        rel_props = self.db.execute_cypher_query(self.rel_properties_query)
        rels = self.db.execute_cypher_query(self.rel_query)
        
        # 过滤掉不需要的节点属性
        filtered_node_props = [
            node for node in node_props
            if node.get("output", {}).get("labels") not in exclude_nodes
        ]

        # 过滤掉不需要的关系（即连接被排除节点的）
        filtered_rels = [
            rel for rel in rels
            if rel.get("output", {}).get("source") not in exclude_nodes
               and not any(t in exclude_nodes for t in rel.get("output", {}).get("target", []))
        ]

        return self.schema_text(filtered_node_props, rel_props, filtered_rels)
    
    def relationship_extraction_prompt(self, conversation_history, text, patient_admission):
        """
        生成关系提取提示
        """
        subject_id = patient_admission['SubjectID']
        hadm_id = patient_admission['AdmissionID']
        schema = self.generate_schema()
        prompt = f"""

        Based on the doctor's query, first determine what the doctor is asking for. Then extract the appropriate relationship and nodes from the knowledge graph. 

        Strictly follow these rules:
        - You MUST only select nodes and relationships that are explicitly defined in the provided Knowledge Graph Schema.
        - DO NOT make up, infer, hallucinate, or invent any node or relationship not found in the schema.
        - Your task is only to return the relevant node and relationship names based on the query's intent, not to write a Cypher query.
        - The output MUST strictly follow the given format and contain no additional explanation.

        For admissions related queries, the query should focus on "HAS_ADMISSION" relationship and "Admission" node. 
        For patient information related queries, the query should focus on the "Patient" node. 
        If the doctor asked about a symptom (e.g. cough, fever, etc.), the query should check if the "symptom" node and the "HAS_SYMPTOM" or "HAS_NOSYMOTOM" relationship; 
        If the doctor asked about the duration, frequency, and intensity of a symptom, the query should first check if the symptom exist. If it exist, then check the "duration", "frequency" and "intensity" node respectively, and "HAS_DURATION", "HAS_FREQUENCY", "HAS_INTENSITY" relationship respectively. 
        If the doctor asked about medical history, the query should check "History" node and the HAS_MEDICAL_HISTORY relationship. 
        If the doctor asked about vitals (temperature, blood pressure etc), the query should check the "Vital" node and "HAS_VITAL" relationship. 
        If the doctor asked about social history (smoking, alcohol consumption etc), the query should check the "SocialHistory" node and "HAS_SOCIAL_HISTORY" relationship. 
        If the doctor aksed about family history, the query should first check the "HAS_FAMILY_MEMBER" relationship and "FamilyMember" node. Then, the query should check the "HAS_MEDICAL_HISTORY" relationship and "FamilyMedicalHistory" node associated with the "FamilyMember" node. 
 
        Output_format: Enclose your output in the following format. Do not give any explanations or reasoning, just provide the answer. For example:
        {{'Nodes': ['symptom', 'duration'], 'Relationships': ['HAS_SYMPTOM', 'HAS_DURATION']}}

        The natural language query is:
        {text}

        The previous conversation history is:
        {conversation_history}


        The Knowledge Graph Schema is:
        {schema}
        """

        return prompt
    
    def cypher_query_construction_prompt(self, conversation_history, text, patient_admission, nodes_edges, abstraction_context=None):
        """
        生成 Cypher 查询构建提示
        """
        subject_id = patient_admission['SubjectID']
        hadm_id = patient_admission['AdmissionID']
        schema = self.generate_schema()
        prompt = f"""
       Write a Cypher query to extract the requested information from the doctor's natural language query. Use SUBJECT_ID: {subject_id} and HADM_ID: {hadm_id}. Focus only on {nodes_edges}.

        If the query is vague, use the current conversation context to determine intent. Query must be case-insensitive and support fuzzy matching (e.g., temperature, blood pressure, seizure, etc.), including lifestyle terms (e.g., smoke, tobacco, alcohol).

        If returning multiple values, separate with commas. When using WITH, retain all variables needed in later clauses to avoid scope errors.

        Only output one single-line Cypher query string. Do NOT include comments, explanations, line breaks, quotation marks, square brackets, backticks, or punctuation.

        Prefer the simplest valid Cypher query. 
        Do not add extra OPTIONAL MATCH, CASE, or reduce logic unless explicitly required by the natural language query. 
        Handle missing values in the application layer, not in Cypher.

        The previous conversation history is:
        {conversation_history}

        The natural language query is:
        {text}"""

        if abstraction_context is not None:
            prompt += f"""
            The step back context is:
            {abstraction_context} 
        """

        prompt += f"""
        The Knowledge Graph Schema is:{schema}"""

        prompt += """
        
        Here are a few examples of Cypher queries, these are examples only, not to be combined or copied literally, you should replay SUBJECT_ID and HADM_ID based on input:

        1. All current symptoms:
        MATCH (p:Patient {SUBJECT_ID: $subject_id} -[:HAS_ADMISSION]->(:Admission)-[:HAS_SYMPTOM]->(s:Symptom) RETURN collect(s.name)

        2. Does the patient have seizures:
        MATCH (p:Patient {SUBJECT_ID: $subject_id} -[:HAS_ADMISSION]->(a:Admission {{HADM_ID: 182203}})-[:HAS_SYMPTOM]->(s:Symptom) WHERE s.name =~ '(?i).*seizure.*'
        WITH p, a, s OPTIONAL MATCH (a)-[:HAS_NOSYMPTOM]->(ns:Symptom) WHERE ns.name =~ '(?i).*seizure.*'
        RETURN CASE WHEN s IS NOT NULL THEN 'HAS seizure' WHEN ns IS NOT NULL THEN 'DOES NOT HAVE seizures' ELSE 'DONT KNOW' END

        3. Vital signs:
        MATCH (:Admission {{HADM_ID: 145012}})-[:HAS_VITAL]->(v:Vital) RETURN v.LABEL + ': ' + toString(v.VALUE)

        4. Family history:
        MATCH (p:Patient {SUBJECT_ID: $subject_id}) 
        MATCH (p)-[:HAS_FAMILY_MEMBER]->(fm:FamilyMember)
        OPTIONAL MATCH (fm)-[r:HAS_MEDICAL_HISTORY]->(fh:FamilyMedicalHistory)
        WHERE r.subject_id = $subject_id 
        RETURN fm.name AS FamilyMember, 
       collect(fh.name) AS MedicalHistory

        5. Medical history:
        MATCH (p:Patient {SUBJECT_ID: $subject_id})-[:HAS_MEDICAL_HISTORY]->(h:History) RETURN collect(h.name)

        6. Allergies:
        MATCH (p:Patient {{SUBJECT_ID: 9054}})-[:HAS_ADMISSION]->(:Admission {{HADM_ID: 101161}})-[:HAS_ALLERGY]->(a:Allergy) RETURN a.name
        7. Find the duration of symptoms:
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission {HADM_ID: 104271})    
        OPTIONAL MATCH (a)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (s)-[r_d:HAS_DURATION]->(d:Duration)
        WHERE r_d.hadm_id = a.HADM_ID
        RETURN 
            s.name AS symptom,
            CASE
                WHEN s IS NULL THEN 'NO SUCH SYMPTOM'
                WHEN d IS NULL THEN 'DONT KNOW'
                ELSE d.name
            END AS duration
        ORDER BY symptom

        """
        return prompt
    
    def clean_cypher_query(self, query):
        """
        清理 Cypher 查询
        """
        # Remove surrounding quotes
        query = query.strip('"')
        # Remove surrounding brackets
        query = query.strip('[]')
        # Remove newline characters
        query = query.replace('\\n', ' ')
        # Remove any leading or trailing whitespace characters
        query = query.strip()
        # Normalize whitespace within the query
        query = re.sub(r'\s+', ' ', query)
        return query
    
    def abstraction_generation_prompt(self, conversation_history, text):
        """
        生成抽象查询提示
        """
        prompt = f"""
        You are an AI and Medical EHR expert. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to use for cypher query generation. 
 
        If the question is vague, consider the conversation history and the current context. Do not give any explanations or reasoning, just provide the answer. 
        Here are a few examples: 
        input: Do you have fevers as a symptom? 
        output: What symptoms does the patient has? 
        input: Is your current temperature above 97 degrees? 
        output: What is the patient's temperature? 

        The current conversation history is:
        {conversation_history}
        The original query is:
        {text}
        """
        return prompt
    
    def query_result_rewrite(self, doctor_query, cypher_query, query_result):
        """
        重写查询结果为自然语言
        """
        prompt = f"""
        You are a doctor's assistant. Based on the cypher_query, please structure the retrieved query results into natural language. Include all subject, relationship and object. 
        For example: 
        doctor query: what symptoms do you have?
        cypher query: MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission {{HADM_ID: 182203}})
        MATCH (a)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE p.SUBJECT_ID = 23709
        RETURN s.name AS Symptom 

        retrieved result: ['black and bloody stools', 'lightheadedness', 'shortness of breath']

        output: The patient has symptoms of black and bloody stools, lightheadedness, shortness of breath. 

        The doctor's original query is:
        {doctor_query}
        The cypher query is:
        {cypher_query}
        The retrieved results are:
        {query_result}
        """

        return prompt
    
    def summarize_text_prompt(self, conversation_history, doctor_query, patient_response):
        """
        生成对话摘要提示
        """
        prompt = f"""
        You are the doctor's assistent responsible for summarizing the conversation between the doctor and the patient.
        Be very brief, include the all the conversation history, doctor and patient's query and response. The last sentence should be about the current context (e.g. vital, symptom, or history).
        Write in full sentences and do not fabricate symptoms or history.
        The previous conversation is as follows:
        {conversation_history}
        The doctor has asked about the following query:
        {doctor_query}
        The patient's response to the doctor's query:
        {patient_response}
        """
        return prompt
    
    def rewrite_response_prompt(self, conversation_history, doctor_query, query_result, patient_admission, personality):
        """
        重写响应为患者语言
        """
        subject_id = patient_admission['SubjectID']
        hadm_id = patient_admission['AdmissionID']
        prompt = f"""
        You are a virtual patient in an office visit.  You are now speaking directly to the doctor.
        Your personality is {personality}.
        Your conversation history with the doctor is as follows:
        {conversation_history}
        The doctor has asked about the following query, focusing on the current context (e.g. vital, symptom, or history):
        {doctor_query}
        The query result is:
        {query_result}
        Based on all above information, please write your response to the doctor following your personality traits. 
        **Important instructions:**
        - You MUST respond in the **first person**, as if you are the patient speaking directly to the doctor.
        - If the doctor's question is vague, assume it refers to the current topic or most recent symptom mentioned.
        - If the query_result is empty or contains no information, reply ONLY with: **"I don't know."**
        - DO NOT fabricate any symptom, medical history, or detail.
        - DO NOT add any information that is not supported by the query_result.
        - DO NOT include quotes or refer to the doctor in the third person.
        - Keep your reply aligned with the conversation flow and your personality.
        DO NOT BREAK CHARACTER. 
        """
        return prompt
    
    def checker_construction_prompt(self, doctor_query, query_result, conversation_history):
        """
        检查查询结果是否合适
        """
        prompt = f"""
               You are assisting a doctor by checking whether a database query result is a reasonably relevant response to the doctor's question.

                The doctor's query is:
                {doctor_query}
                The query result is:
                {query_result}
                
                Your task:
                - If the query result clearly contains information that is relevant to the doctor's question — even if it is incomplete or not perfectly phrased — return only 'Y'.
                - If the query result is clearly irrelevant or unhelpful, rewrite the doctor's question to better match the kind of data the database can return. Only output the rewritten question, and nothing else.

                Guidelines:
                - Focus on **semantic relevance**, not surface-level word matching.
                - Minor differences in phrasing, point-of-view (e.g. "you" vs. "the patient"), or wording are acceptable.
                - Do not rewrite the question if the result already answers it well.
                """

        return prompt
    
    def interactive_session(self, doctor_query, max_token=8192):
        """
        交互式会话处理
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("log.txt"),
                logging.StreamHandler()
            ]
        )

        if doctor_query.lower() == 'exit':
            logging.info("Session terminated by the user.")
            return "Session terminated by the user."

        #########################################################################################################################

        ## Step 2.1: Extract relevant nodes and edges
        logging.info("Extract relevant nodes and edges based on query.")
        nodes_edges_query_cypher_prompt = self.relationship_extraction_prompt(self.summary_history, doctor_query,
                                                                              self.patient_admission)

        if len(nodes_edges_query_cypher_prompt) > max_token:
            nodes_edges_query_cypher_prompt = nodes_edges_query_cypher_prompt[:max_token]
        nodes_edges_results = self.run_model(nodes_edges_query_cypher_prompt)
        logging.info(f"Nodes and edges extracted: {nodes_edges_results}")
        
        ## Step 1: Construct Abstraction Query Prompt
        logging.info("Step 1: Constructing Abstraction Cypher query prompt based on the doctor's query.")
        abstraction_query_prompt = self.abstraction_generation_prompt(self.summary_history, doctor_query)
        if len(abstraction_query_prompt) > max_token:
            abstraction_query_prompt = abstraction_query_prompt[:max_token]
        abstraction_query_nl = self.run_model(abstraction_query_prompt)
        logging.info(f"Abstraction query in natural language generated: {abstraction_query_nl}")

        ## Step 3: Generate Abstraction Cypher Query
        logging.info("Constructing Cypher query prompt based on the abstraction query.")
        abstraction_query_cypher_prompt = self.cypher_query_construction_prompt(self.summary_history,
                                                                                abstraction_query_nl, self.patient_admission,
                                                                                nodes_edges_results)
        if len(abstraction_query_cypher_prompt) > max_token:
            abstraction_query_cypher_prompt = abstraction_query_cypher_prompt[:max_token]
        abstraction_query_cypher = self.run_model(abstraction_query_cypher_prompt)
        logging.info(f"Abstraction cypher generated: {abstraction_query_cypher}")

        ## Step 3.5: Clean Cypher Query
        abstraction_query_cypher = self.clean_cypher_query(abstraction_query_cypher)

        ## Step 4: Execute the generated Cypher query
        logging.info("Step 4: Executing the generated Cypher query.")
        abstraction_result = self.db.execute_cypher_query(abstraction_query_cypher, llm_model=self.run_model)
        abstract_result = []
        if abstraction_result:
            ## Rewrite to natural language
            abstraction_result_rewrite_prompt = self.query_result_rewrite(abstraction_query_nl,
                                                                          abstraction_query_cypher, abstraction_result)
            abstract_result = self.run_model(abstraction_result_rewrite_prompt)

        logging.info(f"Abstraction Query result: {abstract_result}")

        #########################################################################################################################

        ## Step One: Original doctor's query
        logging.info(f"Step Zero: The doctors has asked about: {doctor_query}")
        logging.info("Step One: Constructing Cypher query prompt based on the doctor's query.")
        cypher_query_prompt = self.cypher_query_construction_prompt(self.summary_history, doctor_query,
                                                                    self.patient_admission, nodes_edges_results,
                                                                    abstraction_context=abstract_result)

        ## Step 2.2: Construct Cypher Query
        if len(cypher_query_prompt) > max_token:
            cypher_query_prompt = cypher_query_prompt[:max_token]
            logging.info(f"Cypher query prompt truncated to {max_token} characters.")
        cypher_query = self.run_model(cypher_query_prompt)
        logging.info(f"Cypher query generated: {cypher_query}")

        ## Step 2.3: Clean Cypher Query
        cypher_query = self.clean_cypher_query(cypher_query)

        ## Step Three: Execute the generated Cypher query
        logging.info("Step Three: Executing the generated Cypher query.")
        query_result = self.db.execute_cypher_query(cypher_query, llm_model=self.run_model)
        if query_result:
            ## Rewrite to natural language
            query_result_rewrite_prompt = self.query_result_rewrite(doctor_query, cypher_query, query_result)
            query_result = self.run_model(query_result_rewrite_prompt)
        logging.info(f"Query result: {query_result}")

        ## Step Four: Evaluate if the query properly answered the question
        for attempt in range(2):
            logging.info(f"Attempt {attempt + 1}: Evaluating the query result.")
            checker_prompt = self.checker_construction_prompt(doctor_query, query_result, self.summary_history)
            if len(checker_prompt) > max_token:
                checker_prompt = checker_prompt[:max_token]
                logging.info(f"Checker prompt truncated to {max_token} characters.")
            checked_result = self.run_model(checker_prompt)
            logging.info(f"Checked result: {checked_result}")

            ## If the answer is deemed appropriate, stop the loop
            if checked_result.strip() == 'Y':
                logging.info("Checked result is appropriate. Breaking the loop.")
                break

            ## If the answer is deemed inappropriate, restructure the question and try again
            logging.info("Checked result is inappropriate. Restructuring the question.")
            cypher_query_prompt = self.cypher_query_construction_prompt(self.summary_history, checked_result,
                                                                        self.patient_admission, nodes_edges_results)
            if len(cypher_query_prompt) > max_token:
                cypher_query_prompt = cypher_query_prompt[:max_token]
                logging.info(f"Cypher query prompt truncated to {max_token} characters.")
            cypher_query = self.run_model(cypher_query_prompt, temperature=attempt)
            query_result = self.db.execute_cypher_query(cypher_query, llm_model=self.run_model)
            query_result = self.query_result_rewrite(doctor_query, cypher_query, query_result)
            logging.info(f"New query result: {query_result}")

        ## If after three rounds, still no appropriate answer, return "I don't know."
        if checked_result.strip() != 'Y':
            query_result = ["I don't know"]
            logging.info("After two rounds, still no appropriate answer. Returning 'I don't know'.")

        ## Step Five: Given Query Results, generate the patient response
        logging.info("Step Five: Generating the patient response.")
        if query_result == ["I don't know"]:
            patient_response = "I don't know"
        else:
            rewrite_prompt = self.rewrite_response_prompt(self.summary_history, doctor_query, query_result,
                                                          self.patient_admission, self.personality_profile)
            if len(rewrite_prompt) > max_token:
                rewrite_prompt = rewrite_prompt[:max_token]
                logging.info(f"Rewrite prompt truncated to {max_token} characters.")
            patient_response = self.run_model(rewrite_prompt)
            logging.info(f"Patient response generated: {patient_response}")

        ## Step Six: Update the conversation history
        logging.info("Step Six: Updating the conversation history.")
        summarization_prompt = self.summarize_text_prompt(self.summary_history, doctor_query, patient_response)
        if len(summarization_prompt) > max_token:
            summarization_prompt = summarization_prompt[:max_token]
            logging.info(f"Summarization prompt truncated to {max_token} characters.")
        summarization = self.run_model(summarization_prompt)
        logging.info(f"Conversation history updated: {summarization}")

        ## Update the conversation history based on the most recent interaction
        self.summary_history = summarization
        logging.info(f"Conversation history: {self.summary_history}")

        return patient_response, self.summary_history
    
    def update_interaction(self, role, content):
        """
        追加一条对话记录
        """
        if role not in ("doctor", "patient"):
            raise ValueError("Role must be 'doctor' or 'patient'")

        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)

        self.history.append({
            "role": role,
            "content": content
        })

    def get_conversation_text(self):
        """
        把对话历史拼成带角色的纯文本
        """
        lines = []
        for turn in self.history:
            speaker = "Doctor" if turn["role"] == "doctor" else "Patient"
            lines.append(f"{speaker}: {turn['content']}")
        return "\n".join(lines)
    
    def get_state(self):
        """
        返回初始上下文和交互历史
        """
        conversation = self.get_conversation_text()
        return {
            "initial_info": self.initial_info,
            "interaction_history": conversation,
            "conversation_history": self.summary_history,
            "turn_number": len(self.get_questions()),
        }

    def get_questions(self):
        """
        返回所有医生问题
        """
        return [turn["content"] for turn in self.history if turn["role"] == "doctor"]

    def get_answers(self):
        """
        返回所有患者回答
        """
        return [turn["content"] for turn in self.history if turn["role"] == "patient"]

    def respond_aipatient(self, question):
        """
        使用 AI 生成患者响应
        """
        patient_response, self.summary_history = self.interactive_session(question)
        return patient_response

