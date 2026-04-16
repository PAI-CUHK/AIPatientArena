import time
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError, ServiceUnavailable, SessionExpired, Neo4jError
import logging


class Neo4jDatabase:

    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self._connect()

    def _connect(self):
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=30,
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            keep_alive=True
        )

    def close(self):
        self.driver.close()

    def _reconnect(self):
        try:
            self.driver.close()
        except Exception:
            pass
        self._connect()

    def get_random_patient_admission(self):
        with self.driver.session() as session:
            result = session.execute_read(self._fetch_random_patient_admission)
            return result

    def get_initial_question_answer(self, subject_id, hadm_id):
        query = """
        MATCH (p:Patient {SUBJECT_ID: $subject_id})-[:HAS_ADMISSION]->(a:Admission {HADM_ID: $hadm_id})
        OPTIONAL MATCH (a)-[:HAS_MEDICATION]->(m:Medication)
        WITH a, collect(m.name) AS medication_list

        OPTIONAL MATCH (a)-[:HAS_DRUG_QUESTION]->(q:DrugQuestion)
        
        RETURN 
            head(collect(q.question)) AS question,
            medication_list AS medication
        """
        with self.driver.session() as session:
            result = session.run(query, subject_id=subject_id, hadm_id=hadm_id)
            record = result.single()
            if record:
                return record["question"], record["medication"]
            else:
                return None, None

    def get_patient_demographics(self, subject_id):
        query = """
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        RETURN 
            p.RELIGION AS religion,
            p.MARITAL_STATUS AS marital_status,
            p.GENDER AS gender,
            p.ETHNICITY AS ethnicity,
            p.AGE AS age
        """
        with self.driver.session() as session:
            result = session.run(query, subject_id=subject_id)
            record = result.single()
            if record:
                return {
                    "RELIGION": record["religion"],
                    "MARITAL_STATUS": record["marital_status"],
                    "GENDER": record["gender"],
                    "ETHNICITY": record["ethnicity"],
                    "AGE": record["age"]
                }
            else:
                return None

    def get_patient_diagnoses(self, subject_id, hadm_id):
        query = """
        MATCH (p:Patient {SUBJECT_ID: $subject_id})-[:HAS_ADMISSION]->(a:Admission {HADM_ID: $hadm_id})
        MATCH (a)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        RETURN collect(d.name) AS diagnoses
        """
        with self.driver.session() as session:
            result = session.run(query, subject_id=subject_id, hadm_id=hadm_id)
            record = result.single()
            if record and record["diagnoses"]:
                return record["diagnoses"]
            else:
                return None

    def get_all_drug_answers_list(self, subject_id, admission_id):
        query = """
        MATCH (p:Patient {SUBJECT_ID: $subject_id})-[:HAS_ADMISSION]->(a:Admission {HADM_ID: $admission_id})
        MATCH (a)-[:HAS_DRUG_QUESTION]->(q:DrugQuestion)-[:HAS_DRUG_ANSWER]->(target:DrugAnswer)
        WITH target, elementId(target) AS targetId

        MATCH (d:DrugAnswer)
        WITH target, targetId, d, elementId(d) AS dId
        ORDER BY dId

        WITH target, targetId, COLLECT({id: dId, answer: d.answer}) AS all_answers

        WITH all_answers, [i IN RANGE(0, SIZE(all_answers)-1) WHERE all_answers[i].id = targetId][0] AS idx

        WITH all_answers[CASE WHEN idx - 5 < 0 THEN 0 ELSE idx - 5 END .. idx + 5] AS nearby_answers

        RETURN [item IN nearby_answers | item.answer] AS nearby_answers

        """
        with self.driver.session() as session:
            result = session.run(query, subject_id=subject_id, admission_id=admission_id)
            record = result.single()
            if record:
                return record["nearby_answers"]
            else:
                return []

    def get_all_patient_admissions(self):
        query = """
        MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission)
        RETURN p.SUBJECT_ID AS SubjectID, a.HADM_ID AS AdmissionID
        """
        with self.driver.session() as session:
            result = session.run(query)
            patient_admission_list = []
            for record in result:
                patient_admission_list.append({
                    "SubjectID": record["SubjectID"],
                    "AdmissionID": record["AdmissionID"]
                })
            return patient_admission_list

    ## Step One: randomly select patient from graph database
    @staticmethod
    def _fetch_random_patient_admission(tx):
        query = """
        MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission)
        WITH p, a, rand() AS random
        ORDER BY random
        LIMIT 1
        RETURN p.SUBJECT_ID AS SubjectID, a.HADM_ID AS AdmissionID
        """
        result = tx.run(query)
        return result.single()

    ## Step Three: information extraction
    def execute_cypher_query(self, cypher_query, llm_model=None, retries=3):
        for attempt in range(retries):
            try:
                with self.driver.session() as session:
                    result = session.execute_read(self._run_cypher_query, cypher_query)
                    return result

            except (CypherSyntaxError, ClientError) as e:
                # 语法类错误，尝试用 LLM 修复
                logging.error(f"Cypher query failed: {e}")
                if llm_model:
                    repair_prompt = (
                        f"The following Cypher query caused an error:\n{cypher_query}\n\n"
                        f"Error Message:\n{str(e)}\n\n"
                        f"Please generate a corrected Cypher query that fixes the issue. "
                        f"Only return the corrected Cypher query as plain text."
                    )
                    repaired_query = llm_model(repair_prompt)
                    logging.info(f"Repaired Cypher query: {repaired_query}")
                    try:
                        with self.driver.session() as session:
                            result = session.execute_read(self._run_cypher_query, repaired_query)
                            return result
                    except Exception as e2:
                        logging.error(f"Repaired query still failed: {e2}")
                        return {
                            "error": "Repaired query execution failed.",
                            "original_error": str(e),
                            "repair_error": str(e2),
                            "original_query": cypher_query,
                            "repaired_query": repaired_query
                        }
                else:
                    raise e

            except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
                # 网络类错误：断线、超时、连接池挂掉等
                logging.warning(f"[Neo4j Retry {attempt + 1}/{retries}] {type(e).__name__}: {e}")
                self._reconnect()  # 👈 关键：自动重连
                time.sleep(2 ** attempt)  # 指数退避：1s -> 2s -> 4s

            except Exception as e:
                # 其他未知错误
                logging.error(f"Unknown error during Cypher execution: {e}")
                raise e

        raise RuntimeError("Failed to execute Cypher query after multiple retries.")

    @staticmethod
    def _run_cypher_query(tx, cypher_query):
        result = tx.run(cypher_query)
        return [record.data() for record in result]

