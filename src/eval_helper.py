from llm_model_class import LLM_Models
import json
from helper import *


def run_model(messages, model: LLM_Models):
    
    return model.run_gpt(messages=messages, model='gpt-5')



def questioning_skills_instruction(dialogue, llm_models):
    """
    输入：
      dialogue - 一个包含多条交流的列表，每条是字典，结构示例见下面。
    输出：
      一个字符串，作为给LLM的完整提示词。
    """
    prompt = f"""
You are a clinical consultation quality evaluator.

You will be given a structured list of doctor-patient dialogue.

Your task is to evaluate the quality of the doctor's questions during the medical consultation, identifying any problematic questions and providing explanations for each evaluation.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY step. No step may be skipped.
    - One step is defined as one doctor question together with the immediately following patient response.
2. Whether the doctor's question is correct or incorrect, you MUST explain the reason.
   - For appropriate questions: give a brief justification.
   - For problematic questions: give a clear and specific explanation.
3. Focus ONLY on the doctor's question.
   - Do NOT analyze or judge the patient's response.
4. Be strict, but do NOT over-penalize clinically reasonable follow-up questions.
5. If a question is problematic, assign ONLY ONE primary problem category.

--------------------------------------------------
PROBLEM CATEGORIES
--------------------------------------------------

**1. logical_gap**  
Abrupt topic shifts without clear connection to prior content.

**2. repetitive_question**
Repeated questions requesting already provided medical information. Clarifying vague answers or exploring new details is not considered repetition.


**3. unclear_question**  
Question combining two or more unrelated clinical topics in a single unstructured sentence without clear boundaries.

**4. suggestive_question**  
Question that suggest or bias the patient's answer.

**5. confrontational_question**  
Question that sound aggressive, judgmental, or defensive-provoking.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:

{
  "step_evaluations": {
    "Step 1": {
      "assessment": "appropriate" | "problematic",
      "error_type": "logical_gap" | "repetitive_question" | "unclear_question" | "suggestive_question" | "confrontational_question" | "none",
      "explanation": "<brief reason if appropriate, detailed reason if problematic>"
    },
    "Step 2": {
      ...
    }
  },
  "summary_counts": {
    "logical_gap": <int>,
    "repetitive_question": <int>,
    "unclear_question": <int>,
    "suggestive_question": <int>,
    "confrontational_question": <int> 
  }
}

--------------------------------------------------
IMPORTANT
--------------------------------------------------
- Every step MUST appear in step_evaluations.
- If the question is appropriate, set:
  - assessment = "appropriate"
  - error_type = "none"
- Do NOT include any text outside the JSON output.

Now evaluate the following dialogue:
"""
    prompt += json.dumps(dialogue, indent=2, ensure_ascii=False)

    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)

    return response


def calculate_questioning_skills_score(full_output):
    """
    Calculate questioning skills score and generate step-by-step explanations
    for BOTH appropriate and problematic questions, stored as a JSON object
    with 'Step X' as keys.
    """

    deduction_rules = {
        "logical_gap": {"deduct": 0.5, "max": 1.0},
        "repetitive_question": {"deduct": 0.5, "max": 1.0},
        "unclear_question": {"deduct": 0.5, "max": 1.5},
        "suggestive_question": {"deduct": 0.25, "max": 0.5},
        "confrontational_question": {"deduct": 0.25, "max": 1.0},
    }

    total_score = 5.0
    step_explanations = {}

    step_evaluations = full_output.get("step_evaluations", [])
    summary_counts = full_output.get("summary_counts", {})

    # ---- Step-by-step explanations as JSON (ALL steps) ----
    for step in step_evaluations:
        step_num = step.get("step_number")
        assessment = step.get("assessment")
        error_type = step.get("error_type")
        explanation = step.get("explanation", "")

        step_key = f"Step {step_num}"

        if assessment == "appropriate":
            step_explanations[step_key] = {
                "assessment": "appropriate",
                "reason": explanation
            }
        else:
            step_explanations[step_key] = {
                "assessment": "problematic",
                "error_type": error_type,
                "reason": explanation
            }

    # ---- Apply score deductions based on summary_counts ----
    issue_details = {}

    for issue, rule in deduction_rules.items():
        count = summary_counts.get(issue, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        issue_details[issue] = count

    result = {
        "score": max(0.0, round(total_score, 2)),
        "explanation": step_explanations,
        "details": issue_details
    }

    return result




def information_coverage_instruction(dialogue, llm_models):
    prompt = f"""
You are a clinical consultation quality evaluator.

You will be given a structured list of doctor-patient dialogue.

Your task is to assess whether the doctor has covered key clinical information domains. 
For each domain, first determine whether it is truly relevant for this case in supporting diagnosis, treatment, or safe medication decisions. 
If a domain is irrelevant to the case, mark it as "not-applicable". 
If the domain is relevant, assess coverage as follows:
- "present": The domain was explicitly addressed and core elements were covered.
- "missing-major": A critical domain is entirely missing, undermining clinical understanding.
- "missing-minor": A contextually relevant domain is not addressed, reducing completeness.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY domain. No domain may be skipped.
2. For every domain, you MUST provide a clear explanation for your assessment.
3. Focus ONLY on the information explicitly provided in the dialogue.
4. Be objective and evidence-based in your evaluations.
5. If a domain is not required for clinical decision-making, its absence MUST NOT be counted as an omission.

--------------------------------------------------
EVALUATION DOMAINS
--------------------------------------------------

1. chief_complaint_details — Ask about timing, duration, severity, progression, and triggers of the main complaint.

2. past_medical_history — Major chronic illnesses, hospitalizations, surgeries.
- "present" requires evidence of **active inquiry into multiple relevant aspects** of past medical history. Not just a single complaint-related condition; a reasonable breadth of chronic illnesses, hospitalizations, and surgeries must be explored.
- "missing-minor" if some PMH information was obtained but other relevant conditions or hospitalizations were not asked, reducing completeness.
- "missing-major" if **no past medical history was asked at all**, significantly limiting understanding.
- "not-applicable" if PMH is irrelevant for this case.
3. allergies — Medication or substance allergies.

4. relevant_symptom_review — Associated symptoms related to the main complaint.

5. family_history — Genetic or familial risk factors.

6. social_history — Smoking, alcohol, occupation, living environment.

7. lifestyle_factors — Diet, exercise, sleep, stress.

8. broader_review_of_systems — Other unrelated organ systems.

9. mental_status — Mood, cognition, affect, or psychiatric symptoms.

10. specific_risk_factors — Travel, exposures, sexual history, immunizations.


Instructions:

1. Explanation requirement:
   - For every domain, the explanation must first briefly state whether and why this domain is clinically needed or not needed for this specific case.
   - Then explain why the final status (present / missing-major / missing-minor / not-applicable) was assigned.
   
2. Core decision-impact rule:
   - A domain may only be counted as a major or minor omission if its absence would reasonably influence at least one of the following in this specific case:
    - the diagnostic conclusion,
    - the immediate or expected management plan,
    - or medication safety, including potential risks of standard therapies that would normally be considered in this context.
   - If a domain is not required to make or change any concrete or anticipated clinical decision, its absence MUST NOT be counted as an omission, regardless of whether the domain is commonly collected in general clinical practice.
Output only the following JSON format:

{{
  "major_omission": <int>,
  "minor_omission": <int>,
  "explanation": {{
    "chief_complaint_details": {{
      "status": "present" | "missing-major" | "missing-minor" | "not-applicable",
      "explanation": "<short justification>"
    }},
    "past_medical_history": {{
      "status": "...",
      "explanation": "..."
    }},
    "allergies": {{
      "status": "...",
      "explanation": "..."
    }},
    "relevant_symptom_review": {{
      "status": "...",
      "explanation": "..."
    }},
    "family_history": {{
      "status": "...",
      "explanation": "..."
    }},
    "social_history": {{
      "status": "...",
      "explanation": "..."
    }},
    "lifestyle_factors": {{
      "status": "...",
      "explanation": "..."
    }},
    "broader_review_of_systems": {{
      "status": "...",
      "explanation": "..."
    }},
    "mental_status": {{
      "status": "...",
      "explanation": "..."
    }},
    "specific_risk_factors": {{
      "status": "...",
      "explanation": "..."
    }}
  }}
}}


Notes:
- Do not infer any information not explicitly elicited in the dialogue.
- Always justify why a domain is marked "not-applicable" if it is skipped.
- Explanations must be concise but explicitly reference domain relevance to the case.

Now evaluate the following dialogue:
"""
    prompt += json.dumps(dialogue, indent=2, ensure_ascii=False)

    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_information_coverage_score(full_output):
    """
    Calculate self-awareness score and record explanations for ALL domains,
    including present, missing-major, missing-minor, and not-applicable.
    """

    deduction_rules = {
        "major_omission": {"deduct": 1.0, "max": 3.0},
        "minor_omission": {"deduct": 0.5, "max": 2.0},
    }

    total_score = 5.0

    # Input fields
    explanations = full_output.get("explanation", {})
    issue_counts = {
        "major_omission": full_output.get("major_omission", 0),
        "minor_omission": full_output.get("minor_omission", 0)
    }

    # ---- Apply score deductions ----
    for error_type, rule in deduction_rules.items():
        count = issue_counts.get(error_type, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction

    # ---- Record ALL explanations by domain ----
    # No filtering: every domain is preserved as-is
    domain_explanations = {}

    for domain, info in explanations.items():
        domain_explanations[domain] = {
            "status": info.get("status", "unknown"),
            "explanation": info.get("explanation", "")
        }

    result = {
        "score": max(0.0, round(total_score, 2)),
        "explanation": domain_explanations,
        "details": issue_counts
    }

    return result



def robustness_instruction(dialogue, llm_models):
    prompt = f"""
You are a medical dialogue robustness evaluator.

You will be given a structured list of doctor-patient dialogue.

Your task is to identify and count **only the following two types of robustness failures** in a medical dialogue:

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY step. No step may be skipped.
    - One step is defined as one doctor question together with the immediately following patient response.
2. For each step, you must explicitly state whether the handling is:
   - appropriate
   - or problematic
3. Even if the step is appropriate, you MUST explain WHY it is appropriate.
4. If the step is problematic, you MUST explain WHY it is problematic and classify the error type.
5. Be conservative: only mark as error if missing follow-up clearly harms reasoning or info quality.

--------------------------------------------------
ERROR TYPES
--------------------------------------------------

**1. unclear_expression_handling**
Failure to clarify vague or grouped symptom descriptions within three turns.

**2. uncertain_answer_handling**  
Failure to clarify follow-up within three turns.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:

{{
  "step_evaluations": {{
    "Step 1": {{
      "assessment": "appropriate" | "problematic",
      "error_type": "unclear_expression_handling" | "uncertain_answer_handling" | "none",
      "explanation": "<why this step is appropriate or why it is problematic>"
    }},
    "Step 2": {{
      ...
    }}
  }},
  "summary_counts": {{
    "unclear_expression_handling": <int>,
    "uncertain_answer_handling": <int>
  }}
}}

Now evaluate the following dialogue:
"""
    prompt += json.dumps(dialogue, indent=2, ensure_ascii=False)

    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_robustness_score(full_output):
    """
    Calculate robustness score and record step-by-step explanations
    for BOTH appropriate and problematic handling.
    """

    deduction_rules = {
        "unclear_expression_handling": {"deduct": 0.5, "max": 2.5},
        "uncertain_answer_handling": {"deduct": 0.5, "max": 2.5},
    }

    total_score = 5.0

    step_explanations = {}

    step_evaluations = full_output.get("step_evaluations", {})
    summary_counts = full_output.get("summary_counts", {})

    # ---- Step-by-step explanations (ALL steps) ----
    for step_key, info in step_evaluations.items():
        assessment = info.get("assessment")
        error_type = info.get("error_type", "none")
        explanation = info.get("explanation", "")

        if assessment == "appropriate":
            step_explanations[step_key] = {
                "assessment": "appropriate",
                "reason": explanation
            }
        else:
            step_explanations[step_key] = {
                "assessment": "problematic",
                "error_type": error_type,
                "reason": explanation
            }

    # ---- Apply score deductions ----
    issue_details = {}

    for issue, rule in deduction_rules.items():
        count = summary_counts.get(issue, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        issue_details[issue] = count

    result = {
        "score": max(0.0, round(total_score, 2)),
        "explanation": step_explanations,
        "details": issue_details
    }

    return result



def ethics_instruction(dialogue, llm_models):
    prompt = """
You are a medical ethics evaluator.

You will be given a structured doctor-patient dialogue. Your task is to evaluate whether the clinician behaves ethically and professionally.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY step. No step may be skipped.
    - One step is defined as one doctor question together with the immediately following patient response.
2. For each step, explicitly state whether the doctor's behavior is:
   - ethical
   - or problematic
3. Even if the step is ethical, you MUST explain WHY it is ethical.
4. If the step is problematic, you MUST:
   - Identify the ethical issue type
   - Explain clearly why it is problematic
5. Focus ONLY on the doctor’s speech. The patient’s immediately preceding utterance is only used to determine lack_of_empathy.

--------------------------------------------------
ETHICAL CONCERNS
--------------------------------------------------

**1. Disrespectful tone**
Rude, dismissive, or condescending communication.

**2. Bias or assumptions**
Unjustified assumptions about lifestyle, behavior, or illness cause.

**3. Privacy violation**
Overly intrusive questions without medical justification.

**4. Lack of empathy**
Ignoring or downplaying patient concerns or emotions.

Do **not** flag as lack of empathy if:
- The patient simply says “I don’t know,” “I’m not sure,” or “I can’t remember” **without emotional tone**;
- And the doctor continues with clinically appropriate questions, in a respectful and neutral tone.

Only consider the doctor’s speech. For each issue type, count the number of occurrences and provide short explanations.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:
{
  "step_evaluations": {
    "Step 1": {
      "assessment": "ethical" | "problematic",
      "error_type": "disrespectful_tone" | "bias_or_assumption" | "privacy_violation" | "lack_of_empathy" | "none",
      "explanation": "<why this step is ethical or why it is problematic>"
    },
    "Step 2": {
      ...
    }
  },
  "summary_counts": {
    "disrespectful_tone": <int>,
    "bias_or_assumption": <int>,
    "privacy_violation": <int>,
    "lack_of_empathy": <int>
  }
}
Now evaluate the following dialogue:

"""

    prompt += json.dumps(dialogue, indent=2, ensure_ascii=False)

    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_ethics_score(full_output):
    """
    Calculate ethics score and record step-by-step ethical evaluations
    for BOTH ethical and problematic steps.
    """

    deduction_rules = {
        "disrespectful_tone": {"deduct": 1.0, "max": 2.0},
        "bias_or_assumption": {"deduct": 1.0, "max": 2.0},
        "privacy_violation": {"deduct": 1.0, "max": 2.0},
        "lack_of_empathy": {"deduct": 1.0, "max": 2.0}
    }

    total_score = 5.0

    step_explanations = {}

    step_evaluations = full_output.get("step_evaluations", {})
    summary_counts = full_output.get("summary_counts", {})

    # ---- Step-by-step explanations (ALL steps) ----
    for step_key, info in step_evaluations.items():
        assessment = info.get("assessment")
        error_type = info.get("error_type", "none")
        explanation = info.get("explanation", "")

        if assessment == "ethical":
            step_explanations[step_key] = {
                "assessment": "ethical",
                "reason": explanation
            }
        else:
            step_explanations[step_key] = {
                "assessment": "problematic",
                "error_type": error_type,
                "reason": explanation
            }

    # ---- Apply score deductions based on summary_counts ----
    issue_details = {}

    for error_type, rule in deduction_rules.items():
        count = summary_counts.get(error_type, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        issue_details[error_type] = count

    result = {
        "score": max(0.0, round(total_score, 2)),
        "explanation": step_explanations,
        "details": issue_details
    }

    return result



def explainability_instruction(dialogue, llm_models):
    prompt = """
You are a clinical communication evaluator focusing specifically on whether the doctor's individual questions during a medical interview are understandable to the patient.

You will be given a structured list of doctor-patient dialogue.

Your task is to evaluate whether the doctor's individual questions during a medical interview are understandable to the patient.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY step. No step may be skipped.
    - One step is defined as one doctor question together with the immediately following patient response.
2. For each step, explicitly state whether the doctor’s question is:
   - clear
   - or problematic
3. Even if the question is clear, you MUST explain WHY it is clear.
4. If the question is problematic, you MUST:
   - Identify the clarity issue type
   - Explain clearly why the question may be difficult to understand
5. Assume patients do NOT understand medical terminology by default.

--------------------------------------------------
CLARITY ISSUE TYPES
--------------------------------------------------

**1. major_clarity_failure**
Use of unexplained medical terminology that seriously hinders patient understanding.

**2. minor_clarity_failure**
Use of potentially confusing terms without clarification, causing mild comprehension difficulty.

**3. lack_of_context_for_question**
Asking potentially alarming or confusing questions without explaining purpose or relevance.

----------------------------------------------------------------
CLARITY INTERPRETATION RULES
----------------------------------------------------------------

DEFAULT ASSUMPTION:
- Assume patients do NOT understand medical terminology by default.

- Medical terminology includes named diseases, diagnostic tests, procedures,
or formal medical terms that are not commonly used in everyday speech.

- Do NOT flag a clarity failure if:
  - The patient previously introduced the term.
  - The question is a routine symptom inquiry (e.g., pain, appetite, sleep).
  - The purpose of the question is clearly inferable from prior context.

- EXPLANATORY LIST EXCEPTION:
  If medical terms are used as examples within a broader explanatory list
  (e.g., “heart disease, such as heart attack, angina, or bypass surgery”),
  and the intent is to confirm medical history,
  this should NOT be considered a clarity failure.

----------------------------------------------------------------
COUNTING RULE (MANDATORY)
----------------------------------------------------------------

- For summary_counts:
  1. Review all step_evaluations and identify each clarity issue.
  2. Treat repeated occurrences of the same issue (for example, the same unexplained medical term) as a single distinct problem.
  3. Count each distinct problem only once, even if it appears in multiple steps.
  4. Do NOT increment counts based on the number of steps or repeated mentions.
  5. Each issue should still be explained at the step level where it occurs.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:

{
  "step_evaluations": {
    "Step 1": {
      "assessment": "clear" | "problematic",
      "error_type": "major_clarity_failure" | "minor_clarity_failure" | "lack_of_context_for_question" | "none",
      "explanation": "<why the question is clear OR why it is problematic>"
    },
    "Step 2": {
      ...
    }
  },
  "summary_counts": {
    "major_clarity_failure": <int>,
    "minor_clarity_failure": <int>,
    "lack_of_context_for_question": <int>
  }
}


Now evaluate the following dialogue:
"""

    prompt += json.dumps(dialogue, indent=2, ensure_ascii=False)
    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_explainability_score(full_output):
    """
    Calculate explainability (clarity) score and record step-by-step explanations
    for BOTH clear and problematic doctor questions.
    """

    deduction_rules = {
        "major_clarity_failure": {"deduct": 1.0, "max": 2.0},
        "minor_clarity_failure": {"deduct": 0.5, "max": 1.5},
        "lack_of_context_for_question": {"deduct": 0.5, "max": 1.5},
    }

    total_score = 5.0

    step_explanations = {}

    step_evaluations = full_output.get("step_evaluations", {})
    summary_counts = full_output.get("summary_counts", {})

    # ---- Step-by-step explanations (ALL steps) ----
    for step_key, info in step_evaluations.items():
        assessment = info.get("assessment")          # clear / problematic
        error_type = info.get("error_type", "none")
        explanation = info.get("explanation", "")

        if assessment == "clear":
            step_explanations[step_key] = {
                "assessment": "clear",
                "reason": explanation
            }
        else:
            step_explanations[step_key] = {
                "assessment": "problematic",
                "error_type": error_type,
                "reason": explanation
            }

    # ---- Apply score deductions based on summary_counts ----
    issue_details = {}

    for issue, rule in deduction_rules.items():
        count = summary_counts.get(issue, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        issue_details[issue] = count

    result = {
        "score": max(0.0, round(total_score, 2)),
        "explanation": step_explanations,
        "details": issue_details
    }

    return result



def information_summary_instruction(dialogue, summary, llm_models):
    """
    调用大模型评估对话中患者信息总结的准确性，
    统计并举例说明重大遗漏、微小遗漏、重大捏造、微小捏造。

    参数：
    - dialogue: 医患对话数据（结构化列表）
    - summary: 模型或人工总结的患者信息文本（字符串）
    - llm_models: 调用大模型的对象，需支持 run_gpt 方法

    返回值：
    - 解析后的 JSON 字典，包含各类计数和解释示例
    """

    prompt = f"""
You are a medical summary accuracy evaluator.


You will be given:
 -A structured list of doctor-patient dialogue.
- A patient information summary extracted from the dialogue.

Your task: Identify inaccuracies in the summary by comparing it directly to the dialogue.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate the summary BY CATEGORY.
2. Every category MUST be evaluated, even if it is completely correct.
3. Do NOT skip any category.
4. Only use explicit dialogue content. Do not infer unstated information.
5. For the same piece of information, assign only one error type (choose omission OR fabrication).
6. Only classify as major if the error could substantially change diagnosis or treatment decisions.

--------------------------------------------------
EVALUATION CATEGORIES
--------------------------------------------------

CATEGORIES TO EVALUATE:
- symptoms
- medical_history
- family_history
- vital_signs
- allergies

For EACH category:
1. Decide whether the summary content for this category is:
   - "correct"
   - "omission"
   - "fabrication"

2. Always explain WHY.
   - If correct: briefly explain why the summary is accurate or appropriately marked as not mentioned.
   - If incorrect: list all specific errors found.

--------------------------------------------------
ERROR TYPES
--------------------------------------------------

Error Types:
- **major_omission**: Missing critical information affecting diagnosis or treatment.

- **minor_omission**: Missing contextual information not affecting key clinical decisions.

- **major_fabrication**: False or unsupported information that could mislead diagnosis or treatment.

- **minor_fabrication**: Interpretive errors not materially affecting clinical decisions.

Evaluation Rules:
1. Source of Truth
Use only explicit dialogue content. Do not infer missing information.

2. Denials
All explicit patient denials must appear in the summary.
Missing denials:
Critical → major_omission
Non-critical → minor_omission
Denials must not be changed into uncertainty or affirmation.

3. Uncertainty
“Not sure / don’t know / maybe” = uncertain.
Omitting uncertain info is allowed.
Converting uncertainty into certainty → minor_fabrication.

4. Counting
Count each missing or fabricated item separately.

5. Not Mentioned
If not in dialogue, “not mentioned” in summary is correct.

6. Demographics
May appear even if absent from dialogue (not an error).

7. No Double Labeling
Assign only one error type per item.

8. Severity

Impacts diagnosis/treatment → major
Otherwise → minor

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:

{{
  "category_evaluation": {{
    "symptoms": {{
      "covered_steps": [1, 3, 5],
      "missing_steps": [7],
      "fabricated_items": [
        "Summary reports fever not mentioned in dialogue"
      ],
      "errors": [
        {{
          "type": "minor_omission",
          "step_number": 7,
          "description": "Nausea reported by patient was not included in the summary"
        }},
        {{
          "type": "major_fabrication",
          "description": "Summary states patient had fever without dialogue support"
        }}
      ],
      "notes": "Most core symptoms captured; one minor symptom omitted and one unsupported symptom added."
    }},

    "medical_history": {{
      "covered_steps": [4],
      "missing_steps": [],
      "fabricated_items": [],
      "errors": [...],
      "notes": "Medical history accurately summarized."
    }},

    "family_history": {{
      "covered_steps": [],
      "missing_steps": [],
      "fabricated_items": [],
      "errors": [],
      "notes": "Family history not discussed in dialogue; correctly omitted."
    }},

    "vital_signs": {{
      "covered_steps": [],
      "missing_steps": [],
      "fabricated_items": [],
      "errors": [],
      "notes": "No vital signs mentioned in dialogue; summary appropriately does not infer values."
    }},

    "allergies": {{
      "covered_steps": [6],
      "missing_steps": [],
      "fabricated_items": [],
      "errors": [],
      "notes": "Allergy denial correctly included."
    }}
  }},

  "error_counts": {{
    "major_omission": <int>,
    "minor_omission": <int>,
    "major_fabrication": <int>,
    "minor_fabrication": <int>
  }}
}}

Now evaluate the following dialogue and summary:
Dialogue:
{json.dumps(dialogue, indent=2, ensure_ascii=False)}

Summary:
\"\"\"{summary}\"\"\"
"""

    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_summary_accuracy_score(full_output):
    """
    Calculate summary accuracy score aligned with
    category-first, step-mapped summary evaluation output.
    """

    deduction_rules = {
        "major_omission": {"deduct": 0.5, "max": 1.5},
        "minor_omission": {"deduct": 0.1, "max": 0.5},
        "major_fabrication": {"deduct": 1.5, "max": 3.0},
        "minor_fabrication": {"deduct": 0.5, "max": 0.5},
    }

    total_score = 5.0

    # Error counts are authoritative for scoring
    error_counts = full_output.get("error_counts", {})

    # Apply deductions
    for error_type, rule in deduction_rules.items():
        count = error_counts.get(error_type, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction

    result = {
        "score": max(0.0, round(total_score, 2)),
        # Keep explanation structured and traceable
        "explanation": full_output.get("category_evaluation", {}),
        # DETAILS FORMAT CONSISTENT WITH PREVIOUS FUNCTIONS
        "details": {
            "major_omission": error_counts.get("major_omission", 0),
            "minor_omission": error_counts.get("minor_omission", 0),
            "major_fabrication": error_counts.get("major_fabrication", 0),
            "minor_fabrication": error_counts.get("minor_fabrication", 0),
        }
    }

    return result



def diagnostic_right_instruction(diagnosis, reference_diagnoses, llm_models):
    """
    Part A: Evaluate the final diagnosis accuracy compared to reference diagnoses.
    """
    prompt = f"""
You are a diagnostic reasoning evaluator.

You will be given:
- A final diagnosis made by the clinician
- A list of correct or acceptable reference diagnoses

Task: Perform a Diagnosis Accuracy Evaluation using the following rules:

Step 1: Compare the final diagnosis to the reference diagnoses.
- If the final diagnosis matches **any one** of the reference diagnoses, classify as:
  - **exact_match**: identical to a reference diagnosis → deduction 0
  - **partial_match**: clinically close (e.g., different wording, subtypes, synonyms) but essentially the same condition → deduction 0.5

- If the final diagnosis does **not** sufficiently match any reference diagnosis, classify as:
  - **minorly_incorrect**: diagnosis is plausible but incomplete or misses key aspects of the patient's condition → deduction 1.0
  - **majorly_incorrect**: diagnosis is unrelated or misleading → deduction 1.5

**Note:**  
- Matching any one reference diagnosis is sufficient to avoid higher deductions.  
- Do not require the diagnosis to cover all reference diagnoses unless explicitly specified.

Then explain your match judgment, including why it qualifies for the selected category.

Return results in this JSON format:
{{
  "diagnosis_accuracy": {{
    "match_level": "...",
    "deduction": ...,
    "comment": "..."
  }}
}}

Clinician Diagnosis:
\"\"\"{diagnosis}\"\"\"

Reference Diagnoses:
\"\"\"{reference_diagnoses}\"\"\"
"""
    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    return json.loads(response)


def diagnostic_reasoning_instruction(dialogue, patient_info, diagnosis_reasoning, diagnosis, llm_models):
    """
    Part B: Evaluate reasoning errors in the clinician's diagnostic process.
    """
    prompt = f"""
You are a diagnostic reasoning evaluator.

You will be given:
- Structured doctor–patient dialogue.
- Patient personal information.
- Clinician's diagnostic reasoning process.
- Clinician's final diagnosis.

Task:
- Evaluate strictly the clinician's **diagnostic reasoning process**, based only on the dialogue and patient information.
- Do NOT generate new diseases, tests, imaging, or lab results.
- Focus strictly on logic, completeness, and internal consistency.
- Do NOT evaluate the dialogue itself; only the reasoning process is evaluated.

Error Types:

1. **information_omission**: Reasoning introduces unsupported facts treated or confirmed
2. **incorrect_assumption**: Clinically invalid, disorganized, or step-jumping reasoning.
3. **flawed_reasoning_process**: Clinically invalid, disorganized, or step-jumping reasoning.
4. **diagnosis_reasoning_mismatch**: The reasoning process does not logically lead to the clinician’s final diagnosis.
5. **diagnostic_questioning_bias**: Assign this when multiple reasonable differential diagnoses exist, but only one line of questioning is pursued without justification.

Counting and Explanation:
- Count each error type independently.
- Do not assign the same fact to multiple error types.
- Explain each error **with reference to dialogue or patient info**.


Return results in this JSON format:
{{
  "reasoning_errors": {{
    "information_omission": <int>,
    "incorrect_assumption": <int>,
    "flawed_reasoning_process": <int>,
    "diagnosis_reasoning_mismatch": <int>,
    "diagnostic_questioning_bias": <int>,
    "explanation": {{
      "information_omission": [
        "Explanation of omitted facts OR explanation of why no relevant facts were omitted."
      ],
      "incorrect_assumption": [
        "Explanation of unsupported assumptions OR explanation of why no unsupported facts were introduced."
      ],
      "flawed_reasoning_process": [
        "Explanation of flawed reasoning OR explanation of why the reasoning appropriately interprets all mentioned facts."
      ],
      "diagnosis_reasoning_mismatch": [
        "Explanation of mismatch OR explanation of why the final diagnosis logically follows from the reasoning."
      ],
      "diagnostic_questioning_bias": [
        "Explanation of biased questioning (e.g., focusing on a single diagnostic direction while omitting other reasonable directions) OR explanation of why questioning appropriately screened multiple reasonable directions."
      ]
    }}
  }}
}}

Patient Personal Information:
{json.dumps(patient_info, ensure_ascii=False, indent=2)}

Doctor-Patient Dialogue:
{json.dumps(dialogue, ensure_ascii=False, indent=2)}

Clinician Diagnosis:
{diagnosis}

Clinician Diagnosis reasoning:
{json.dumps(diagnosis_reasoning, ensure_ascii=False, indent=2)}
"""
    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    return json.loads(response)


def diagnostic_unkown_instruction(dialogue, reference_diagnoses, llm_models):
    prompt = f"""
        Task:
        You are an AI evaluator. Based on the doctor–patient dialogue and the list of reference diagnoses, determine whether the dialogue provides enough information to reasonably support **at least one** of the reference diagnoses.

        Key Rule:
        - Only consider what is stated in the dialogue.
        - It is NOT required to have exam findings, labs, imaging, or complete medical details.
        - As long as the dialogue contains clues that logically fit any diagnosis in the reference list, the result should be true.
        - If none of the reference diagnoses can be supported by the dialogue, the result should be false.

        Output the result in this JSON format:

        {{
            "determinable": true/false,
            "explanation": "Why the dialogue does or does not support at least one reference diagnosis."
        }}

        Here is the dialogue:
        {json.dumps(dialogue, ensure_ascii=False, indent=2)}
        
        Reference Diagnoses:
        \"\"\"{reference_diagnoses}\"\"\"
    """
    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    return json.loads(response)


def evaluate_diagnosis_case(patient_info, diagnosis, llm_models):
    """
    Run both parts and combine the results into a single JSON.
    """
    diagnosis_result = diagnostic_right_instruction(diagnosis["clinician_provided_diagnosis"],
                                                    diagnosis["correct_diagnoses"], llm_models)
    reasoning_result = diagnostic_reasoning_instruction(patient_info['dialogue'], patient_info['info'],
                                                        diagnosis["diagnosis_reasoning"],
                                                        diagnosis["clinician_provided_diagnosis"], llm_models)
    diagnosis_certain = diagnostic_unkown_instruction(patient_info['dialogue'], diagnosis["correct_diagnoses"],
                                                      llm_models)
    combined_result = {
        "diagnosis_accuracy": diagnosis_result.get("diagnosis_accuracy", {}),
        "reasoning_errors": reasoning_result.get("reasoning_errors", {}),
        "diagnosis_certain": diagnosis_certain,
    }

    return combined_result


def calculate_diagnostic_reasoning_score(full_output):
    reasoning_weights = {
        "information_omission": {"deduct": 0.25, "max": 1.0},
        "incorrect_assumption": {"deduct": 0.5, "max": 1.0},
        "flawed_reasoning_process": {"deduct": 0.25, "max": 0.5},
        "diagnosis_reasoning_mismatch": {"deduct": 0.5, "max": 0.5},
        "diagnostic_questioning_bias":{"deduct": 0.5, "max": 0.5},
    }

    total_score = 5.0
    explanation_lines = []

    # 获取诊断准确性部分
    diagnosis_accuracy = full_output.get("diagnosis_accuracy", {})
    match_level = diagnosis_accuracy.get("match_level", "")
    deduction_from_accuracy = diagnosis_accuracy.get("deduction", 0)
    comment = diagnosis_accuracy.get("comment", "")

    total_score -= deduction_from_accuracy

    if deduction_from_accuracy > 0:
        explanation_lines.append(
            f"diagnosis_accuracy ({match_level}): -{deduction_from_accuracy}\n  {comment}"
        )

    # 获取推理错误信息并扣分
    reasoning_errors = full_output.get("reasoning_errors", {})
    reasoning_explanations = reasoning_errors.get("explanation", {})
    deduction_details = {}

    for err_type, rule in reasoning_weights.items():
        count = reasoning_errors.get(err_type, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        deduction_details[err_type] = count

        lines = reasoning_explanations.get(err_type, [])
        if count > 0:

            explanation_lines.append(
                f"{err_type}: {count} occurrence(s), -{deduction}\n  " + "\n  ".join(lines)
            )
        else:
            explanation_lines.append(
                f"{err_type}: 0 occurrence(s)\n  " +
                "\n  ".join(lines)
            )

    # 诊断判断部分
    diagnosis_certain = full_output.get("diagnosis_certain", {})
    determinable = diagnosis_certain.get("determinable", False)
    diagnosis_certain_explanations = diagnosis_certain.get("explanation", {})

    explanation_lines.append(f"diagnosis_certain_explanations: " + diagnosis_certain_explanations)

    return {
        "score": max(0.0, round(total_score, 2)),
        "diagnosis_accuracy": {
            "match_level": match_level,
            "deduction": deduction_from_accuracy,
            "comment": comment
        },
        "reasoning_errors": deduction_details,
        "determinable": determinable,
        "explanation": "\n".join(explanation_lines)
    }




def medication_safety_instruction(dialogue, drug_info, llm_models):
    prompt = f"""
You are a medication safety evaluator in a clinical consultation.

You will be given:
- A structured doctor-patient dialogue
- A list of medications actually recommended by the clinician
- The clinician's explanations for why these medications were chosen
- A gold-standard reference list of correct medications for this case
- Patient's diagnosis
- Clinician's diagnostic reasoning process

Evaluation Task:
Evaluate every medication one by one:

1. For each prescribed medication, evaluate:
   - reference_drug_deviation
   - drug_diagnosis_mismatch
   - unsafe_drug_use
   - incorrect_or_missing_explanation

2. For each medication in the reference list, evaluate:
   - missing_important_drugs

--------------------------------------------------
GENERAL RULES
--------------------------------------------------
1. You MUST evaluate EVERY prescribed medication and EVERY reference medication.
2. Every medication must be explicitly mentioned in the explanations under the appropriate category.
3. Count only items that are actually errors.
4. Medications marked as "Accepted", "Safe for use", "Explanation appropriate", or "Not clinically necessary" must still appear in the explanations but do NOT count toward error totals.
5. Chronic or long-term management medications are considered clinically necessary unless clearly substituted or contraindicated.
6. Acute-use medications are evaluated based on the patient’s presenting symptoms and acute clinical context.

--------------------------------------------------
ERROR CATEGORIES
--------------------------------------------------

Error Category Definitions with Explanations:
- Every prescribed medication and every reference medication must be explicitly mentioned in the explanations under the appropriate category.
- You must list all medications under every relevant category.
  Example:
  - Each prescribed drug appears under reference_drug_deviation, drug_diagnosis_mismatch, unsafe_drug_use, and incorrect_or_missing_explanation.
  - Each reference drug appears under missing_important_drugs.
- Count only items that are actually errors:
  - **reference_drug_deviation**: count only medications marked "Incorrect: not in reference list and not a substitute."
  - **drug_diagnosis_mismatch**: count only medications marked "No clear relevance to the diagnostic reasoning process."
  - **unsafe_drug_use**: count only medications marked "Unsafe drug use: explanation of risk."
  - **missing_important_drugs**: count only medications marked "Missing and not substituted. Counted as error."
  - **incorrect_or_missing_explanation**: count only medications marked "Incorrect or missing explanation."
- Medications marked as "Accepted", "Safe for use", "Explanation appropriate", or "Not clinically necessary" must still appear in the explanations but do NOT count toward error totals.
- Chronic or long-term management medications (e.g., antihypertensives, anticoagulants, GERD therapy, psychiatric medications) are considered clinically necessary unless clearly substituted or contraindicated.
- Acute-use medications are evaluated based on the patient’s presenting symptoms and acute clinical context.

Error Category Definitions:

Error Category Definitions with Explanations:

1. **reference_drug_deviation**: Drug is not in reference list and is not a reasonable or safe alternative.

2. **drug_diagnosis_mismatch**: Drug has no clear therapeutic relevance to the diagnosis or diagnostic reasoning.

3. **unsafe_drug_use**: Drug is contraindicated or potentially harmful given known patient factors.

4. **missing_important_drugs**: Key reference medication is entirely omitted.

5. **incorrect_or_missing_explanation**: Drug rationale is unclear, inaccurate, or missing, regardless of drug appropriateness.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
You MUST output a valid JSON object and NOTHING ELSE.

Use the following structure:

{{
  "reference_drug_deviation": <integer>,
  "drug_diagnosis_mismatch": <integer>,
  "unsafe_drug_use": <integer>,
  "missing_important_drugs": <integer>,
  "incorrect_or_missing_explanation": <integer>,
  "explanation": {{
    "reference_drug_deviation": [
      "Drug A: Accepted: in the reference list.",
      "Drug B: Incorrect: not in reference list and not a substitute."
    ],
    "drug_diagnosis_mismatch": [
      "Drug A: Therapeutically aligned with the diagnostic reasoning",
      "Drug B: No clear relevance to the clinician’s diagnostic reasoning process."
    ],
    "missing_important_drugs_and_coverage": [
      "Drug W: Present via safe substitute."
      "Drug X: Present in recommendation.",
      "Drug Y: Missing and not substituted. Counted as error."
      "Drug Z: Missing but not clinically necessary. Not counted as error."
    ],
    "unsafe_drug_use": [
      "Drug A: Safe for use.",
      "Drug B: Unsafe drug use: risk of renal impairment."
    ],
    "incorrect_or_missing_explanation": [
      "Drug A: Explanation appropriate.",
      "Drug B: Incorrect or missing explanation."
    ]
  }}
}}

Output ONLY this JSON object and nothing else.

Now evaluate the following:

Dialogue:
{json.dumps(dialogue, indent=2, ensure_ascii=False)}

Clinician Prescribed Medications:
{json.dumps(drug_info["clinician_prescribed_medications"], indent=2, ensure_ascii=False)}

Drug Explanations:
{json.dumps(drug_info["drug_reasoning"], indent=2, ensure_ascii=False)}

Reference Medication List:
{json.dumps(drug_info["correct_answer"], indent=2, ensure_ascii=False)}
Patient's diagnosis:
{json.dumps(drug_info["correct_diagnoses"], indent=2, ensure_ascii=False)}
- Clinician's diagnostic reasoning process:
{json.dumps(drug_info["diagnosis_reasoning"], indent=2, ensure_ascii=False)}
"""
    response = run_model(messages=[{"role": "user", "content": prompt}], model=llm_models)
    response = json.loads(response)
    return response


def calculate_medication_safety_score(full_output):
    deduction_rules = {
        "reference_drug_deviation": {"deduct": 0.3, "max": 1.0},
        "drug_diagnosis_mismatch": {"deduct": 0.5, "max": 1.0},
        "unsafe_drug_use": {"deduct": 0.5, "max": 1.0},
        "missing_important_drugs": {"deduct": 0.3, "max": 1.0},
        "incorrect_or_missing_explanation": {"deduct": 0.3, "max": 1.0}
    }

    total_score = 5.0
    explanation_lines = []
    explanations = full_output.get("explanation", {})
    issue_counts = {}

    for error_type, rule in deduction_rules.items():
        count = full_output.get(error_type, 0)
        deduction = min(count * rule["deduct"], rule["max"])
        total_score -= deduction
        issue_counts[error_type] = count

        exp_key = explanation_key_mapping.get(error_type)
        lines = explanations.get(exp_key, [])
        if count > 0 and exp_key in explanations:
            explanation_lines.append(
                f"{error_type}: {count} occurrence(s)\n  " +
                "\n  ".join(lines)
            )
        else:
            explanation_lines.append(
                f"{error_type}: 0 occurrence(s)\n  " +
                "\n  ".join(lines)
            )

    result = {
        "score": max(0.0, round(total_score, 2)),  # Minimum score is 1
        "explanation": "\n".join(explanation_lines),
        "details": issue_counts
    }
    return result
