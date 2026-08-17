# AI 评估结果人工评价界面 - 修复版本
from nicegui import ui, app
import os, json
from typing import Dict, Any, List

# ----------------------------
# 配置：数据源文件 & 维度定义
# ----------------------------
DATA_DIR = "../data/dialogue"
AI_EVAL_DIR = "../data/eval"
AI_REVIEW_BASE_DIR = "../data/ai_review"

ai_evaluation_dimensions = [
    ("Medical interview questioning skills", "dynamic_steps"),
    ("Information Coverage", "dynamic_items"),
    ("Handling of ambiguous patient responses", "dynamic_steps"),
    ("Ethical and professional conduct", "dynamic_steps"),
    ("Clarity and transparency of clinical explanations", "dynamic_steps"),
    ("information integration", "dynamic_categories"),
    ("Diagnostic reasoning", "complex_structure"),
    ("Medication safety and justification", "text_explanation")
]

DIM_KEY_MAP = {
    0: "questioning_skills_review",
    1: "self_awareness_review",
    2: "robustness_review",
    3: "ethics_review",
    4: "explainability_review",
    5: "information_summary_review",
    6: "diagnostic_reasoning_review",
    7: "medication_safety_review"
}

# ----------------------------
# 模型显示编号映射（页面上显示数字，隐藏真实模型名）
# ----------------------------
MODEL_DISPLAY_OPTIONS = {"1": "gpt4o", "2": "claude4", "3": "qwen3", "4": "medgemma"}

def get_model_display_name(model: str) -> str:
    """将模型内部名称转为显示用的数字编号"""
    for display, real in MODEL_DISPLAY_OPTIONS.items():
        if real == model:
            return display
    return model

# ----------------------------
# 多用户状态管理
# ----------------------------
def get_current_model() -> str:
    try:
        if not hasattr(app.storage.user, 'current_model'):
            app.storage.user.current_model = 'qwen3'
        return app.storage.user.current_model
    except:
        return 'gpt4o'

def set_current_model(model: str):
    try:
        app.storage.user.current_model = model
    except:
        pass

def get_current_user_id() -> str:
    try:
        if not hasattr(app.storage.user, 'current_user_id'):
            app.storage.user.current_user_id = ''
        return app.storage.user.current_user_id
    except:
        return ''

def set_current_user_id(user_id: str):
    try:
        app.storage.user.current_user_id = user_id
    except:
        pass#
# ----------------------------
# 工具函数
# ----------------------------
def load_records_for_model(model_name: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, f"{model_name}_match30.jsonl")
    if not os.path.exists(path):
        print(f"load_records_for_model error with {path}")
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
    return records

def load_ai_evaluation(model: str, subject_id: str) -> Dict[str, Any]:
    possible_files = [f"{model}.jsonl"]
    
    for filename in possible_files:
        path = os.path.join(AI_EVAL_DIR, filename)
        if not os.path.exists(path):
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    sid = data.get("SubjectID")
                    if sid is not None and str(sid) == subject_id:
                        return data
        except Exception as e:
            print(f"load_ai_evaluation error with {filename}: {e}")
            continue
    
    return {}

def get_review_dir_path(model: str, user_id: str) -> str:
    return os.path.join(AI_REVIEW_BASE_DIR, model, str(user_id))

def check_review_dir_exists(model: str, user_id: str) -> bool:
    review_dir = get_review_dir_path(model, user_id)
    return os.path.isdir(review_dir)

def save_ai_review_result(model: str, user_id: str, subject_id: str, payload: dict) -> bool:
    if not check_review_dir_exists(model, user_id):
        return False
    
    review_dir = get_review_dir_path(model, user_id)
    path = os.path.join(review_dir, f"{subject_id}.jsonl")
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def load_user_reviews(model: str, user_id: str) -> Dict[str, Dict]:
    cache = {}
    review_dir = get_review_dir_path(model, user_id)
    if not os.path.isdir(review_dir):
        return cache
    
    try:
        for fname in os.listdir(review_dir):
            if not fname.endswith('.jsonl'):
                continue
            subject_id = fname[:-6]
            try:
                with open(os.path.join(review_dir, fname), 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        cache[subject_id] = json.loads(lines[-1])
            except Exception:
                pass
    except Exception:
        pass
    return cache# 
##----------------------------
# 顶部栏
# ----------------------------
def top_bar():
    with ui.row().style(
            "align-items:center; gap:12px; padding:10px; background:white; "
            "position:sticky; top:0; z-index:999; border-bottom:1px solid #eee;"
    ):
        ui.label("模型：").style("font-weight:600; font-size:15px;")
        model_select = ui.select(
            options=MODEL_DISPLAY_OPTIONS,
            value=get_current_model(),
        ).props('outlined dense').style("width:150px;")

        def on_model_change(e):
            set_current_model(model_select.value)
            ui.navigate.to("/")

        model_select.on("update:model-value", on_model_change)

        ui.label("Reviewer ID:").style("font-weight:600; font-size:15px;")
        user_id_input = ui.input(
            placeholder="请输入评价者ID",
            value=get_current_user_id(),
        ).style("width:120px;")

        def save_user_id_on_enter(e):
            set_current_user_id(user_id_input.value)
            ui.notify(f"Reviewer ID 已设置为: {user_id_input.value}", color="positive")
            ui.navigate.to("/")

        def save_user_id_on_change(e):
            set_current_user_id(user_id_input.value)

        user_id_input.on("keydown.enter", save_user_id_on_enter)
        user_id_input.on("blur", save_user_id_on_change)
        
        ui.button("确认", on_click=save_user_id_on_enter).style("height:36px; margin-left:5px;")
        ui.button("刷新病例列表", on_click=lambda e: ui.navigate.to("/")).style("height:36px; line-height:36px;")

# ----------------------------
# 主页面
# ----------------------------
@ui.page('/')
def main_page():
    top_bar()

    model = get_current_model()
    user_id = get_current_user_id()
    records = load_records_for_model(model)
    user_reviews = load_user_reviews(model, user_id) if user_id else {}

    ui.markdown(f"### AI 评估结果评价 | 模型：`{get_model_display_name(model)}` | Reviewer ID: `{user_id or '未输入'}` | 已评价: `{len(user_reviews)}`")

    with ui.card().style("padding:12px; margin-top:6px;"):
        ui.label(f"共 {len(records)} 条病例").style("font-weight:600")

        with ui.row().classes('w-full').props("q-col-gutter-y-sm q-col-gutter-x-md").style("margin-top:8px;"):
            for rec in records:
                pid = rec.get("id", {}) or {}
                sid = str(pid.get("SubjectID", "unknown"))

                status = "未评价"
                status_color = "#f3f4f6"
                completed = 0

                if sid in user_reviews:
                    r = user_reviews[sid]
                    completed = sum(1 for k in DIM_KEY_MAP.values() if k in r and r[k] is not None)
                    if completed >= 8:
                        status = "已完成"
                        status_color = "#d1fae5"
                    elif completed > 0:
                        status = f"进行中 ({completed}/8)"
                        status_color = "#fff7ed"

                with ui.element('div').classes('col-12 col-md-4 col-lg-4'):
                    with ui.row().style(
                            "align-items:center; gap:12px; padding:8px; border-radius:8px; "
                            "border: 1px solid #e5e7eb;"
                    ).classes('w-full'):

                        ui.label(sid).style("width:130px; font-weight:600")

                        info = rec.get("info", {}).get("initial_info", {})
                        brief = f"AGE:{info.get('AGE', '-')} | GENDER:{info.get('GENDER', '-')}"
                        ui.label(brief).style("flex:1; color:#374151")

                        ui.html(
                            f"<div style='background:{status_color}; padding:6px 10px; "
                            "border-radius:6px; min-width:130px; text-align:center;'>"
                            f"{status}</div>"
                        )

                        def on_open_case(e=None, subject_id=sid, completed_count=completed):
                            uid = get_current_user_id()
                            m = get_current_model()

                            if not uid:
                                ui.notify("⚠️ 请先在顶部输入 Reviewer ID，再开始评价。", color="negative")
                                return

                            ai_eval = load_ai_evaluation(m, subject_id)
                            if not ai_eval:
                                ui.notify("⚠️ 该病例没有 AI 评估结果，无法进行评价。", color="negative")
                                return

                            next_step = min(completed_count + 1, 8)
                            ui.navigate.to(f"/review/{subject_id}/{next_step}")

                        btn_label = (
                            "开始评价" if status == "未评价"
                            else "查看/继续" if status.startswith("进行中")
                            else "查看结果"
                        )

                        ui.button(btn_label, on_click=on_open_case).style("height:34px;")#
# ----------------------------
# AI 评估结果评价页面
# ----------------------------
@ui.page('/review/{subject_id}/{step}')
def review_page(subject_id: str, step: int):
    model = get_current_model()
    user_id = get_current_user_id() or ""
    subject_id = str(subject_id)
    step = int(step)

    if not user_id:
        ui.notify("⚠️ 未输入 Reviewer ID，请先在主页面顶部输入 Reviewer ID 后再开始评价。", color="negative")
        ui.navigate.to("/")
        return

    records = load_records_for_model(model)
    record = None
    for r in records:
        pid = r.get("id", {}) or {}
        if str(pid.get("SubjectID")) == subject_id:
            record = r
            break

    if record is None:
        ui.notify(f"⚠️ 未找到 SubjectID={subject_id} 在模型 {get_model_display_name(model)} 中。")
        ui.navigate.to("/")
        return

    ai_eval = load_ai_evaluation(model, subject_id)
    if not ai_eval:
        ui.notify("⚠️ 该病例没有 AI 评估结果，无法进行评价。", color="negative")
        ui.navigate.to("/")
        return

    # 加载已有的评价数据
    existing_reviews = load_user_reviews(model, user_id)
    existing_review = existing_reviews.get(subject_id, {})
    
    # 获取当前维度的已保存评价
    keyname = DIM_KEY_MAP.get(step - 1)
    saved_review = existing_review.get(keyname, {}) if keyname else {}
    saved_step_reviews = saved_review.get("step_by_step_reviews", {}) if isinstance(saved_review, dict) else {}
    
    def get_saved_rating(item_key):
        """获取已保存的评分，如果没有则返回默认值"""
        if item_key in saved_step_reviews:
            rating = saved_step_reviews[item_key].get("rating", 5)
            return f"{rating} - " + ["非常不满意", "不满意", "一般", "满意", "非常满意"][rating - 1]
        return "5 - 非常满意"

    # 页面标题 + 导航操作
    with ui.card().style("padding:12px; margin:12px; width:100%; position:relative;"):
        ui.label(f"AI 评估结果评价 - 病例 {subject_id} - 第 {step}/8 项").style("font-weight:700; font-size:18px;")

        with ui.row().style("gap:10px; flex-wrap:wrap; position:absolute; top:12px; right:12px;"):
            if step > 1:
                ui.button("← 上一项", on_click=lambda e, prev_step=step - 1: ui.navigate.to(f"/review/{subject_id}/{prev_step}")).props("outline").style("min-width:130px;")

            ui.button("提交评价", on_click=lambda: submit_review()).props("color=primary").style("min-width:130px;")
            ui.button("返回病例列表", on_click=lambda e: ui.navigate.to("/")).props("outline").style("min-width:130px;")

    # 患者基本信息
    with ui.card().style("padding:10px; margin:12px;"):
        info = record.get("info", {}).get("initial_info", {})
        pid = record.get("id", {}) or {}
        patient_text = (
            f"**SubjectID:** {pid.get('SubjectID', 'Unknown')} | "
            f"**AdmissionID:** {pid.get('AdmissionID', 'Unknown')} | "
            f"**AGE:** {info.get('AGE', 'Unknown')} | "
            f"**GENDER:** {info.get('GENDER', 'Unknown')} | "
            f"**RELIGION:** {info.get('RELIGION', 'Unknown')} | "
            f"**MARITAL_STATUS:** {info.get('MARITAL_STATUS', 'Unknown')} | "
            f"**ETHNICITY:** {info.get('ETHNICITY', 'Unknown')}"
        )
        ui.markdown(patient_text)

    # 获取维度信息
    dim_idx = step - 1
    dim_entry = ai_evaluation_dimensions[dim_idx]
    dim_name = dim_entry[0]
    keyname = DIM_KEY_MAP.get(dim_idx)
    
    original_keyname = {
        "questioning_skills_review": "questioning_skills_score",
        "self_awareness_review": "self_awareness_score", 
        "robustness_review": "robustness_score",
        "ethics_review": "ethics_score",
        "explainability_review": "explainability_score",
        "information_summary_review": "information_summary_score",
        "diagnostic_reasoning_review": "diagnostic_reasoning_score",
        "medication_safety_review": "medication_safety_score"
    }.get(keyname, keyname)
    
    ai_dim_data = ai_eval.get(original_keyname, {})
    dialogue = record.get("interactive_system", {}).get("conversation_history", [])
    
    # 主体区域
    with ui.card().style("padding:12px; margin:12px;"):
        ui.markdown(f"### {dim_name} - 逐步评价")
        
        controls = {}
        
        if dim_idx in [0, 2, 3, 4]:  # robustness_score, ethics_score, explainability_score - 步骤式评估
            ui.markdown("左侧显示对话，右侧显示对应的 AI 评估和您的评分")
            
            explanation_data = ai_dim_data.get("explanation", {})
            
            if not isinstance(explanation_data, dict) or not explanation_data:
                ui.label("该维度没有详细的步骤评估数据").style("color:gray;")
                return
            
            # 创建滚动容器包含所有步骤
            with ui.scroll_area().style("height:600px; width:100%; min-width:3000px;"):
                # 为每个步骤创建统一格式的显示框
                for step_key, step_data in explanation_data.items():
                    if not isinstance(step_data, dict):
                        continue
                        
                    # 提取步骤编号
                    step_num = None
                    if step_key.startswith("Step "):
                        try:
                            step_num = int(step_key.split(" ")[1])
                        except:
                            pass
                    
                    # 统一的步骤卡片格式
                    with ui.card().style("padding:20px; margin-bottom:20px; border:2px solid #e5e7eb; border-radius:12px;"):
                        
                        # 1. 步骤标题（固定格式）
                        with ui.row().style("width:100%; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6;"):
                            ui.label(f"{step_key}").style("font-weight:700; font-size:18px; color:#1e40af;")
                        
                        # 2. 三栏布局（固定宽度比例）
                        with ui.row().style("gap:16px; width:100%;"):
                            
                            # 左栏：对话内容（固定宽度）
                            with ui.column().style("width:400px; min-height:200px;"):
                                # 标题
                                ui.label("📞 对话内容").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#059669;")
                                
                                # 对话框（固定高度）
                                with ui.card().style("padding:12px; min-height:150px; background:#f0fdf4; border-left:4px solid #10b981; width:100%;"):
                                    if step_num and dialogue:
                                        # 查找对应步骤的对话
                                        dialogue_found = False
                                        i = 0
                                        current_step = 1
                                        n = len(dialogue)
                                        
                                        while i < n and current_step <= step_num:
                                            if current_step == step_num:
                                                # 医生问题
                                                if i < n:
                                                    turn_doctor = dialogue[i]
                                                    text_d = turn_doctor.get("content", "") or turn_doctor.get("text", "") or ""
                                                    ui.label("医生：").style("font-weight:600; color:#1d4ed8; margin-bottom:4px;")
                                                    ui.label(text_d).style("white-space:pre-wrap; margin-bottom:8px; font-size:13px; line-height:1.4;")
                                                
                                                # 患者回答
                                                if i + 1 < n:
                                                    turn_patient = dialogue[i + 1]
                                                    text_p = turn_patient.get("content", "") or turn_patient.get("text", "") or ""
                                                    ui.label("患者：").style("font-weight:600; color:#047857; margin-bottom:4px;")
                                                    ui.label(text_p).style("white-space:pre-wrap; font-size:13px; line-height:1.4;")
                                                
                                                dialogue_found = True
                                                break
                                            
                                            current_step += 1
                                            i += 2
                                        
                                        if not dialogue_found:
                                            ui.label("未找到对应的对话内容").style("color:gray; font-style:italic; text-align:center; margin-top:60px;")
                                    else:
                                        ui.label("无对话数据").style("color:gray; font-style:italic; text-align:center; margin-top:60px;")
                            
                            # 中栏：AI 评估结果（固定宽度）
                            with ui.column().style("width:400px; min-height:200px;"):
                                # 标题
                                ui.label("🤖 AI 评估结果").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#dc2626;")
                                
                                # AI 评估框（固定高度）
                                with ui.card().style("padding:12px; min-height:150px; background:#fef2f2; border-left:4px solid #ef4444; width:100%;"):
                                    assessment = step_data.get("assessment", "N/A")
                                    reason = step_data.get("reason", "N/A")
                                    error_type = step_data.get("error_type", "")
                                    
                                    ui.label(f"评估结果: {assessment}").style("font-weight:600; margin-bottom:8px; color:#dc2626; font-size:14px;")
                                    
                                    if error_type:
                                        ui.label(f"错误类型: {error_type}").style("color:#dc2626; font-size:13px; margin-bottom:8px; font-weight:600;")
                                    
                                    ui.label("AI 理由:").style("font-weight:600; margin-bottom:6px; color:#374151; font-size:13px;")
                                    ui.label(reason).style("white-space:pre-wrap; font-size:12px; line-height:1.5; color:#6b7280;")
                            
                            # 右栏：人工评分（固定宽度）
                            with ui.column().style("width:600px; min-height:200px;"):
                                # 标题
                                ui.label("👤 您的评价").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#7c3aed;")
                                
                                # 评分框（固定高度）
                                with ui.card().style("padding:12px; min-height:150px; background:#faf5ff; border-left:4px solid #8b5cf6; width:100%;"):
                                    
                                    # 评分输入（单选按钮）
                                    ui.label("评分 (1-5):").style("font-weight:600; margin-bottom:8px; color:#7c3aed; font-size:14px;")
                                    
                                    # 单选按钮组
                                    rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                    rating_input = ui.radio(
                                        options=rating_options,
                                        value=get_saved_rating(step_key)
                                    ).props('color=purple inline').style("margin-bottom:12px; font-size:13px;")
                                    
                                    controls[step_key] = {
                                        "rating": rating_input,
                                        "reason": None
                                    }
        
        elif dim_idx == 1:  # Self-awareness of clinical information needs
            ui.markdown("左侧显示完整对话，右侧显示各维度评估结果和评分")
            
            explanation_data = ai_dim_data.get("explanation", {})
            
            if not isinstance(explanation_data, dict) or not explanation_data:
                ui.label("该维度没有详细的评估数据").style("color:gray;")
                return
            
            # 创建主容器 - 两个独立滚动框
            with ui.card().style("padding:20px; margin:12px; border:2px solid #e5e7eb; border-radius:12px; height:600px; max-width:100%;"):
                
                # 标题
                with ui.row().style("margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6;"):
                    ui.label("临床信息需求自我意识评估").style("font-weight:700; font-size:18px; color:#1e40af;")
                
                # 两栏布局：左栏显示对话，右栏显示评估
                with ui.row().style("gap:16px; height:520px; display:flex; flex-wrap:nowrap;"):
                    
                    # 左栏：完整对话内容（独立滚动）
                    with ui.column().style("width:600px; height:100%;"):
                        ui.label("📞 完整对话内容").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#059669;")
                        
                        # 对话内容独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            with ui.card().style("padding:12px; background:#f0fdf4; border-left:4px solid #10b981; width:100%;"):
                                if dialogue:
                                    step_num = 1
                                    for i, turn in enumerate(dialogue):
                                        role = "医生" if i % 2 == 0 else "患者"
                                        color = "#1d4ed8" if i % 2 == 0 else "#047857"
                                        text = turn.get("content", "") or turn.get("text", "") or ""
                                        
                                        # 在每个医生问题前显示步骤标识
                                        if i % 2 == 0:  # 医生的轮次
                                            ui.label(f"Step {step_num}").style("font-weight:700; color:#6366f1; margin-top:12px; margin-bottom:6px; font-size:14px; background:#e0e7ff; padding:4px 8px; border-radius:4px; display:inline-block;")
                                            step_num += 1
                                        
                                        ui.label(f"{role}：").style(f"font-weight:600; color:{color}; margin-bottom:4px; margin-top:4px;")
                                        ui.label(text).style("white-space:pre-wrap; margin-bottom:8px; font-size:13px; line-height:1.4;")
                                else:
                                    ui.label("无对话数据").style("color:gray; font-style:italic; text-align:center; margin-top:60px;")
                    
                    # 右栏：各维度评估结果和评分（独立滚动）
                    with ui.column().style("width:650px; height:100%;"):
                        ui.label("🤖 AI 各维度评估结果").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#dc2626;")
                        
                        # 评估结果独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            assessment_dimensions = [
                                "chief_complaint_details", "past_medical_history", "allergies",
                                "relevant_symptom_review", "family_history", "social_history",
                                "lifestyle_factors", "broader_review_of_systems", 
                                "mental_status", "specific_risk_factors"
                            ]
                            
                            # 为每个维度创建评估和评分区域
                            for dim_key in assessment_dimensions:
                                if dim_key in explanation_data:
                                    dim_assessment = explanation_data[dim_key]
                                    
                                    with ui.card().style("padding:12px; margin-bottom:12px; background:#fef2f2; border-left:4px solid #ef4444; width:100%;"):
                                        # 维度名称
                                        ui.label(dim_key.replace("_", " ").title()).style("font-weight:600; margin-bottom:6px; color:#dc2626;")
                                        
                                        # AI评估结果
                                        if isinstance(dim_assessment, dict):
                                            status = dim_assessment.get("status", "N/A")
                                            explanation = dim_assessment.get("explanation", "N/A")
                                            
                                            # 显示状态
                                            if status == "problematic":
                                                ui.label(f"状态: {status}").style("font-size:13px; margin-bottom:4px; color:#dc2626; font-weight:600;")
                                            else:
                                                ui.label(f"状态: {status}").style("font-size:13px; margin-bottom:4px;")
                                            
                                            # 显示详细说明
                                            ui.label("详细说明:").style("font-size:12px; font-weight:600; color:#374151; margin-bottom:4px;")
                                            ui.label(explanation).style("font-size:12px; color:#6b7280; margin-bottom:8px; white-space:pre-wrap; line-height:1.4;")
                                            
                                            # 如果有其他详细信息，也显示出来
                                            for key, value in dim_assessment.items():
                                                if key not in ["status", "explanation"] and value:
                                                    ui.label(f"{key.replace('_', ' ').title()}: {value}").style("font-size:11px; color:#6b7280; margin-bottom:2px;")
                                        else:
                                            ui.label(f"评估: {dim_assessment}").style("font-size:13px; margin-bottom:8px;")
                                        
                                        # 评分区域
                                        ui.label("您的评分:").style("font-weight:600; font-size:13px; color:#7c3aed; margin-bottom:4px;")
                                        rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                        rating_input = ui.radio(
                                            options=rating_options,
                                            value=get_saved_rating(dim_key)
                                        ).props('color=purple inline').style("margin-bottom:8px; font-size:12px;")
                                        
                                        controls[dim_key] = {
                                            "rating": rating_input,
                                            "reason": None
                                        }
        
        elif dim_idx == 5:  # Clinical information integration
            ui.markdown("左侧显示完整对话，右侧显示采集信息、AI评估和人工评价")
            
            explanation_data = ai_dim_data.get("explanation", {})
            
            if not isinstance(explanation_data, dict) or not explanation_data:
                ui.label("该维度没有详细的评估数据").style("color:gray;")
                return
            
            # 获取Category Summaries数据
            answer = record.get("interactive_system", {}).get("final_answer", {})
            category_summaries = answer.get("category_summaries", {}) or {}
            print(category_summaries)
            
            # 创建主容器 - 三栏布局
            with ui.card().style("padding:20px; margin:12px; border:2px solid #e5e7eb; border-radius:12px; height:600px; max-width:100%;"):
                
                # 标题
                with ui.row().style("margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6;"):
                    ui.label("临床信息整合评估").style("font-weight:700; font-size:18px; color:#1e40af;")
                
                # 两栏布局：左栏显示完整对话，右栏显示所有采集信息和评估
                with ui.row().style("gap:16px; height:520px; display:flex; flex-wrap:nowrap;"):
                    
                    # 左栏：完整对话内容（独立滚动）
                    with ui.column().style("width:600px; height:100%;"):
                        ui.label("📞 完整对话内容").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#059669;")
                        
                        # 对话内容独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            with ui.card().style("padding:12px; background:#f0fdf4; border-left:4px solid #10b981; width:100%;"):
                                if dialogue:
                                    step_num = 1
                                    for i, turn in enumerate(dialogue):
                                        role = "医生" if i % 2 == 0 else "患者"
                                        color = "#1d4ed8" if i % 2 == 0 else "#047857"
                                        text = turn.get("content", "") or turn.get("text", "") or ""
                                        
                                        # 在每个医生问题前显示步骤标识
                                        if i % 2 == 0:  # 医生的轮次
                                            ui.label(f"Step {step_num}").style("font-weight:700; color:#6366f1; margin-top:12px; margin-bottom:6px; font-size:14px; background:#e0e7ff; padding:4px 8px; border-radius:4px; display:inline-block;")
                                            step_num += 1
                                        
                                        ui.label(f"{role}：").style(f"font-weight:600; color:{color}; margin-bottom:4px; margin-top:4px;")
                                        ui.label(text).style("white-space:pre-wrap; margin-bottom:8px; font-size:13px; line-height:1.4;")
                                else:
                                    ui.label("无对话数据").style("color:gray; font-style:italic; text-align:center; margin-top:60px;")
                        
                    # 右栏：所有采集信息、AI评估和人工评价（独立滚动）
                    with ui.column().style("width:1000px; height:100%;"):
                        ui.label("� 病史采集信息与评估").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#dc2626;")
                        
                        # 病史采集评估独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            
                            # 为每个Category Summary项目创建紧凑的评估卡片
                            for item_key, collected_info in category_summaries.items():
                                if item_key in explanation_data:  # 只显示有AI评估的项目
                                    item_assessment = explanation_data[item_key]
                                    
                                    with ui.card().style("padding:12px; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; width:100%;"):
                                        
                                        # 项目标题
                                        ui.label(f"{item_key.replace('_', ' ').title()}").style("font-weight:600; font-size:14px; color:#1e40af; margin-bottom:8px;")
                                        
                                        # 三栏紧凑布局：采集信息 | AI评估 | 人工评价
                                        with ui.row().style("gap:8px; width:100%; display:flex;"):
                                            
                                            # 采集信息（紧凑）
                                            with ui.column().style("flex:0 0 300px; min-height:120px;"):
                                                ui.label("📋 采集信息").style("font-size:12px; font-weight:600; color:#059669; margin-bottom:4px;")
                                                with ui.card().style("padding:8px; min-height:80px; background:#f0fdf4; border-left:3px solid #10b981; width:100%;"):
                                                    ui.label(collected_info).style("white-space:pre-wrap; font-size:14px; line-height:1.3; color:#374151;")
                                                # AI评估（详细）
                                            with ui.column().style("flex:0 0 320px; min-height:120px;"):
                                                ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#dc2626; margin-bottom:4px;")
                                                with ui.card().style("padding:8px; min-height:80px; background:#fef2f2; border-left:3px solid #ef4444; width:100%;"):
                                                    if isinstance(item_assessment, dict):
                                                        # 显示覆盖的步骤
                                                        covered_steps = item_assessment.get("covered_steps", [])
                                                        if covered_steps:
                                                            ui.label(f"✅ 覆盖步骤: {covered_steps}").style("font-size:12px; margin-bottom:3px; color:#059669; font-weight:600;")
                                                        
                                                        # 显示缺失的步骤
                                                        missing_steps = item_assessment.get("missing_steps", [])
                                                        if missing_steps:
                                                            ui.label(f"❌ 缺失步骤: {missing_steps}").style("font-size:12px; margin-bottom:3px; color:#dc2626; font-weight:600;")
                                                        
                                                        # 显示虚构项目（详细）
                                                        fabricated_items = item_assessment.get("fabricated_items", [])
                                                        if fabricated_items:
                                                            ui.label(f"⚠️ 虚构项目:").style("font-size:12px; margin-bottom:2px; color:#f59e0b; font-weight:600;")
                                                            for fab_item in fabricated_items:
                                                                ui.label(f"  • {fab_item}").style("font-size:12px; margin-bottom:2px; margin-left:6px; color:#6b7280; white-space:pre-wrap;")
                                                        
                                                        # 显示错误信息（详细）
                                                        errors = item_assessment.get("errors", [])
                                                        if errors:
                                                            ui.label(f"🚫 错误详情:").style("font-size:12px; margin-bottom:2px; color:#dc2626; font-weight:600;")
                                                            for error in errors:
                                                                if isinstance(error, dict):
                                                                    error_type = error.get("type", "N/A")
                                                                    error_desc = error.get("description", "N/A")
                                                                    ui.label(f"  • {error_type}").style("font-size:12px; margin-bottom:1px; color:#dc2626; font-weight:600;")
                                                                    ui.label(f"    {error_desc}").style("font-size:12px; margin-bottom:3px; margin-left:10px; color:#6b7280; white-space:pre-wrap;")
                                                        
                                                        # 显示备注
                                                        notes = item_assessment.get("notes", "")
                                                        if notes:
                                                            ui.label(f"📝 备注:").style("font-size:12px; margin-top:3px; margin-bottom:1px; color:#6b7280; font-weight:600;")
                                                            ui.label(notes).style("font-size:12px; color:#6b7280; font-style:italic; white-space:pre-wrap;")
                                                        
                                                        # 如果没有问题，显示正常状态
                                                        if not errors and not missing_steps and not fabricated_items:
                                                            ui.label("✅ 评估正常").style("font-size:12px; color:#059669; font-weight:600;")
                                                    else:
                                                        ui.label(f"{item_assessment}").style("font-size:12px;")
                                            
                                            # 人工评价（紧凑）
                                            with ui.column().style("flex:0 0 300px; min-height:120px;"):
                                                ui.label("👤 您的评价").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#7c3aed;")
                                                
                                                # 评分框（固定高度）
                                                with ui.card().style("padding:12px; min-height:150px; background:#faf5ff; border-left:4px solid #8b5cf6; width:100%;"):
                                                    
                                                    # 评分输入（单选按钮）
                                                    ui.label("评分 (1-5):").style("font-weight:600; margin-bottom:8px; color:#7c3aed; font-size:12px;")
                                                    
                                                    # 单选按钮组
                                                    rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                                    rating_input = ui.radio(
                                                        options=rating_options,
                                                        value=get_saved_rating(item_key)
                                                    ).props('color=purple inline').style("margin-bottom:12px; font-size:13px;")
                                                    
                                                    controls[item_key] = {
                                                        "rating": rating_input,
                                                        "reason": None
                                                    }
        
        elif dim_idx == 6:  # Clinical diagnostic reasoning
            ui.markdown("左侧显示对话和诊断内容，右侧显示推理错误评估")
            
            # 获取推理错误数据
            reasoning_errors = ai_dim_data.get("reasoning_errors", {})
            print(reasoning_errors)
            diagnosis_accuracy = ai_dim_data.get("diagnosis_accuracy", {})
            explanation_text = ai_dim_data.get("explanation", "")
            determinable = ai_dim_data.get("determinable", False)
            
            if not reasoning_errors and not diagnosis_accuracy:
                ui.label("该维度没有详细的评估数据").style("color:gray;")
                return
            
            # 获取诊断和相关信息
            answer = record.get("interactive_system", {}).get("final_answer", {})
            info = record.get("info", {})
            
            diagnosis_content = answer.get("diagnosis", "") or ""
            diagnosis_reasoning = answer.get("diagnosis_reasoning", "") or ""
            correct_answer = info.get("correct_answer", "") or ""
            correct_diagnoses = info.get("correct_diagnoses", []) or []
            
            # 创建主容器
            with ui.card().style("padding:20px; margin:12px; border:2px solid #e5e7eb; border-radius:12px; height:600px; display:inline-block;"):
                
                # 标题
                with ui.row().style("margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6;"):
                    ui.label("临床诊断推理评估").style("font-weight:700; font-size:18px; color:#1e40af;")
                
                # 三栏布局：左栏显示对话，中栏显示诊断信息，右栏显示错误选项评估
                with ui.row().style("gap:16px; height:520px; display:flex; flex-wrap:nowrap;"):
                    
                    # 左栏：对话内容（独立滚动）
                    with ui.column().style("width:600px; height:100%;"):
                        ui.label("📞 对话内容").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#059669;")
                        
                        # 对话内容独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            with ui.card().style("padding:12px; background:#f0fdf4; border-left:4px solid #10b981; width:100%;"):
                                if dialogue:
                                    step_num = 1
                                    for i, turn in enumerate(dialogue):
                                        role = "医生" if i % 2 == 0 else "患者"
                                        color = "#1d4ed8" if i % 2 == 0 else "#047857"
                                        text = turn.get("content", "") or turn.get("text", "") or ""
                                        
                                        # 在每个医生问题前显示步骤标识
                                        if i % 2 == 0:  # 医生的轮次
                                            ui.label(f"Step {step_num}").style("font-weight:700; color:#6366f1; margin-top:8px; margin-bottom:4px; font-size:12px; background:#e0e7ff; padding:2px 6px; border-radius:4px; display:inline-block;")
                                            step_num += 1
                                        
                                        ui.label(f"{role}：").style(f"font-weight:600; color:{color}; margin-bottom:2px; margin-top:2px; font-size:12px;")
                                        ui.label(text).style("white-space:pre-wrap; margin-bottom:6px; font-size:11px; line-height:1.3;")
                                else:
                                    ui.label("无对话数据").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                    
                    # 中栏：诊断信息（独立滚动）
                    with ui.column().style("width:600px; height:100%;"):
                        ui.label("🩺 诊断信息").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#d97706;")
                        
                        # 诊断信息独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            
                            # 正确诊断
                            with ui.card().style("padding:12px; margin-bottom:12px; background:#ecfdf5; border-left:4px solid #10b981; width:100%;"):
                                ui.label("正确诊断 (Correct Diagnoses):").style("font-weight:600; margin-bottom:8px; color:#059669;")
                                if correct_diagnoses:
                                    for i, diagnosis in enumerate(correct_diagnoses, 1):
                                        ui.label(f"{i}. {diagnosis}").style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151; margin-bottom:4px;")
                                else:
                                    ui.label("无正确诊断").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                            
                            # AI诊断内容
                            with ui.card().style("padding:12px; margin-bottom:12px; background:#fef3c7; border-left:4px solid #f59e0b; width:100%;"):
                                ui.label("AI诊断 (AI Diagnosis):").style("font-weight:600; margin-bottom:8px; color:#d97706;")
                                if diagnosis_content:
                                    ui.label(diagnosis_content).style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151;")
                                else:
                                    ui.label("无AI诊断").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                            
                            # AI诊断推理
                            with ui.card().style("padding:12px; background:#fef3c7; border-left:4px solid #f59e0b; width:100%;"):
                                ui.label("AI诊断推理 (AI Diagnosis Reasoning):").style("font-weight:600; margin-bottom:8px; color:#d97706;")
                                if diagnosis_reasoning:
                                    ui.label(diagnosis_reasoning).style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151;")
                                else:
                                    ui.label("无诊断推理").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                    
                    # 右栏：错误选项评估（独立滚动）
                    with ui.column().style("width:500px; height:100%;"):
                        ui.label("🔍 诊断推理错误评估").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#dc2626;")
                        
                        # 错误选项评估独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            
                            # 首先显示诊断准确性评估
                            if diagnosis_accuracy:
                                with ui.card().style("padding:12px; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; width:100%;"):
                                    ui.label("诊断准确性 (Diagnosis Accuracy)").style("font-weight:600; font-size:14px; color:#1e40af; margin-bottom:8px;")
                                    
                                    with ui.row().style("gap:12px; width:100%; display:flex;"):
                                        # AI评估
                                        with ui.column().style("flex:1; min-height:100px;"):
                                            ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#dc2626; margin-bottom:4px;")
                                            with ui.card().style("padding:8px; min-height:80px; background:#fef2f2; border-left:3px solid #ef4444; width:100%;"):
                                                match_level = diagnosis_accuracy.get("match_level", "N/A")
                                                deduction = diagnosis_accuracy.get("deduction", 0)
                                                comment = diagnosis_accuracy.get("comment", "N/A")
                                                
                                                ui.label(f"匹配程度: {match_level}").style("font-size:11px; margin-bottom:4px; color:#dc2626; font-weight:600;")
                                                ui.label(f"扣分: {deduction}").style("font-size:11px; margin-bottom:4px; color:#dc2626; font-weight:600;")
                                                ui.label(f"评论: {comment}").style("font-size:11px; color:#6b7280; white-space:pre-wrap;")
                                        
                                        # 人工评价
                                        with ui.column().style("flex:0 0 200px; min-height:100px;"):
                                            ui.label("👤 您的评价").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:4px;")
                                            with ui.card().style("padding:8px; min-height:80px; background:#faf5ff; border-left:3px solid #8b5cf6; width:100%;"):
                                                ui.label("评分 (1-5):").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:6px;")
                                                rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                                rating_input = ui.radio(
                                                    options=rating_options,
                                                    value=get_saved_rating("diagnosis_accuracy")
                                                ).props('color=purple inline').style("margin-bottom:6px; font-size:13px;")
                                                
                                                controls["diagnosis_accuracy"] = {
                                                    "rating": rating_input,
                                                    "reason": None
                                                }
                            
                            # 然后显示推理错误评估 - 显示所有类型，包括没有错误的
                            for error_type, error_count in reasoning_errors.items():
                                    with ui.card().style("padding:12px; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; width:100%;"):
                                        
                                        # 错误类型标题
                                        error_title = error_type.replace("_", " ").title()
                                        if error_count > 0:
                                            ui.label(f"{error_title} (发现 {error_count} 个)").style("font-weight:600; font-size:14px; color:#dc2626; margin-bottom:8px;")
                                        else:
                                            ui.label(f"{error_title} (无错误)").style("font-weight:600; font-size:14px; color:#059669; margin-bottom:8px;")
                                        
                                        # 两栏布局：AI评估 | 人工评价
                                        with ui.row().style("gap:12px; width:100%; display:flex;"):
                                            
                                            # AI评估 - 从explanation中提取相关信息
                                            with ui.column().style("flex:1; min-height:100px;"):
                                                if error_count > 0:
                                                    ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#dc2626; margin-bottom:4px;")
                                                    card_style = "padding:8px; min-height:80px; background:#fef2f2; border-left:3px solid #ef4444; width:100%;"
                                                    count_color = "#dc2626"
                                                else:
                                                    ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#059669; margin-bottom:4px;")
                                                    card_style = "padding:8px; min-height:80px; background:#f0fdf4; border-left:3px solid #10b981; width:100%;"
                                                    count_color = "#059669"
                                                
                                                with ui.card().style(card_style):
                                                    ui.label(f"错误次数: {error_count}").style(f"font-size:12px; margin-bottom:4px; color:{count_color}; font-weight:600;")
                                                    
                                                    # 从explanation文本中提取对应的详细说明
                                                    if explanation_text:
                                                        # 解析explanation文本，提取对应错误类型的详细说明
                                                        explanation_parts = explanation_text.split('\n')
                                                        error_explanation = ""
                                                        
                                                        # 查找对应错误类型的说明
                                                        for i, line in enumerate(explanation_parts):
                                                            if error_type in line and "occurrence(s)" in line:
                                                                # 找到错误类型行，收集后续的说明
                                                                error_explanation = line.strip()
                                                                # 收集后续的详细说明行
                                                                j = i + 1
                                                                while j < len(explanation_parts) and explanation_parts[j].strip():
                                                                    next_line = explanation_parts[j].strip()
                                                                    # 如果下一行是另一个错误类型，停止
                                                                    if any(err_type in next_line for err_type in reasoning_errors.keys()) and "occurrence(s)" in next_line:
                                                                        break
                                                                    if next_line and not next_line.startswith("diagnosis_"):
                                                                        error_explanation += "\n" + next_line
                                                                    j += 1
                                                                break
                                                        
                                                        if error_explanation:
                                                            ui.label("详细说明:").style("font-size:13px; color:#374151; font-weight:600; margin-bottom:4px;")
                                                            ui.label(error_explanation).style("font-size:12px; color:#6b7280; white-space:pre-wrap; line-height:1.4;")
                                                        elif error_count == 0:
                                                            # 如果没有错误，显示原因
                                                            ui.label("无此类错误").style("font-size:13px; color:#059669; font-weight:600; margin-bottom:4px;")
                                                            ui.label("AI评估认为在此方面没有发现推理错误").style("font-size:12px; color:#6b7280; font-style:italic;")
                                                        else:
                                                            ui.label("详细说明请查看完整解释").style("font-size:12px; color:#6b7280; font-style:italic;")
                                                    else:
                                                        ui.label("无详细说明").style("font-size:12px; color:#6b7280; font-style:italic;")
                                            
                                            # 人工评价
                                            with ui.column().style("flex:0 0 200px; min-height:100px;"):
                                                ui.label("👤 您的评价").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:4px;")
                                                
                                                with ui.card().style("padding:8px; min-height:80px; background:#faf5ff; border-left:3px solid #8b5cf6; width:100%;"):
                                                    
                                                    # 评分输入
                                                    ui.label("评分 (1-5):").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:6px;")
                                                    
                                                    # 单选按钮组
                                                    rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                                    rating_input = ui.radio(
                                                        options=rating_options,
                                                        value=get_saved_rating(error_type)
                                                    ).props('color=purple inline').style("margin-bottom:6px; font-size:13px;")
                                                    
                                                    controls[error_type] = {
                                                        "rating": rating_input,
                                                        "reason": None
                                                    }
                            
                            # 添加determinable评估项目
                            with ui.card().style("padding:12px; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; width:100%;"):
                                
                                # 特殊标题显示
                                if determinable:
                                    ui.label("Correct diagnoses can be made from the conversation (可确定)").style("font-weight:600; font-size:14px; color:#059669; margin-bottom:8px;")
                                else:
                                    ui.label("Correct diagnoses can be made from the conversation (不可确定)").style("font-weight:600; font-size:14px; color:#dc2626; margin-bottom:8px;")
                                
                                # 两栏布局：AI评估 | 人工评价
                                with ui.row().style("gap:12px; width:100%; display:flex;"):
                                    
                                    # AI评估 - 提取diagnosis_certain_explanations
                                    with ui.column().style("flex:1; min-height:100px;"):
                                        if determinable:
                                            ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#059669; margin-bottom:4px;")
                                            card_style = "padding:8px; min-height:80px; background:#f0fdf4; border-left:3px solid #10b981; width:100%;"
                                            status_color = "#059669"
                                        else:
                                            ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#dc2626; margin-bottom:4px;")
                                            card_style = "padding:8px; min-height:80px; background:#fef2f2; border-left:3px solid #ef4444; width:100%;"
                                            status_color = "#dc2626"
                                        
                                        with ui.card().style(card_style):
                                            ui.label(f"可确定性: {'是' if determinable else '否'}").style(f"font-size:12px; margin-bottom:4px; color:{status_color}; font-weight:600;")
                                            
                                            # 从explanation中提取diagnosis_certain_explanations
                                            if explanation_text:
                                                diagnosis_certain_explanation = ""
                                                lines = explanation_text.split("\n")
                                                for i, line in enumerate(lines):
                                                    if line.strip().startswith("diagnosis_certain_explanations:"):
                                                        # 提取冒号后的内容
                                                        content = line.split(":", 1)[1].strip() if ":" in line else ""
                                                        # 如果内容跨多行，继续收集后续行
                                                        collected = [content] if content else []
                                                        for j in range(i + 1, len(lines)):
                                                            next_line = lines[j].strip()
                                                            # 如果是空行或新的标题，停止
                                                            if not next_line or (":" in next_line and 
                                                                               not lines[j].startswith("  ") and 
                                                                               next_line.endswith(":")):
                                                                break
                                                            collected.append(next_line)
                                                        diagnosis_certain_explanation = " ".join(collected).strip()
                                                        break
                                                
                                                if diagnosis_certain_explanation:
                                                    ui.label("详细说明:").style("font-size:13px; color:#374151; font-weight:600; margin-bottom:4px;")
                                                    ui.label(diagnosis_certain_explanation).style("font-size:12px; color:#6b7280; white-space:pre-wrap; line-height:1.4;")
                                                else:
                                                    ui.label("无详细说明").style("font-size:12px; color:#6b7280; font-style:italic;")
                                            else:
                                                ui.label("无详细说明").style("font-size:12px; color:#6b7280; font-style:italic;")
                                    
                                    # 人工评价
                                    with ui.column().style("flex:0 0 200px; min-height:100px;"):
                                        ui.label("👤 您的评价").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:4px;")
                                        
                                        with ui.card().style("padding:8px; min-height:80px; background:#faf5ff; border-left:3px solid #8b5cf6; width:100%;"):
                                            
                                            # 评分输入
                                            ui.label("评分 (1-5):").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:6px;")
                                            
                                            # 单选按钮组
                                            rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                            rating_input = ui.radio(
                                                options=rating_options,
                                                value=get_saved_rating("determinable")
                                            ).props('color=purple inline').style("margin-bottom:6px; font-size:13px;")
                                            
                                            controls["determinable"] = {
                                                "rating": rating_input,
                                                "reason": None
                                            }

        
        elif dim_idx == 7:  # Medication safety and justification
            ui.markdown("左侧显示对话，中间显示药物信息，右侧显示AI评估和人工评价")
            
            # 获取药物相关信息
            answer = record.get("interactive_system", {}).get("final_answer", {})
            info = record.get("info", {})
            
            correct_answer = info.get("correct_answer", "") or ""
            question = info.get("question", "") or ""
            recommended_drugs = answer.get("recommended_drugs", []) or []
            drug_reasoning = answer.get("drug_reasoning", {}) or {}
            
            explanation_text = ai_dim_data.get("explanation", "")
            
            # 创建主容器 - 三栏布局
            with ui.card().style("padding:20px; margin:12px; border:2px solid #e5e7eb; border-radius:12px; height:600px; width:1900px; max-width:100%;"):
                
                # 标题
                with ui.row().style("margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6;"):
                    ui.label("药物安全性与合理性评估").style("font-weight:700; font-size:16px; color:#1e40af;")
                
                # 三栏布局
                with ui.row().style("gap:16px; height:520px; display:flex; flex-wrap:nowrap;"):
                    
                    # 左栏：对话内容（独立滚动）
                    with ui.column().style("width:600px; height:100%;"):
                        ui.label("📞 对话内容").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#059669;")
                        
                        # 对话内容独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            with ui.card().style("padding:12px; background:#f0fdf4; border-left:4px solid #10b981; width:100%;"):
                                if dialogue:
                                    step_num = 1
                                    for i, turn in enumerate(dialogue):
                                        role = "医生" if i % 2 == 0 else "患者"
                                        color = "#1d4ed8" if i % 2 == 0 else "#047857"
                                        text = turn.get("content", "") or turn.get("text", "") or ""
                                        
                                        if i % 2 == 0:
                                            ui.label(f"Step {step_num}").style("font-weight:700; color:#6366f1; margin-top:8px; margin-bottom:4px; font-size:12px; background:#e0e7ff; padding:2px 6px; border-radius:4px; display:inline-block;")
                                            step_num += 1
                                        
                                        ui.label(f"{role}：").style(f"font-weight:600; color:{color}; margin-bottom:2px; margin-top:2px; font-size:12px;")
                                        ui.label(text).style("white-space:pre-wrap; margin-bottom:6px; font-size:11px; line-height:1.3;")
                                else:
                                    ui.label("无对话数据").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                    
                    # 中栏：药物信息（独立滚动）
                    with ui.column().style("width:400px; height:100%;"):
                        ui.label("💊 药物信息").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#d97706;")
                        
                        # 药物信息独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            
                            # 问题
                            # if question:
                            #     with ui.card().style("padding:12px; margin-bottom:12px; background:#ede9fe; border-left:4px solid #8b5cf6; width:100%;"):
                            #         ui.label("问题 (Question):").style("font-weight:600; margin-bottom:8px; color:#7c3aed;")
                            #         ui.label(question).style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151;")
                            
                            # 正确答案
                            with ui.card().style("padding:12px; margin-bottom:12px; background:#ecfdf5; border-left:4px solid #10b981; width:100%;"):
                                ui.label("正确答案 (Correct Answer):").style("font-weight:600; margin-bottom:8px; color:#059669;")
                                if correct_answer:
                                    ui.label(correct_answer).style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151;")
                                else:
                                    ui.label("无正确答案").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                            
                            # AI推荐药物
                            with ui.card().style("padding:12px; margin-bottom:12px; background:#fef3c7; border-left:4px solid #f59e0b; width:100%;"):
                                ui.label("AI推荐药物 (AI Recommended Drugs):").style("font-weight:600; margin-bottom:8px; color:#d97706;")
                                if recommended_drugs:
                                    for i, drug in enumerate(recommended_drugs, 1):
                                        ui.label(f"{i}. {drug}").style("white-space:pre-wrap; font-size:12px; line-height:1.4; color:#374151; margin-bottom:4px;")
                                else:
                                    ui.label("无推荐药物").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                            
                            # AI药物推理
                            with ui.card().style("padding:12px; background:#fef3c7; border-left:4px solid #f59e0b; width:100%;"):
                                ui.label("AI药物推理 (AI Drug Reasoning):").style("font-weight:600; margin-bottom:8px; color:#d97706;")
                                if drug_reasoning:
                                    for drug, reasoning in drug_reasoning.items():
                                        ui.label(f"• {drug}:").style("font-weight:600; font-size:11px; color:#d97706; margin-top:6px; margin-bottom:2px;")
                                        ui.label(reasoning).style("white-space:pre-wrap; font-size:11px; line-height:1.4; color:#374151; margin-bottom:6px; margin-left:12px;")
                                else:
                                    ui.label("无药物推理").style("color:gray; font-style:italic; text-align:center; margin-top:30px;")
                    
                    # 右栏：药物安全评估（独立滚动）
                    with ui.column().style("width:800px; height:100%;"):
                        ui.label("🔍 药物安全性评估").style("font-weight:600; font-size:14px; margin-bottom:8px; color:#dc2626;")
                        
                        # 药物安全评估独立滚动区域
                        with ui.scroll_area().style("height:480px; width:100%;"):
                            
                            # 定义药物安全评估的五个维度（新格式）
                            safety_dimensions = [
                                ("reference_drug_deviation", "Reference Drug Deviation"),
                                ("drug_diagnosis_mismatch", "Drug Diagnosis Mismatch"),
                                ("unsafe_drug_use", "Unsafe Drug Use"),
                                ("missing_important_drugs", "Missing Important Drugs"),
                                ("incorrect_or_missing_explanation", "Incorrect or Missing Explanation")
                            ]
                            
                            # 从文本格式的explanation中解析各维度的详细信息
                            explanation_text = ai_dim_data.get("explanation", "")
                            dimension_details = {}
                            
                            if explanation_text:
                                lines = explanation_text.split('\n')
                                current_dimension = None
                                current_content = []
                                
                                for line in lines:
                                    line = line.strip()
                                    # 检查是否是维度标题行
                                    for dim_key, _ in safety_dimensions:
                                        if line.startswith(f"{dim_key}:") and "occurrence(s)" in line:
                                            # 保存前一个维度的内容
                                            if current_dimension:
                                                dimension_details[current_dimension] = '\n'.join(current_content)
                                            # 开始新维度
                                            current_dimension = dim_key
                                            current_content = [line]
                                            break
                                    else:
                                        # 不是维度标题，添加到当前维度内容
                                        if current_dimension and line:
                                            current_content.append(line)
                                
                                # 保存最后一个维度
                                if current_dimension:
                                    dimension_details[current_dimension] = '\n'.join(current_content)
                            
                            # 为每个药物安全维度创建评估卡片
                            for dim_key, dim_title in safety_dimensions:
                                # 从维度详情中提取错误次数和详细内容
                                error_count = 0
                                detail_text = dimension_details.get(dim_key, "")
                                detail_lines = []
                                
                                if detail_text:
                                    lines = detail_text.split('\n')
                                    # 从第一行提取错误次数
                                    if lines and "occurrence(s)" in lines[0]:
                                        try:
                                            error_count = int(lines[0].split(':')[1].split('occurrence(s)')[0].strip())
                                        except:
                                            error_count = 0
                                    
                                    # 收集详细说明行（跳过第一行的错误次数）
                                    detail_lines = [line.strip() for line in lines[1:] if line.strip()]
                                
                                with ui.card().style("padding:12px; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; width:100%;"):
                                    
                                    # 维度标题
                                    if error_count > 0:
                                        ui.label(f"{dim_title} (发现 {error_count} 个)").style("font-weight:600; font-size:14px; color:#dc2626; margin-bottom:8px;")
                                    else:
                                        ui.label(f"{dim_title} (无错误)").style("font-weight:600; font-size:14px; color:#059669; margin-bottom:8px;")
                                    
                                    # 两栏布局：AI评估 | 人工评价
                                    with ui.row().style("gap:12px; width:100%; display:flex;"):
                                        
                                        # AI评估
                                        with ui.column().style("flex:1; min-height:100px;"):
                                            if error_count > 0:
                                                ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#dc2626; margin-bottom:4px;")
                                                card_style = "padding:8px; min-height:80px; background:#fef2f2; border-left:3px solid #ef4444; width:100%;"
                                                count_color = "#dc2626"
                                            else:
                                                ui.label("🤖 AI评估").style("font-size:12px; font-weight:600; color:#059669; margin-bottom:4px;")
                                                card_style = "padding:8px; min-height:80px; background:#f0fdf4; border-left:3px solid #10b981; width:100%;"
                                                count_color = "#059669"
                                            
                                            with ui.card().style(card_style):
                                                ui.label(f"错误次数: {error_count}").style(f"font-size:14px; margin-bottom:4px; color:{count_color}; font-weight:600;")
                                                
                                                # 显示详细说明
                                                if detail_lines:
                                                    ui.label("详细说明:").style("font-size:13px; color:#374151; font-weight:600; margin-bottom:4px;")
                                                    for detail in detail_lines:
                                                        ui.label(f"• {detail}").style("font-size:12px; color:#6b7280; white-space:pre-wrap; line-height:1.4; margin-bottom:2px;")
                                                elif error_count == 0:
                                                    ui.label("无此类错误").style("font-size:13px; color:#059669; font-weight:600; margin-bottom:4px;")
                                                    ui.label("AI评估认为在此方面没有发现问题").style("font-size:12px; color:#6b7280; font-style:italic;")
                                                else:
                                                    ui.label("无详细说明").style("font-size:12px; color:#6b7280; font-style:italic;")
                                        
                                        # 人工评价
                                        with ui.column().style("flex:0 0 400px; min-height:100px;"):
                                            ui.label("👤 您的评价").style("font-size:12px; font-weight:600; color:#7c3aed; margin-bottom:4px;")
                                            
                                            with ui.card().style("padding:8px; min-height:80px; background:#faf5ff; border-left:3px solid #8b5cf6; width:100%;"):
                                                ui.label("评分 (1-5):").style("font-size:14px; font-weight:600; color:#7c3aed; margin-bottom:6px;")
                                                
                                                rating_options = ["1 - 非常不满意", "2 - 不满意", "3 - 一般", "4 - 满意", "5 - 非常满意"]
                                                rating_input = ui.radio(
                                                    options=rating_options,
                                                    value=get_saved_rating(dim_key)
                                                ).props('color=purple inline').style("margin-bottom:6px; font-size:13px;")
                                                
                                                controls[dim_key] = {
                                                    "rating": rating_input,
                                                    "reason": None
                                                }
        
        else:
            # 其他维度显示提示信息
            ui.label("该维度暂时使用原有的评价方式，请返回使用其他评价界面。").style("color:gray; font-style:italic;")    
    def submit_review():
        # 收集评价数据
        review_data = {}
        total_ratings = []
        
        for item, ctrl in controls.items():
            try:
                # 从字符串中提取评分数字 (如 "3 - 一般" -> 3)
                rating_str = ctrl["rating"].value or "5 - 非常满意"
                rating = int(rating_str.split(" - ")[0])
                reason = "" if ctrl["reason"] is None else (ctrl["reason"].value or "").strip()
                review_data[item] = {
                    "rating": rating,
                    "reason": reason
                }
                total_ratings.append(rating)
            except (ValueError, TypeError):
                ui.notify(f"⚠️ 请为 '{item}' 输入有效的评分（1-5）", color="negative")
                return
        
        if not total_ratings:
            ui.notify("⚠️ 请至少完成一项评价", color="negative")
            return
        
        # 计算平均分
        avg_score = sum(total_ratings) / len(total_ratings)
        
        # 生成保存结构
        result_piece = {
            "SubjectID": subject_id,
            "AdmissionID": record.get("id", {}).get("AdmissionID"),
            keyname: {
                "average_score": round(avg_score, 2),
                "total_items": len(total_ratings),
                "step_by_step_reviews": review_data,
                "ai_evaluation": ai_dim_data,
                "dimension_type": dim_name,
                "review_timestamp": __import__('datetime').datetime.now().isoformat()
            }
        }
        
        # 合并已有评价
        existing_reviews = load_user_reviews(model, user_id)
        existing = existing_reviews.get(subject_id, {})
        
        merged = dict(existing)
        merged["SubjectID"] = subject_id
        merged["AdmissionID"] = record.get("id", {}).get("AdmissionID")
        merged.update(result_piece)
        
        # 保存评价结果
        if save_ai_review_result(model, user_id, subject_id, merged):
            ui.notify(f"第 {step} 项评价已保存（平均分: {avg_score:.2f}）。", color="positive")
            
            # 自动跳下一步或到汇总
            next_step = step + 1
            if next_step <= 8:
                ui.navigate.to(f"/review/{subject_id}/{next_step}")
            else:
                ui.navigate.to("/")
        else:
            review_dir = get_review_dir_path(model, user_id)
            ui.notify(f"⚠️ 保存失败！评价目录不存在: {review_dir}", color="negative")
            ui.notify("请联系管理员创建评价目录后再进行评价。", color="warning")

# ----------------------------
# 启动
# ----------------------------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=11240, title="AI Evaluation Review System - Fixed", storage_secret="ai_review_secret_key_2024")