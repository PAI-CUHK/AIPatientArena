#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional

from nicegui import ui, app
import nicegui.run

try:
    nicegui.run.process_pool = None
    nicegui.run.setup = lambda: None
except Exception as e:
    logging.warning(f"Failed to patch nicegui.run: {e}")

def add_screenshot_scripts():
    ui.add_head_html('<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>')
    ui.add_head_html('<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>')
    js_code = '''
    function captureAndDownload(format) {
        const element = document.body;
        
        html2canvas(element, {
            scale: 2,
            useCORS: true,
            allowTaint: true,
            ignoreElements: function(element) {
                return element.classList.contains('no-capture');
            }
        }).then(canvas => {
            const link = document.createElement('a');
            if (format === 'pdf') {
                const imgData = canvas.toDataURL('image/png');
                const pdf = new jsPDF('p', 'mm', 'a4');
                const imgWidth = 210;
                const pageHeight = 297;
                const imgHeight = (canvas.height * imgWidth) / canvas.width;
                let heightLeft = imgHeight;
                let position = 0;
                
                pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                heightLeft -= pageHeight;
                
                while (heightLeft >= 0) {
                    position = heightLeft - imgHeight;
                    pdf.addPage();
                    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                    heightLeft -= pageHeight;
                }
                
                link.download = 'evaluation_page.pdf';
                link.href = pdf.output('bloburl');
            } else {
                link.download = 'evaluation_page.png';
                link.href = canvas.toDataURL('image/png');
            }
            link.click();
        }).catch(err => {
            console.error('截图失败:', err);
            alert('截图失败，请重试');
        });
    }
    '''
    ui.add_head_html(f'<script>{js_code}</script>')

DATA_DIR = "/media/disk1/niujiahui/aipatient_eval/data/dialogue"
DEFAULT_MODEL = "gpt4o"
AUTO_EVAL_DIR = "/media/disk1/niujiahui/aipatient_eval/data/eval"

dimensions_info = [
    ("问诊提问技巧", [
        "logical jumps", "repetitive questions", "unclear questions",
        "suggestive questions", "confrontational questions"
    ]),
    ("问诊过程中临床信息覆盖程度", [
        "major omission", "minor omission"
    ]),
    ("处理病人模棱两可答复的能力", [
        "unclear expression handling", "uncertain answer handling"
    ]),
    ("道德和职业操守", [
        "disrespectful tone", "bias or assumptions", "privacy violation", "lack of empathy"
    ]),
    ("临床解释的清晰度和透明度", [
        "major clarity failure", "minor clarity failure", "lack of context for question"
    ]),
    ("在书写病历摘要过程中的临床信息整合能力", [
        "major omission", "minor omission", "major fabrication", "minor fabrication"
    ]),
    ("临床诊断推理能力", [
        "matching degree", "information omission", "incorrect assumption",
        "flawed reasoning process", "diagnosis reasoning mismatch", "determinable"
    ], {"matching degree": ["exact match", "partial match", "minorly incorrect", "majorly incorrect"],
        "determinable": ["true", "false"]}),
    ("药物安全性和合理性", [
        "reference drug deviation", "drug diagnosis mismatch", "unsafe drug use",
        "incorrect or missing explanation", "missing important drugs"
    ])
]

DIM_KEY_MAP = {
    0: "questioning_skills_score",
    1: "self_awareness_score",
    2: "robustness_score",
    3: "ethics_score",
    4: "explainability_score",
    5: "information_summary_score",
    6: "diagnostic_reasoning_score",
    7: "medication_safety_score"
}

# ----------------------------
# 模型显示编号映射（页面上显示数字，隐藏真实模型名）
# ----------------------------
MODEL_DISPLAY_OPTIONS = {"1": "gpt4o", "2": "claude46", "3": "qwen3", "4": "medgemma"}

def get_model_display_name(model: str) -> str:
    """将模型内部名称转为显示用的数字编号"""
    for display, real in MODEL_DISPLAY_OPTIONS.items():
        if real == model:
            return display
    return model

RECORD_CACHE: Dict[str, List[Dict[str, Any]]] = {}
RECORD_CACHE_TTL: Dict[str, float] = {}
AUTO_EVAL_CACHE: Dict[str, Dict[str, Any]] = {}
AUTO_EVAL_CACHE_TTL: Dict[str, float] = {}
CACHE_TTL_SECONDS = 300

def get_cache_key(model_name: str) -> str:
    return f"records_{model_name}"

async def async_load_records_for_model(model_name: str) -> List[Dict[str, Any]]:
    cache_key = get_cache_key(model_name)
    now = time.time()
    if cache_key in RECORD_CACHE and (now - RECORD_CACHE_TTL.get(cache_key, 0) < CACHE_TTL_SECONDS):
        return RECORD_CACHE[cache_key]
    path = os.path.join(DATA_DIR, f"{model_name}_match30.jsonl")
    if not os.path.exists(path):
        RECORD_CACHE[cache_key] = []
        RECORD_CACHE_TTL[cache_key] = now
        return []
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except Exception:
                    continue
    except Exception as e:
        print(f"async_load_records_for_model error: {e}")
    RECORD_CACHE[cache_key] = records
    RECORD_CACHE_TTL[cache_key] = now
    return records

def invalidate_records_cache(model_name: str):
    cache_key = get_cache_key(model_name)
    RECORD_CACHE.pop(cache_key, None)
    RECORD_CACHE_TTL.pop(cache_key, None)

async def async_user_scores_cache_load(model: str, user_id: str) -> Dict[str, Dict]:
    cache = {}
    base_dir = "/media/disk1/niujiahui/aipatient_eval/data/manual_eval"
    user_dir = os.path.join(base_dir, str(model), str(user_id))
    if not os.path.isdir(user_dir):
        return cache
    try:
        for fname in os.listdir(user_dir):
            if not fname.endswith('.jsonl'):
                continue
            subject_id = fname[:-6]
            try:
                with open(os.path.join(user_dir, fname), 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        cache[subject_id] = json.loads(lines[-1])
            except Exception:
                pass
    except Exception as e:
        print(f"async_user_scores_cache_load error: {e}")
    return cache

def ensure_user_dir(model: str, user_id: str) -> Optional[str]:
    base_dir = "/media/disk1/niujiahui/aipatient_eval/data/manual_eval"
    model_dir = os.path.join(base_dir, str(model))
    user_dir = os.path.join(model_dir, str(user_id))
    if os.path.isdir(user_dir):
        return user_dir
    return None

async def async_append_result_jsonl(model: str, user_id: str, subject_id: str, payload: dict):
    user_dir = ensure_user_dir(model, user_id)
    if user_dir is None:
        raise ValueError(f"用户文件夹不存在: {model}/{user_id}")
    path = os.path.join(user_dir, f"{subject_id}.jsonl")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

async def async_overwrite_result_json(model: str, user_id: str, subject_id: str, payload: dict):
    user_dir = ensure_user_dir(model, user_id)
    if user_dir is None:
        raise ValueError(f"用户文件夹不存在: {model}/{user_id}")
    path = os.path.join(user_dir, f"{subject_id}.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

async def async_load_auto_eval(model: str, subject_id: str) -> Dict[str, Any]:
    cache_key = f"auto_eval_{model}_{subject_id}"
    now = time.time()
    if cache_key in AUTO_EVAL_CACHE and (now - AUTO_EVAL_CACHE_TTL.get(cache_key, 0) < CACHE_TTL_SECONDS):
        return AUTO_EVAL_CACHE[cache_key]
    model = str(model)
    subject_id = str(subject_id)
    filename = f"{model}_0108.jsonl"
    path = os.path.join(AUTO_EVAL_DIR, filename)
    if not os.path.exists(path):
        AUTO_EVAL_CACHE[cache_key] = {}
        AUTO_EVAL_CACHE_TTL[cache_key] = now
        return {}
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
                    AUTO_EVAL_CACHE[cache_key] = data
                    AUTO_EVAL_CACHE_TTL[cache_key] = now
                    return data
    except Exception as e:
        print(f"async_load_auto_eval error: {e}")
    AUTO_EVAL_CACHE[cache_key] = {}
    AUTO_EVAL_CACHE_TTL[cache_key] = now
    return {}

def convert_new_eval_format(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    if "scores" in data:
        return data
    result = {"scores": {}}
    score_keys = [
        "questioning_skills_score",
        "self_awareness_score",
        "robustness_score",
        "ethics_score",
        "explainability_score",
        "information_summary_score",
        "diagnostic_reasoning_score",
        "medication_safety_score"
    ]
    for key in score_keys:
        if key in data:
            dim_data = data[key]
            if isinstance(dim_data, dict):
                converted = dict(dim_data)
                if "details" not in converted:
                    converted["details"] = {}
                result["scores"][key] = converted
    return result


def top_bar():
    session = app.storage.user
    if 'current_model' not in session:
        session['current_model'] = "qwen3"
    if 'current_user_id' not in session:
        session['current_user_id'] = ""
    with ui.row().style(
            "align-items:center; gap:12px; padding:10px; background:white; "
            "position:sticky; top:0; z-index:999; border-bottom:1px solid #eee;"
    ):
        ui.label("模型：").style("font-weight:600; font-size:15px;")
        model_select = ui.select(
            options=MODEL_DISPLAY_OPTIONS,
            value=session['current_model'],
        ).props('outlined dense').style("width:150px;")

        def on_model_change(e):
            session['current_model'] = model_select.value
            invalidate_records_cache(session['current_model'])
            ui.navigate.to("/")

        model_select.on("update:model-value", on_model_change)

        ui.label("User ID:").style("font-weight:600; font-size:15px;")
        user_id_input = ui.input(
            placeholder="请输入用户ID",
            value=session['current_user_id'],
        ).style("width:120px;")

        def save_user_id(e):
            session['current_user_id'] = user_id_input.value
            ui.navigate.to("/")

        user_id_input.on("keydown.enter", save_user_id)
        user_id_input.on("change", save_user_id)

        ui.button(
            "刷新病例列表",
            on_click=lambda e: ui.navigate.to("/")
        ).style("height:36px; line-height:36px;")


@ui.page('/')
async def main_page():
    top_bar()
    session = app.storage.user
    model = session.get('current_model', "qwen3")
    user_id = session.get('current_user_id', "")
    records = await async_load_records_for_model(model)
    user_cache = await async_user_scores_cache_load(model, user_id) if user_id else {}

    ui.markdown(f"### 模型：`{get_model_display_name(model)}` | User ID: `{user_id or '未输入'}`")

    if user_id:
        user_dir = ensure_user_dir(model, user_id)
        if user_dir is None:
            ui.label(f"⚠️ 用户文件夹不存在：{get_model_display_name(model)}/{user_id}，请使用已存在的 User ID。").style("color:red; font-weight:600;")

    with ui.card().style("padding:12px; margin-top:6px;"):
        ui.label(f"共 {len(records)} 条病例").style("font-weight:600")
        with ui.row().classes('w-full').props("q-col-gutter-y-sm q-col-gutter-x-md").style("margin-top:8px;"):
            for rec in records:
                pid = rec.get("id", {}) or {}
                sid = str(pid.get("SubjectID", "unknown"))

                status = "未开始"
                status_color = "#f3f4f6"
                completed = 0

                if sid in user_cache:
                    r = user_cache[sid]
                    completed = sum(
                        1 for k in DIM_KEY_MAP.values()
                        if k in r and r[k] is not None
                    )
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

                        async def on_open_case(e=None, subject_id=sid, completed_count=completed):
                            uid = session.get('current_user_id', "")
                            m = session.get('current_model', "qwen3")
                            if not uid:
                                ui.notify("⚠️ 请先在顶部输入 User ID，再开始评估。", color="negative")
                                return
                            user_dir = ensure_user_dir(m, uid)
                            if user_dir is None:
                                ui.notify(f"⚠️ 用户文件夹不存在：{m}/{uid}，请使用已存在的 User ID。", color="negative")
                                return
                            cache_now = await async_user_scores_cache_load(m, uid) if uid else {}
                            current_completed = 0
                            if subject_id in cache_now:
                                r = cache_now[subject_id]
                                current_completed = sum(
                                    1 for k in DIM_KEY_MAP.values()
                                    if k in r and r[k] is not None
                                )
                            next_step = min(current_completed + 1, 8)
                            ui.navigate.to(f"/evaluate/{subject_id}/{next_step}")

                        btn_label = (
                            "开始评估" if status == "未开始"
                            else "查看/继续" if status.startswith("进行中")
                            else "查看结果"
                        )
                        ui.button(btn_label, on_click=on_open_case).style("height:34px;")


async def get_saved_progress_for_case(model: str, user_id: str, subject_id: str) -> Dict:
    subject_id_str = str(subject_id)
    cache = await async_user_scores_cache_load(model, user_id) if user_id else {}
    if subject_id_str in cache:
        return cache[subject_id_str]
    return {}


def _get_detail_value(details: Dict[str, Any], item_name: str):
    if item_name in details:
        return details[item_name]
    alt1 = item_name.replace(" ", "_")
    if alt1 in details:
        return details[alt1]
    alt2 = item_name.replace("_", " ")
    if alt2 in details:
        return details[alt2]
    alias_map = {
        "suggestive questions": "leading_questions",
    }
    key = alias_map.get(item_name)
    if key and key in details:
        return details[key]
    return None


def _get_diagnostic_reasoning_value(auto_dim_data: Dict[str, Any], item_name: str):
    if item_name == "matching degree":
        diagnosis_accuracy = auto_dim_data.get("diagnosis_accuracy", {}) or {}
        match_level = diagnosis_accuracy.get("match_level", "")
        if match_level:
            return match_level.replace("_", " ")
        return None
    elif item_name == "determinable":
        determinable = auto_dim_data.get("determinable")
        if determinable is not None:
            return "true" if determinable else "false"
        return None
    else:
        reasoning_errors = auto_dim_data.get("reasoning_errors", {}) or {}
        item_key = item_name.replace(" ", "_")
        if item_key in reasoning_errors:
            return reasoning_errors[item_key]
        if item_name in reasoning_errors:
            return reasoning_errors[item_name]
        return None


ITEM_PENALTY = {
    "questioning_skills_score": {
        "logical jumps": {"per_error": 0.5, "max": 1},
        "repetitive questions": {"per_error": 0.5, "max": 1},
        "unclear questions": {"per_error": 0.5, "max": 1.5},
        "suggestive questions": {"per_error": 0.25, "max": 0.5},
        "confrontational questions": {"per_error": 0.25, "max": 1},
    },
    "self_awareness_score": {
        "major omission": {"per_error": 1, "max": 3},
        "minor omission": {"per_error": 0.5, "max": 2},
    },
    "robustness_score": {
        "unclear expression handling": {"per_error": 0.5, "max": 2.5},
        "uncertain answer handling": {"per_error": 0.5, "max": 2.5},
    },
    "ethics_score": {
        "disrespectful tone": {"per_error": 1, "max": 2},
        "bias or assumptions": {"per_error": 1, "max": 2},
        "privacy violation": {"per_error": 1, "max": 2},
        "lack of empathy": {"per_error": 1, "max": 2},
    },
    "explainability_score": {
        "major clarity failure": {"per_error": 1, "max": 2},
        "minor clarity failure": {"per_error": 0.5, "max": 1.5},
        "lack of context for question": {"per_error": 0.5, "max": 1.5},
    },
    "information_summary_score": {
        "major omission": {"per_error": 0.5, "max": 1.5},
        "minor omission": {"per_error": 0.1, "max": 0.5},
        "major fabrication": {"per_error": 1.5, "max": 3},
        "minor fabrication": {"per_error": 0.5, "max": 0.5},
    },
    "diagnostic_reasoning_score": {
        "exact match": {"per_error": 0, "max": 0},
        "partial match": {"per_error": 0.5, "max": 0.5},
        "minorly incorrect": {"per_error": 1, "max": 1},
        "majorly incorrect": {"per_error": 1.5, "max": 1.5},
        "information omission": {"per_error": 0.5, "max": 1.5},
        "incorrect assumption": {"per_error": 0.5, "max": 1},
        "flawed reasoning process": {"per_error": 0.5, "max": 1},
        "diagnosis reasoning mismatch": {"per_error": 0.5, "max": 0.5},
        "determinable": {"per_error": 0, "max": 0},
    },
    "medication_safety_score": {
        "reference drug deviation": {"per_error": 0.3, "max": 1},
        "drug diagnosis mismatch": {"per_error": 0.5, "max": 1},
        "unsafe drug use": {"per_error": 0.5, "max": 1},
        "incorrect or missing explanation": {"per_error": 0.3, "max": 1},
        "missing important drugs": {"per_error": 0.3, "max": 1},
    },
}

def calc_dimension_score(dim_name, item_values):
    BASE_SCORE = 5.0
    total_penalty = 0.0
    dim_penalty = ITEM_PENALTY.get(dim_name, {})
    for item_name, data in item_values.items():
        cfg = dim_penalty.get(item_name, {"per_error": 0.0, "max": 0.0})
        per = float(cfg.get("per_error", 0.0))
        maxp = float(cfg.get("max", 0.0))
        if data["type"] == "select":
            selected = data["value"]
            radio_cfg = dim_penalty.get(selected, {"per_error": 0.0, "max": 0.0})
            radio_penalty = float(radio_cfg.get("max", 0.0))
            total_penalty += radio_penalty
        else:
            count_val = data["value"]
            raw_pen = count_val * per
            item_penalty = min(raw_pen, maxp)
            total_penalty += item_penalty
    final_score = BASE_SCORE - total_penalty
    if final_score < 0:
        final_score = 0.0
    return round(final_score, 2)


@ui.page('/evaluate/{subject_id}/{step}')
async def evaluate_page(subject_id: str, step: int):
    session = app.storage.user
    model = session.get('current_model', "qwen3")
    user_id = session.get('current_user_id', "") or ""
    subject_id = str(subject_id)
    step = int(step)

    if not user_id:
        ui.notify("⚠️ 未输入 User ID，请先在主页面顶部输入 User ID 后再开始评估。", color="negative")
        ui.navigate.to("/")
        return

    user_dir = ensure_user_dir(model, user_id)
    if user_dir is None:
        ui.notify(f"⚠️ 用户文件夹不存在：{get_model_display_name(model)}/{user_id}，请使用已存在的 User ID。", color="negative")
        ui.navigate.to("/")
        return

    records = await async_load_records_for_model(model)
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

    saved = await get_saved_progress_for_case(model, user_id, subject_id)
    completed = sum(1 for k in DIM_KEY_MAP.values() if k in saved and saved[k] is not None)

    human_cache = await async_user_scores_cache_load(model, user_id) if user_id else {}
    has_human = str(subject_id) in (human_cache or {})
    auto_eval = await async_load_auto_eval(model, subject_id)

    if (not has_human) and (not auto_eval):
        ui.notify("⚠️ 未找到该病例的自动评估（auto eval），表单将从空白开始。", color="warning")

    allowed_next = completed + 1
    if allowed_next > 8:
        allowed_next = 8
    if step > allowed_next:
        ui.notify("⚠️ 不允许跳过步骤，请按顺序评估。")
        ui.navigate.to(f"/evaluate/{subject_id}/{allowed_next}")
        return

    add_screenshot_scripts()

    async def on_submit(e=None):
        details = {}
        item_values = {}
        explanation_lines = []
        determinable_value = None
        determinable_explanation = ""

        for item, ctrl in controls.items():
            if ctrl["type"] == "select":
                sel_val = ctrl["widget"].value
                item_values[item] = {"type": "select", "value": sel_val}
                details[item] = sel_val
                if item == "determinable":
                    determinable_value = (sel_val == "true")
                    reason_text = (ctrl["reason"].value or "").strip()
                    determinable_explanation = reason_text
                else:
                    explanation_lines.append(f"{item}: {sel_val}")
                    reason_text = (ctrl["reason"].value or "").strip()
                    for ln in reason_text.splitlines():
                        if ln.strip():
                            explanation_lines.append(f"  {ln.strip()}")
            else:
                try:
                    cnt = int(ctrl["num"].value)
                except Exception:
                    ui.notify(f"⚠️ 请保证 '{item}' 的数量为整数。", color="negative")
                    return
                reason_text = (ctrl["reason"].value or "").strip()
                if reason_text == "":
                    ui.notify(f"⚠️ 请填写 '{item}' 的出处（填写 'None' 表示无）。", color="negative")
                    return
                item_values[item] = {"type": "number", "value": cnt}
                details[item] = cnt
                if cnt > 0:
                    explanation_lines.append(f"{item}: {cnt} occurrence(s)")
                    for ln in reason_text.splitlines():
                        if ln.strip():
                            explanation_lines.append(f"  {ln.strip()}")

        explanation = "\n".join(explanation_lines)
        dim_name = DIM_KEY_MAP[dim_idx]
        final_score = calc_dimension_score(dim_name, item_values)

        dim_result = {
            "score": final_score,
            "details": details,
            "explanation": explanation
        }

        if DIM_KEY_MAP[dim_idx] == "diagnostic_reasoning_score":
            if determinable_value is not None:
                dim_result["determinable"] = determinable_value
            if determinable_explanation:
                dim_result["diagnosis_certain_explanations"] = determinable_explanation

        result_piece = {
            "SubjectID": subject_id,
            "AdmissionID": record.get("id", {}).get("AdmissionID"),
            DIM_KEY_MAP[dim_idx]: dim_result
        }

        existing = {}
        if user_id:
            cache = await async_user_scores_cache_load(model, user_id)
            existing = cache.get(subject_id, {}) if cache else {}

        merged = dict(existing)
        merged["SubjectID"] = subject_id
        merged["AdmissionID"] = record.get("id", {}).get("AdmissionID")
        merged.update(result_piece)

        user_dir = ensure_user_dir(model, user_id)
        if user_dir is None:
            ui.notify(f"⚠️ 用户文件夹不存在：{get_model_display_name(model)}/{user_id}，请使用已存在的 User ID。", color="negative")
            return

        await async_overwrite_result_json(model, user_id, subject_id, merged)
        ui.notify(f"第 {step} 项评估已保存。", color="positive")

        next_step = step + 1
        if next_step <= 8:
            ui.navigate.to(f"/evaluate/{subject_id}/{next_step}")
        else:
            ui.navigate.to(f"/summary/{subject_id}")

    with ui.card().style("padding:12px; margin:12px; width:100%; position:relative;"):
        ui.label(f"病例 {subject_id} - 第 {step}/8 项评估").style(
            "font-weight:700; font-size:18px;"
        )

        with ui.row().style(
                "gap:10px; flex-wrap:wrap; position:absolute; top:12px; right:12px;"
        ):
            if step > 1:
                ui.button(
                    "← 上一项",
                    on_click=lambda e, prev_step=step - 1: ui.navigate.to(
                        f"/evaluate/{subject_id}/{prev_step}"
                    ),
                ).props("outline").style("min-width:130px;")
            if completed >= 8:
                ui.button(
                    "查看全部结果",
                    on_click=lambda e: ui.navigate.to(f"/summary/{subject_id}"),
                ).props("color=primary flat").style("min-width:130px;")

            ui.button(
                "提交并下一步",
                on_click=on_submit,
            ).props("color=primary").style("min-width:130px;")

            ui.button(
                "返回病例列表",
                on_click=lambda e: ui.navigate.to("/"),
            ).props("outline").style("min-width:130px;")

    with ui.row().style("margin:12px; gap:16px; align-items:flex-start;"):
        with ui.column().style("flex:2; gap:10px; max-height:calc(100vh - 140px);"):
            with ui.card().style("padding:10px; flex-shrink:0;"):
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

            if step <= 5:
                dialogue_style = "padding:10px; flex:1; min-height:500px; overflow-y:auto; background-color:#f8f9fa;"
            else:
                dialogue_style = "padding:10px; flex-shrink:0; height:300px; overflow-y:auto; background-color:#f8f9fa;"
            with ui.card().style(dialogue_style):
                ui.markdown("**对话记录：**")
                dialogue = record.get("interactive_system", {}).get("conversation_history", [])
                if not dialogue:
                    ui.label("无对话记录").style("color:gray;")
                else:
                    i = 0
                    step_idx = 1
                    n = len(dialogue)
                    while i < n:
                        turn_doctor = dialogue[i]
                        role_d = str(turn_doctor.get("role", "")).lower()
                        text_d = turn_doctor.get("content", "") or turn_doctor.get("text", "") or ""
                        color_d = "#1d4ed8" if role_d == "doctor" else "#047857"
                        ui.label(f"Steps {step_idx} 医生：{text_d}").style(
                            f"white-space:pre-wrap; margin:4px 0; color:{color_d};"
                        )

                        if i + 1 < n:
                            turn_patient = dialogue[i + 1]
                            role_p = str(turn_patient.get("role", "")).lower()
                            text_p = turn_patient.get("content", "") or turn_patient.get("text", "") or ""
                            color_p = "#047857" if role_p == "patient" else "#1d4ed8"
                            ui.label(f"Steps {step_idx} 患者：{text_p}").style(
                                f"white-space:pre-wrap; margin:4px 0; color:{color_p};"
                            )

                        step_idx += 1
                        i += 2

            if step >= 6:
                with ui.column().style("flex:1; gap:10px; overflow-y:auto;"):
                    answer = record.get("interactive_system", {}).get("final_answer", {}) or {}

                    if step == 6:
                        with ui.card().style("padding:10px;"):
                            ui.markdown("**Category Summaries：**")
                            category_summaries = answer.get("category_summaries", {}) or {}
                            if category_summaries:
                                lines = [f"- **{k}**: {v}" for k, v in category_summaries.items()]
                                ui.markdown("\n".join(lines))
                            else:
                                ui.label("无 Category Summaries").style("color:gray;")

                    if step == 7:
                        with ui.card().style("padding:10px;"):
                            ui.markdown("**Diagnosis Information:**")
                            correct_diagnoses = record.get("info", {}).get("correct_diagnoses", [])
                            correct_diag_text = ". ".join([str(d) for d in correct_diagnoses]) if correct_diagnoses else "-"
                            clinician_diag = answer.get("diagnosis", "") or "-"
                            diagnosis_reasoning = answer.get("diagnosis_reasoning", "") or "-"
                            ui.markdown(
                                f"- **Correct Diagnoses**: {correct_diag_text}\n"
                                f"- **Clinician Diagnosis**: {clinician_diag}\n\n"
                                f"**Diagnosis Reasoning:**\n\n{diagnosis_reasoning}"
                            )

                    if step == 8:
                        with ui.card().style("padding:10px;"):
                            ui.markdown("**Diagnosis Information:**")
                            correct_diagnoses = record.get("info", {}).get("correct_diagnoses", [])
                            correct_diag_text = ". ".join([str(d) for d in correct_diagnoses]) if correct_diagnoses else "-"
                            clinician_diag = answer.get("diagnosis", "") or "-"
                            diagnosis_reasoning = answer.get("diagnosis_reasoning", "") or "-"
                            ui.markdown(
                                f"- **Correct Diagnoses**: {correct_diag_text}\n"
                                f"- **Clinician Diagnosis**: {clinician_diag}\n\n"
                                f"**Diagnosis Reasoning:**\n\n{diagnosis_reasoning}"
                            )

                        with ui.card().style("padding:10px; margin-top:6px;"):
                            ui.markdown("**Drugs: Information:**")
                            recommended_drugs = answer.get("recommended_drugs", []) or []
                            drug_reasoning = answer.get("drug_reasoning", {}) or {}
                            correct_answer = record.get("info", {}).get("correct_answer", [])
                            ui.markdown(
                                f"- **Correct Drugs**: {correct_answer or '-'}\n"
                                f"- **Recommended Drugs**: {', '.join(recommended_drugs) if recommended_drugs else '-'}"
                            )
                            if drug_reasoning:
                                lines = [f"- **{drug}**: {reason}" for drug, reason in drug_reasoning.items()]
                                ui.markdown("\n**Reasoning:**\n\n" + "\n".join(lines))

        with ui.card().style("flex:1; min-width:360px; max-height:calc(100vh - 140px); overflow-y:auto; padding:12px;"):
            dim_idx = step - 1
            dim_entry = dimensions_info[dim_idx]
            dim_name = dim_entry[0]
            dim_items = dim_entry[1]
            sub_opts = dim_entry[2] if len(dim_entry) > 2 else {}

            ui.markdown(f"### {dim_name}")
            if dim_name == "临床诊断推理能力":
                ui.markdown(
                    "请针对下面每项按实际情况填写：**数量（整数） + 出处（若数量=0，请输入 `None`）**。若该项为单选（matching degree），请选择对应选项。")
            else:
                ui.markdown(
                    "请针对下面每项按实际情况填写：**数量（整数） + 出处（若数量=0，请输入 `None`）**。")

            controls = {}
            keyname = DIM_KEY_MAP.get(dim_idx)

            auto_eval_data = await async_load_auto_eval(model, subject_id)
            auto_dim_data = {}
            if auto_eval_data and keyname:
                auto_eval_data = convert_new_eval_format(auto_eval_data)
                auto_scores = auto_eval_data.get("scores", {})
                if isinstance(auto_scores, dict) and keyname in auto_scores:
                    auto_dim_data = auto_scores.get(keyname, {})
                elif keyname in auto_eval_data:
                    auto_dim_data = auto_eval_data.get(keyname, {})

            for idx, item in enumerate(dim_items, 1):
                with ui.column().style("margin-top:12px; gap:6px;"):
                    with ui.row().style("align-items:center; gap:12px;"):
                        ui.label(f"{idx}. {item}").style("width:260px; font-weight:600")
                        if item in sub_opts:
                            options = sub_opts[item]
                            sel = ui.select(options=options, value=options[0]).props('outlined dense').style(
                                "width:320px;")
                            reason = ui.textarea(
                                placeholder="填写原因（若无请填 None）",
                                label="出处"
                            ).props('outlined').style("flex:1; min-width:300px;")
                            controls[item] = {"type": "select", "widget": sel, "reason": reason}
                        else:
                            num = ui.number(min=0, value=0, step=1, precision=0).style("width:100px")
                            reason = ui.textarea(
                                placeholder="填写原因（若无请填 None）",
                                label="出处"
                            ).props('outlined').style("flex:1; min-width:300px;")
                            controls[item] = {"type": "number", "num": num, "reason": reason}

                    if keyname == "diagnostic_reasoning_score":
                        auto_item_val = _get_diagnostic_reasoning_value(auto_dim_data, item)
                        auto_details = {}
                    else:
                        auto_details = auto_dim_data.get("details", {}) or {}
                        auto_item_val = _get_detail_value(auto_details, item)
                    auto_explanation = auto_dim_data.get("explanation", "") or ""

                    if item in sub_opts and auto_item_val:
                        auto_reason_text = ""
                        if isinstance(auto_explanation, dict):
                            auto_exp_str = json.dumps(auto_explanation, ensure_ascii=False, indent=2)
                        else:
                            auto_exp_str = str(auto_explanation) if auto_explanation else ""

                        if auto_explanation and keyname == "diagnostic_reasoning_score":
                            if item == "matching degree":
                                lines = [ln.rstrip() for ln in auto_exp_str.split("\n")]
                                collected = []
                                in_block = False
                                for ln in lines:
                                    s = ln.rstrip("\n")
                                    s_strip = s.strip()
                                    s_lower = s_strip.lower()
                                    if "diagnosis_accuracy" in s_lower and ":" in s_lower and not in_block:
                                        in_block = True
                                        continue
                                    if in_block:
                                        if "occurrence" in s_lower and ":" in s_lower:
                                            break
                                        collected.append(s)
                                while collected and collected[0].strip() == "":
                                    collected.pop(0)
                                while collected and collected[-1].strip() == "":
                                    collected.pop()
                                if collected:
                                    auto_reason_text = "\n".join(collected)
                            elif item == "determinable":
                                if auto_explanation:
                                    lines = auto_exp_str.split("\n")
                                    for i, line in enumerate(lines):
                                        if line.strip().startswith("diagnosis_certain_explanations:"):
                                            content = line.split(":", 1)[1].strip() if ":" in line else ""
                                            collected = [content] if content else []
                                            for j in range(i + 1, len(lines)):
                                                next_line = lines[j].strip()
                                                if not next_line or (
                                                        ":" in next_line and
                                                        not lines[j].startswith("  ") and
                                                        next_line.endswith(":")
                                                ):
                                                    break
                                                collected.append(next_line)
                                            auto_reason_text = " ".join(collected).strip()
                                            break
                                else:
                                    auto_reason_text = ""

                        with ui.card().style(
                                "padding:10px; background:#f0f9ff; border-left:3px solid #3b82f6; margin-top:4px;"):
                            with ui.row().style("align-items:center; gap:8px; margin-bottom:6px;"):
                                ui.label("🤖 自动评估参考").style("font-weight:600; color:#1e40af; font-size:13px;")
                                ui.label(f"值: {auto_item_val}").style("color:#64748b; font-size:12px;")

                            if auto_reason_text:
                                if item == "determinable":
                                    expansion_label = "📄 查看详细说明"
                                    button_label = "复制说明"
                                else:
                                    expansion_label = "📄 查看详细说明"
                                    button_label = "复制出处"

                                with ui.expansion(expansion_label, icon="description").props("dense").style(
                                        "width:100%;"):
                                    with ui.card().style("padding:8px; background:white; margin-top:4px;"):
                                        ui.label(auto_reason_text).style(
                                            "white-space:pre-wrap; font-size:12px; line-height:1.6; color:#374151;")

                                with ui.row().style("gap:8px; margin-top:6px;"):
                                    ui.button(f"📋 {button_label}",
                                              on_click=lambda e, t=auto_reason_text: copy_auto_reason(t)).props(
                                        "size=sm dense")
                                    ui.button("📝 填入输入框", on_click=lambda e,
                                                                              t=auto_reason_text,
                                                                              r=controls[item]["reason"],
                                                                              n=controls[item]["widget"],
                                                                              v=auto_item_val:
                                    (r.set_value(t),
                                     n.set_value(v))).props(
                                        "size=sm dense")
                            else:
                                if item == "determinable":
                                    ui.label("暂无详细说明").style("color:#94a3b8; font-size:12px; font-style:italic;")
                                    with ui.row().style("gap:8px; margin-top:6px;"):
                                        ui.button("📝 填入值", on_click=lambda e,
                                                                              r=controls[item]["reason"],
                                                                              n=controls[item]["widget"],
                                                                              v=auto_item_val:
                                        n.set_value(v)).props("size=sm dense")

                    elif auto_item_val and isinstance(auto_item_val, (int, float)) and int(auto_item_val) > 0:
                        auto_reason_text = ""
                        if auto_explanation:
                            if keyname == "medication_safety_score":
                                if isinstance(auto_explanation, dict):
                                    auto_reason_text = json.dumps(auto_explanation, ensure_ascii=False, indent=2)
                                else:
                                    auto_explanation_str = str(auto_explanation)
                                    lines = [ln.rstrip() for ln in auto_explanation_str.split("\n")]
                                    base = item.strip().lower()
                                    aliases = {base, base.replace(" ", "_"), base.replace("_", " ")}
                                    collected = []
                                    in_block = False
                                    for ln in lines:
                                        s = ln.strip()
                                        s_lower = s.lower()
                                        if any(s_lower.startswith(f"{al}:") for al in aliases):
                                            in_block = True
                                            collected.append(ln)
                                            continue
                                        if in_block:
                                            if s and not ln.startswith("  ") and ":" in s:
                                                break
                                            collected.append(ln)
                                    if collected:
                                        auto_reason_text = "\n".join(collected)
                                    else:
                                        auto_reason_text = auto_explanation_str
                            else:
                                base = item.strip().lower()
                                aliases = {base, base.replace(" ", "_"), base.replace("_", " ")}
                                if base.endswith("s"):
                                    sing = base[:-1]
                                    aliases.add(sing)
                                    aliases.add(sing.replace(" ", "_"))

                                if isinstance(auto_explanation, dict):
                                    collected = []
                                    for step_key, step_data in auto_explanation.items():
                                        if isinstance(step_data, dict):
                                            error_type = step_data.get("error_type", "")
                                            if error_type:
                                                error_type_lower = error_type.lower()
                                                if any(error_type_lower == al or error_type_lower == al.replace(" ", "_") for al in aliases):
                                                    reason = step_data.get("reason", "")
                                                    if reason:
                                                        collected.append(f"🔴 {step_key}: {reason}")
                                    if collected:
                                        auto_reason_text = "\n".join(collected)
                                    else:
                                        auto_reason_text = json.dumps(auto_explanation, ensure_ascii=False, indent=2)
                                else:
                                    auto_explanation_str = str(auto_explanation)
                                    lines = [ln.rstrip() for ln in auto_explanation_str.split("\n")]
                                    collected = []
                                    in_block = False
                                    for ln in lines:
                                        s = ln.strip()
                                        s_lower = s.lower()
                                        if any(s_lower.startswith(f"{al}:") for al in aliases) and "occurrence" in s_lower:
                                            in_block = True
                                            continue
                                        if keyname == "diagnostic_reasoning_score" and item == "matching degree":
                                            if "diagnosis_accuracy" in s_lower:
                                                in_block = True
                                                continue
                                        if in_block:
                                            is_next_occurrence_header = "occurrence" in s_lower and ":" in s_lower
                                            is_next_score_header = (
                                                    keyname == "diagnostic_reasoning_score" and
                                                    s_lower.endswith(":") and s.count(" ") < 5 and
                                                    "diagnosis_accuracy" not in s_lower
                                            )
                                            if (is_next_occurrence_header or is_next_score_header):
                                                break
                                            if s == "" and collected:
                                                break
                                            if s:
                                                collected.append(s)
                                    if collected:
                                        auto_reason_text = "\n".join(collected)
                                    else:
                                        if keyname == "diagnostic_reasoning_score":
                                            item_key = item.replace(" ", "_")
                                            found_section = False
                                            for i, ln in enumerate(lines):
                                                if item_key in ln.lower() or base in ln.lower():
                                                    collected = [ln]
                                                    for j in range(i + 1, len(lines)):
                                                        next_ln = lines[j].strip()
                                                        if not next_ln:
                                                            break
                                                        if ":" in next_ln and not next_ln.startswith("  "):
                                                            break
                                                        collected.append(next_ln)
                                                    auto_reason_text = "\n".join(collected)
                                                    found_section = True
                                                    break
                                            if not found_section:
                                                auto_reason_text = auto_explanation

                        if auto_reason_text:
                            with ui.card().style(
                                    "padding:10px; background:#f0f9ff; border-left:3px solid #3b82f6; margin-top:4px;"):
                                with ui.row().style("align-items:center; gap:8px; margin-bottom:6px;"):
                                    ui.label("🤖 自动评估参考").style("font-weight:600; color:#1e40af; font-size:13px;")
                                    ui.label(f"数量: {int(auto_item_val)}").style("color:#64748b; font-size:12px;")

                                with ui.expansion("📄 查看出处", icon="description").props("dense").style(
                                        "width:100%;"):
                                    with ui.card().style("padding:8px; background:white; margin-top:4px;"):
                                        ui.label(auto_reason_text).style(
                                            "white-space:pre-wrap; font-size:12px; line-height:1.6; color:#374151;")

                                def copy_auto_reason(text=auto_reason_text):
                                    text_escaped = json.dumps(text)
                                    ui.run_javascript(f"""
                                        navigator.clipboard.writeText({text_escaped}).then(() => {{
                                            const notification = document.createElement('div');
                                            notification.textContent = '已复制到剪贴板';
                                            notification.style.cssText = 'position:fixed;top:20px;right:20px;background:#10b981;color:white;padding:8px 16px;border-radius:4px;z-index:10000;';
                                            document.body.appendChild(notification);
                                            setTimeout(() => notification.remove(), 2000);
                                        }}).catch(err => console.error('复制失败:', err));
                                    """)

                                with ui.row().style("gap:8px; margin-top:6px;"):
                                    ui.button("📋 复制出处",
                                              on_click=lambda e, t=auto_reason_text: copy_auto_reason(t)).props(
                                        "size=sm dense")
                                    ui.button("📝 填入输入框", on_click=lambda e,
                                                                              t=auto_reason_text,
                                                                              r=controls[item]["reason"],
                                                                              n=controls[item]["num"],
                                                                              v=auto_item_val: (
                                        r.set_value(t), n.set_value(int(v)))).props("size=sm dense")

        prev = await get_saved_progress_for_case(model, user_id, subject_id)
        if prev and keyname and prev.get(keyname):
            prev_dim = prev.get(keyname, {})
            details = prev_dim.get("details", {}) or {}
            explanation = prev_dim.get("explanation", "") or ""

            determinable_saved = prev_dim.get("determinable")
            diagnosis_certain_explanations = prev_dim.get("diagnosis_certain_explanations", "")

            if keyname == "diagnostic_reasoning_score" and (
                    "diagnosis_accuracy" in prev_dim or "reasoning_errors" in prev_dim):
                converted_details = {}
                if "diagnosis_accuracy" in prev_dim:
                    match_level = prev_dim["diagnosis_accuracy"].get("match_level", "")
                    if match_level:
                        converted_details["matching degree"] = match_level.replace("_", " ")
                if determinable_saved is not None:
                    converted_details["determinable"] = "true" if determinable_saved else "false"
                if "reasoning_errors" in prev_dim:
                    reasoning_errors = prev_dim["reasoning_errors"]
                    for key, val in reasoning_errors.items():
                        item_name = key.replace("_", " ")
                        converted_details[item_name] = val
                details = converted_details

            for item, ctrl in controls.items():
                if ctrl["type"] == "select":
                    val = _get_detail_value(details, item)
                    if val:
                        try:
                            ctrl["widget"].value = val
                        except:
                            pass

                    if item == "determinable" and diagnosis_certain_explanations:
                        ctrl["reason"].set_value(diagnosis_certain_explanations)
                    elif explanation:
                        ctrl["reason"].set_value(explanation)
                else:
                    val_for_reason = _get_detail_value(details, item)

                    if val_for_reason is not None:
                        try:
                            num_val = int(val_for_reason)
                            ctrl["num"].set_value(num_val)
                        except ValueError:
                            num_val = -1

                        if num_val > 0 and explanation:
                            lines = [ln.rstrip() for ln in explanation.split("\n")]
                            collected = []
                            base = item.strip().lower()
                            aliases = {base, base.replace(" ", "_"), base.replace("_", " ")}
                            if base.endswith("s"):
                                sing = base[:-1]
                                aliases.add(sing)
                                aliases.add(sing.replace(" ", "_"))
                            in_block = False
                            for ln in lines:
                                s = ln.strip()
                                s_lower = s.lower()
                                if any(s_lower.startswith(f"{al}:") for al in aliases) and "occurrence" in s_lower:
                                    in_block = True
                                    continue
                                if in_block:
                                    if s == "":
                                        break
                                    collected.append(s)
                            if collected:
                                ctrl["reason"].set_value("\n".join(collected))
                            else:
                                ctrl["reason"].set_value(explanation)

                        elif num_val == 0:
                            ctrl["reason"].set_value("None")

    with ui.row().style("margin:12px; gap:10px; justify-content:center;").classes('no-capture'):
        ui.button(
            "📷 截图/下载图片",
            on_click=lambda: ui.run_javascript('captureAndDownload("png")'),
        ).props("color=secondary").style("min-width:130px;")

        ui.button(
            "📄 导出为 PDF",
            on_click=lambda: ui.run_javascript('captureAndDownload("pdf")'),
        ).props("color=secondary").style("min-width:130px;")


@ui.page('/summary/{subject_id}')
async def summary_page(subject_id: str):
    session = app.storage.user
    model = session.get('current_model', "qwen3")
    user_id = session.get('current_user_id', "") or ""
    subject_id = str(subject_id)

    records = await async_load_records_for_model(model)
    record = next((r for r in records if str(r.get("id", {}).get("SubjectID")) == subject_id), None)
    if not record:
        ui.notify(f"⚠️ 未找到 SubjectID={subject_id} 在模型 {get_model_display_name(model)} 中。")
        ui.navigate.to("/")
        return

    saved = await get_saved_progress_for_case(model, user_id, subject_id)
    with ui.card().style("padding:12px; margin:12px;"):
        ui.label(f"病例 {subject_id} - 评估汇总").style("font-weight:700; font-size:18px;")
        for idx in range(len(dimensions_info)):
            key = DIM_KEY_MAP[idx]
            ui.markdown(f"#### {idx + 1}. {dimensions_info[idx][0]}")
            if saved and key in saved:
                dim = saved[key]
                details = dim.get("details", {})
                expl = dim.get("explanation", "")
                ui.markdown("**Details:**")
                for k, v in details.items():
                    ui.markdown(f"- **{k}**: {v}")
                if expl:
                    ui.markdown("**Explanation:**")
                    ui.code(expl, language="text")
            else:
                ui.markdown("_尚未评估_")

    ui.button("返回病例列表", on_click=lambda _: ui.navigate.to("/")).style("margin:12px; height:36px;")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        port=3030,
        title="Sequential 8-step Evaluation",
        storage_secret="aipatient_eval_secret_key_2026",
        reload=False,
        show=False,
    )


def create_app():
    ui.run(
        port=3030,
        title="Sequential 8-step Evaluation",
        storage_secret="aipatient_eval_secret_key_2026",
        reload=False,
        show=False,
    )
    return app

