"""
Agent Tools module for SkillBridge EdTech Agentic RAG System (Cell 8 logic).
Implements the Python functions and OpenAI-compatible tool schemas for Groq / LLM tool calling.
"""

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import DATA_DIR, REFERENCE_DATE, REFERENCE_DATE_STR
from src.vector_store import get_vector_store

logger = logging.getLogger("rag_tools")

# Cache for CSV datasets to avoid re-reading disk constantly
_CSV_CACHE: Dict[str, pd.DataFrame] = {}


def invalidate_csv_cache():
    """Clears the CSV dataframe cache so reloaded or modified data is fresh."""
    global _CSV_CACHE
    _CSV_CACHE.clear()


def save_dataframe(table_name: str, records: List[Dict[str, Any]]) -> bool:
    """Overwrites or updates CSV table with provided records and invalidates cache."""
    clean_name = table_name.lower().replace(".csv", "").strip()
    file_path = DATA_DIR / f"{clean_name}.csv"
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False)
    invalidate_csv_cache()
    return True


def _get_dataframe(table_name: str) -> pd.DataFrame:
    """Loads and caches CSV dataset by table name."""
    clean_name = table_name.lower().replace(".csv", "").strip()
    if clean_name not in _CSV_CACHE:
        file_path = DATA_DIR / f"{clean_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset table '{clean_name}.csv' not found in {DATA_DIR}")
        _CSV_CACHE[clean_name] = pd.read_csv(file_path)
    return _CSV_CACHE[clean_name]


# --------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS (PYTHON FUNCTIONS)
# --------------------------------------------------------------------------

def retrieve_policy_documents(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Semantic search over the policy documents vector store.
    Use this to look up institutional formulas, attendance rules, grade scales,
    alert thresholds, and out-of-scope boundaries.
    """
    try:
        vs = get_vector_store()
        matches = vs.retrieve(query, top_k=top_k)
        if not matches:
            return {"status": "empty", "message": "No matching policy sections found."}
        return {
            "status": "success",
            "results": [
                {
                    "section": m["section_title"],
                    "source": m["source_file"],
                    "content": m["text"],
                    "similarity": round(m["score"], 4),
                }
                for m in matches
            ],
        }
    except Exception as e:
        logger.error(f"Error in retrieve_policy_documents: {e}")
        return {"status": "error", "error": str(e)}


def lookup_student_data(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Queries student academic data from one of: 'students', 'courses', 'attendance', 'assessments'.
    
    Args:
        table_name: 'students', 'courses', 'attendance', or 'assessments'.
        filters: Key-value pairs to filter by (e.g. {"student_id": "STU001"}).
        fields: List of column names to return (optional).
    """
    try:
        df = _get_dataframe(table_name)
        filtered = df.copy()

        if filters:
            for col, val in filters.items():
                if col not in filtered.columns:
                    return {
                        "status": "error",
                        "error": f"Column '{col}' does not exist in table '{table_name}'. Valid columns: {list(filtered.columns)}",
                    }
                filtered = filtered[filtered[col].astype(str).str.upper() == str(val).upper()]

        if filtered.empty:
            return {
                "status": "not_found",
                "message": f"No records found in '{table_name}' matching filters {filters}",
            }

        if fields:
            valid_fields = [f for f in fields if f in filtered.columns]
            if valid_fields:
                filtered = filtered[valid_fields]

        records = filtered.to_dict(orient="records")
        return {
            "status": "success",
            "table": table_name,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        logger.error(f"Error in lookup_student_data: {e}")
        return {"status": "error", "error": str(e)}


def calculate_weighted_attendance(student_id: str, course_id: str) -> Dict[str, Any]:
    """
    Computes the exact weighted attendance for a student in a course according
    to the official policy formula:
    Weighted Attendance = (Lecture % * 0.40) + (Lab % * 0.60)
    """
    try:
        att_df = _get_dataframe("attendance")
        match = att_df[
            (att_df["student_id"].str.upper() == student_id.upper())
            & (att_df["course_id"].str.upper() == course_id.upper())
        ]

        if match.empty:
            return {
                "status": "not_found",
                "message": f"No attendance record found for student {student_id} in course {course_id}.",
            }

        row = match.iloc[0]
        lec_pct = float(row.get("lecture_attendance_pct", 0.0))
        lab_pct = float(row.get("lab_attendance_pct", 0.0))

        # Official weighting: Lecture (40%), Lab (60%)
        weighted_att = round((lec_pct * 0.40) + (lab_pct * 0.60), 2)

        # Policy Thresholds:
        # Good Standing >= 75.0%
        # Attendance Warning: 65.0% <= Att < 75.0%
        # Debarred: Att < 65.0%
        if weighted_att >= 75.0:
            standing = "Good Standing"
            alert = False
            warning_level = None
        elif weighted_att >= 65.0:
            standing = "Attendance Shortage / Warning"
            alert = True
            warning_level = "Level 1 Warning (Cap at grade B)"
        else:
            standing = "Critical Debarment"
            alert = True
            warning_level = "Debarred from Final Exam (Grade F)"

        return {
            "status": "success",
            "student_id": student_id.upper(),
            "course_id": course_id.upper(),
            "lecture_attendance_pct": round(lec_pct, 2),
            "lab_attendance_pct": round(lab_pct, 2),
            "weighted_attendance_pct": weighted_att,
            "meets_75_pct_minimum": weighted_att >= 75.0,
            "standing": standing,
            "alert_triggered": alert,
            "status_detail": warning_level,
        }
    except Exception as e:
        logger.error(f"Error in calculate_weighted_attendance: {e}")
        return {"status": "error", "error": str(e)}


def calculate_composite_grade(student_id: str, course_id: str) -> Dict[str, Any]:
    """
    Computes a student's final composite course score and letter grade according to policy:
    Composite Score = (Continuous * 0.30) + (Midterm * 0.30) + (Final Exam * 0.40)
    Also evaluates dual passing criteria (overall >= 50% AND final exam >= 40%).
    """
    try:
        ass_df = _get_dataframe("assessments")
        match = ass_df[
            (ass_df["student_id"].str.upper() == student_id.upper())
            & (ass_df["course_id"].str.upper() == course_id.upper())
        ]

        if match.empty:
            return {
                "status": "not_found",
                "message": f"No assessment record found for student {student_id} in course {course_id}.",
            }

        row = match.iloc[0]
        cont = float(row.get("continuous_score", 0.0))
        mid = float(row.get("midterm_score", 0.0))
        final = float(row.get("final_project_score", 0.0))

        composite_score = round((cont * 0.30) + (mid * 0.30) + (final * 0.40), 2)

        # Dual passing requirement: Final Exam must be >= 40.0%
        final_passed = final >= 40.0

        if composite_score >= 90.0 and final_passed:
            letter_grade = "A"
            gpa = 4.0
        elif composite_score >= 80.0 and final_passed:
            letter_grade = "B"
            gpa = 3.0
        elif composite_score >= 70.0 and final_passed:
            letter_grade = "C"
            gpa = 2.0
        elif composite_score >= 50.0 and final_passed:
            letter_grade = "D"
            gpa = 1.0
        elif composite_score >= 50.0 and not final_passed:
            # Failed final exam component rule: capped at D
            letter_grade = "D"
            gpa = 1.0
        else:
            letter_grade = "F"
            gpa = 0.0

        return {
            "status": "success",
            "student_id": student_id.upper(),
            "course_id": course_id.upper(),
            "continuous_score": cont,
            "midterm_score": mid,
            "final_project_score": final,
            "composite_score": composite_score,
            "letter_grade": letter_grade,
            "gpa": gpa,
            "passed": letter_grade != "F",
        }
    except Exception as e:
        logger.error(f"Error in calculate_composite_grade: {e}")
        return {"status": "error", "error": str(e)}


def calculate_date_metrics(
    start_date: str,
    reference_date: str = REFERENCE_DATE_STR,
    total_course_weeks: int = 16,
) -> Dict[str, Any]:
    """
    Computes elapsed days and weeks from start_date relative to the institutional
    reference date (January 15, 2025).
    """
    try:
        d_start = datetime.strptime(start_date.strip(), "%Y-%m-%d").date()
        d_ref = datetime.strptime(reference_date.strip(), "%Y-%m-%d").date()

        days_elapsed = (d_ref - d_start).days
        if days_elapsed < 0:
            return {
                "status": "invalid_date",
                "message": f"Start date {start_date} is in the future relative to reference date {reference_date}.",
            }

        weeks_elapsed = days_elapsed // 7
        progress_pct = min(100.0, round((weeks_elapsed / total_course_weeks) * 100.0, 2))

        return {
            "status": "success",
            "start_date": str(d_start),
            "reference_date": str(d_ref),
            "days_elapsed": days_elapsed,
            "weeks_elapsed": weeks_elapsed,
            "expected_progress_pct": progress_pct,
        }
    except Exception as e:
        logger.error(f"Error in calculate_date_metrics: {e}")
        return {"status": "error", "error": str(e)}


def evaluate_academic_alert(student_id: str, course_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluates whether a student triggers Level 1 Warning or Level 2 Critical Probation
    as of January 15, 2025, according to institutional policy.
    """
    try:
        att_res = calculate_weighted_attendance(student_id, course_id) if course_id else None
        
        ass_df = _get_dataframe("assessments")
        filt = ass_df[ass_df["student_id"].str.upper() == student_id.upper()]
        if course_id:
            filt = filt[filt["course_id"].str.upper() == course_id.upper()]

        if filt.empty:
            return {"status": "not_found", "message": f"No assessment records for student {student_id}"}

        alerts_triggered = []
        is_level_1 = False
        is_level_2 = False

        # Attendance checks
        if att_res and att_res.get("status") == "success":
            weighted_att = att_res["weighted_attendance_pct"]
            if weighted_att < 60.0:
                is_level_2 = True
                alerts_triggered.append(f"Severe attendance shortage ({weighted_att}% < 60.0%) -> Level 2")
            elif weighted_att < 75.0:
                is_level_1 = True
                alerts_triggered.append(f"Attendance shortage ({weighted_att}% < 75.0%) -> Level 1")

        # Score checks
        scores_under_40 = 0
        for _, row in filt.iterrows():
            if row.get("continuous_score", 100) < 50.0:
                is_level_1 = True
                alerts_triggered.append("Continuous assessment score below 50.0% -> Level 1")
            if row.get("midterm_score", 100) < 50.0:
                is_level_1 = True
                alerts_triggered.append("Midterm exam score below 50.0% -> Level 1")
            
            for score_col in ["continuous_score", "midterm_score", "final_project_score"]:
                if row.get(score_col, 100) < 40.0:
                    scores_under_40 += 1

            if row.get("current_gpa", 4.0) < 2.0:
                is_level_2 = True
                alerts_triggered.append(f"Cumulative GPA below 2.0 ({row.get('current_gpa')}) -> Level 2")

        if scores_under_40 >= 2:
            is_level_2 = True
            alerts_triggered.append(f"{scores_under_40} assessments below 40.0% -> Level 2")

        status_flag = "Normal"
        if is_level_2:
            status_flag = "Level 2: Critical Academic Probation"
        elif is_level_1:
            status_flag = "Level 1: Academic Warning"

        return {
            "status": "success",
            "student_id": student_id.upper(),
            "course_id": course_id.upper() if course_id else "All Courses",
            "alert_status": status_flag,
            "level_1_triggered": is_level_1,
            "level_2_triggered": is_level_2,
            "reasons": alerts_triggered,
        }
    except Exception as e:
        logger.error(f"Error in evaluate_academic_alert: {e}")
        return {"status": "error", "error": str(e)}


def execute_math(expression: str) -> Dict[str, Any]:
    """
    Safely computes mathematical and numerical expressions using a restricted execution scope.
    Allowed functions: round, min, max, sum, abs, math constants.
    """
    allowed_names = {
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "math": math,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    # Sanitization
    cleaned = expression.strip()
    if any(forbidden in cleaned for forbidden in ["__", "import", "eval", "exec", "open", "os", "sys"]):
        return {"status": "error", "error": "Forbidden tokens in expression."}

    try:
        result = eval(cleaned, {"__builtins__": {}}, allowed_names)
        if isinstance(result, (int, float)):
            result = round(result, 4) if isinstance(result, float) else result
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "error": f"Evaluation error: {str(e)}"}


# --------------------------------------------------------------------------
# OPENAI TOOL SCHEMAS FOR GROQ / LLM CALLING
# --------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy_documents",
            "description": "Searches unstructured institutional policy documents for official rules, formulas, thresholds, alert levels, letter grade scales, and scope boundaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query describing the rule or policy to look up (e.g. 'weighted attendance formula', 'letter grade cutoffs', 'probation conditions').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant policy passages to retrieve (default: 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_student_data",
            "description": "Queries structured academic CSV datasets: 'students', 'courses', 'attendance', or 'assessments'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "enum": ["students", "courses", "attendance", "assessments"],
                        "description": "The dataset table to query.",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Dictionary of column filter conditions, e.g. {'student_id': 'STU001'} or {'course_id': 'CS101'}.",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific columns to retrieve (e.g. ['enrollment_date', 'status']).",
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_weighted_attendance",
            "description": "Calculates the exact weighted attendance percentage for a student in a specific course: (Lecture % * 0.40) + (Lab % * 0.60), and checks policy thresholds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "Unique identifier of the student (e.g. 'STU001').",
                    },
                    "course_id": {
                        "type": "string",
                        "description": "Course identifier (e.g. 'CS101').",
                    },
                },
                "required": ["student_id", "course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_composite_grade",
            "description": "Computes composite final course score and assigns official letter grade (A, B, C, D, F) according to policy weighting: Continuous (30%), Midterm (30%), Final (40%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "Unique identifier of the student (e.g. 'STU001').",
                    },
                    "course_id": {
                        "type": "string",
                        "description": "Course identifier (e.g. 'CS101').",
                    },
                },
                "required": ["student_id", "course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_date_metrics",
            "description": "Calculates elapsed days, weeks, and course progress from a starting date relative to the institutional reference date January 15, 2025.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start or enrollment date in YYYY-MM-DD format (e.g. '2024-09-01').",
                    },
                    "reference_date": {
                        "type": "string",
                        "description": "Benchmark date for evaluation (must strictly default to '2025-01-15').",
                    },
                },
                "required": ["start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_academic_alert",
            "description": "Determines whether a student triggers Level 1 Warning or Level 2 Critical Academic Probation based on attendance, scores, and GPA thresholds as of January 15, 2025.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "Student identifier (e.g. 'STU001').",
                    },
                    "course_id": {
                        "type": "string",
                        "description": "Optional course identifier to check course-specific alert.",
                    },
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_math",
            "description": "Safely computes numerical formulas or arithmetic expressions (e.g. '(85.71 * 0.4) + (75.0 * 0.6)').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Returns the full list of OpenAI-formatted tool schemas for LLM tool calling."""
    return TOOL_SCHEMAS


def execute_tool(tool_name: str, arguments: Union[str, Dict[str, Any]]) -> str:
    """
    Executes a tool by name with provided arguments and returns a JSON string response.
    """
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments or {}

    dispatch_map = {
        "retrieve_policy_documents": retrieve_policy_documents,
        "lookup_student_data": lookup_student_data,
        "calculate_weighted_attendance": calculate_weighted_attendance,
        "calculate_composite_grade": calculate_composite_grade,
        "calculate_date_metrics": calculate_date_metrics,
        "evaluate_academic_alert": evaluate_academic_alert,
        "execute_math": execute_math,
    }

    fn = dispatch_map.get(tool_name)
    if not fn:
        err = {"status": "error", "message": f"Unknown tool '{tool_name}'"}
        return json.dumps(err)

    try:
        result = fn(**args)
        return json.dumps(result, default=str)
    except TypeError as te:
        logger.error(f"Parameter mismatch for tool '{tool_name}' with args {args}: {te}")
        return json.dumps({"status": "error", "error": f"Invalid arguments: {str(te)}"})
    except Exception as e:
        logger.error(f"Execution error for tool '{tool_name}': {e}")
        return json.dumps({"status": "error", "error": str(e)})
