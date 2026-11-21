"""
Agent Pipeline module for SkillBridge EdTech Agentic RAG (Cell 9 logic).
Implements the system prompt, multi-turn tool calling loop with Groq key rotation,
query classification, and structured response parsing.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field

from src.config import (
    REFERENCE_DATE_STR,
    DEFAULT_GROQ_MODEL,
    config,
)
from src.tools import (
    get_all_tool_schemas,
    execute_tool,
    retrieve_policy_documents,
    calculate_weighted_attendance,
    calculate_composite_grade,
    calculate_date_metrics,
    evaluate_academic_alert,
    lookup_student_data,
)

logger = logging.getLogger("rag_agent")


class AgentResponse(BaseModel):
    """Structured response from the SkillBridge EdTech Academic Reasoning Agent."""
    query_id: Optional[str] = None
    query: str
    answerable: int = Field(..., description="1 if answerable from data/policies, 0 if out-of-scope or unanswerable")
    category: str = Field(..., description="'Data Lookup', 'Rule-Based', or 'Out-of-Scope'")
    answer: Union[str, float, int] = Field(..., description="Final concise answer (number, string, or 'Out-of-scope')")
    reasoning: str = Field("", description="Step-by-step rationale for the conclusion")
    tool_calls_count: int = 0


# --------------------------------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are the SkillBridge EdTech Academic Reasoning Agent.
Your mission is to accurately answer natural language queries regarding student performance,
attendance, grading, and academic status across structured student records and institutional policies.

CRITICAL INSTRUCTIONS & CONSTRAINTS:
1. REFERENCE BENCHMARK DATE:
   The official reference date for ALL calculations is strictly **{REFERENCE_DATE_STR}** (January 15, 2025).
   Any date-relative computation (weeks since enrollment, elapsed days, expected progress, status to date)
   MUST be calculated relative to {REFERENCE_DATE_STR}.

2. QUERY CLASSIFICATION:
   Every query belongs to exactly one of three categories:
   - [Data Lookup]: Answerable directly by querying CSV tables (students, courses, attendance, assessments).
   - [Rule-Based]: Requires applying an institutional formula or policy threshold to the data.
     * Always look up policy documents first or use the dedicated calculation tools.
     * Weighted Attendance Formula: (Lecture Attendance % * 0.40) + (Lab Attendance % * 0.60). Minimum passing attendance is 75.0%.
     * Composite Course Grade Formula: (Continuous * 0.30) + (Midterm * 0.30) + (Final Exam * 0.40). Passing requires Composite >= 50% AND Final Exam >= 40%.
     * Weeks Elapsed Formula: floor((2025-01-15 - Start_Date) / 7).
   - [Out-of-Scope]: Cannot be answered from provided academic records or institutional policies.
     * Examples: Tuition fees, financial balances, sports/athletics, student dorm rooms, cafeteria meals, personal phone numbers/addresses, starting salaries, or unregistered student/course IDs.
     * For Out-of-Scope queries, you MUST set answerable = 0, category = "Out-of-Scope", and answer = "Out-of-scope".

3. TOOL USAGE:
   - Use `retrieve_policy_documents` to find policy rules and formulas.
   - Use `lookup_student_data` to inspect student, course, and exam records.
   - Use `calculate_weighted_attendance`, `calculate_composite_grade`, `calculate_date_metrics`, and `evaluate_academic_alert` for exact computations.
   - Use `execute_math` for any arithmetic calculations.

4. OUTPUT FORMAT:
   You MUST return your final response as valid JSON with the following schema:
   {{
     "answerable": 1 or 0,
     "category": "Data Lookup" | "Rule-Based" | "Out-of-Scope",
     "answer": "<final answer number or text, or 'Out-of-scope'>",
     "reasoning": "<concise summary of steps and formulas used>"
   }}
"""


class SkillBridgeAgent:
    """
    Main Agentic RAG Pipeline.
    Manages conversations, tool invocation loop, key rotation upon rate limits,
    and structured response validation.
    """
    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        max_turns: int = 8,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.tools = get_all_tool_schemas()

    def _parse_llm_json(self, content: str) -> Dict[str, Any]:
        """Extracts JSON object from model's text response."""
        # Clean markdown code blocks if present
        text = content.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                # Fallback to finding first { to last }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start : end + 1]
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse strict JSON from text: {content[:100]}... Error: {e}")
            # Fallback heuristic parser
            is_out_of_scope = any(
                term in content.lower()
                for term in ["out-of-scope", "out of scope", "cannot be answered", "tuition", "dorm", "salary"]
            )
            return {
                "answerable": 0 if is_out_of_scope else 1,
                "category": "Out-of-Scope" if is_out_of_scope else "Data Lookup",
                "answer": "Out-of-scope" if is_out_of_scope else content.strip(),
                "reasoning": content.strip(),
            }

    def _heuristic_fallback(self, query: str, query_id: Optional[str] = None, reference_date: Optional[str] = None) -> AgentResponse:
        """
        Comprehensive rule-based and data retrieval agent.
        Accurately answers data lookups, rule applications, policy searches,
        and entity queries over structured records and institutional policy documents.
        """
        q_lower = query.lower()
        ref_date = reference_date or REFERENCE_DATE_STR

        # ------------------------------------------------------------------
        # 1. OUT-OF-SCOPE FILTER
        # ------------------------------------------------------------------
        out_of_scope_keywords = [
            "tuition", "balance", "fee", "dorm", "room number", "phone number",
            "emergency contact", "salary", "job offer", "meal plan", "basketball",
            "championship", "sports", "housing", "club", "dining hall", "cafeteria"
        ]
        if any(kw in q_lower for kw in out_of_scope_keywords):
            return AgentResponse(
                query_id=query_id,
                query=query,
                answerable=0,
                category="Out-of-Scope",
                answer="Out-of-scope",
                reasoning="The requested information (fees, housing, athletics, or personal contacts) is not tracked in the academic records or institutional policies.",
                tool_calls_count=0,
            )

        # ------------------------------------------------------------------
        # 2. ENTITY RESOLUTION (Student ID & Course ID)
        # ------------------------------------------------------------------
        # Extract student ID by regex or name
        student_match = re.search(r"\b(STU\d{3})\b", query, re.IGNORECASE)
        student_id = student_match.group(1).upper() if student_match else None

        students_df = lookup_student_data("students").get("data", [])
        if not student_id:
            for s in students_df:
                s_name = s.get("name", "").lower()
                s_first = s_name.split()[0] if s_name else ""
                if (s_name and s_name in q_lower) or (s_first and len(s_first) > 3 and s_first in q_lower):
                    student_id = s.get("student_id")
                    break

        # Extract course ID by regex or keywords
        all_codes = re.findall(r"\b([A-Za-z]{2,5}\d{3})\b", query)
        course_id = None
        for code in all_codes:
            if not code.upper().startswith("STU"):
                course_id = code.upper()
                break

        courses_df = lookup_student_data("courses").get("data", [])
        if not course_id:
            for c in courses_df:
                c_title = c.get("course_title", "").lower()
                if c_title and c_title in q_lower:
                    course_id = c.get("course_id")
                    break
                # Partial match on distinctive words
                for w in ["machine learning", "computer science", "data structures", "database", "neural", "deep learning", "software engineering", "discrete math", "probability"]:
                    if w in q_lower and w in c_title:
                        course_id = c.get("course_id")
                        break
                if course_id:
                    break

        # ------------------------------------------------------------------
        # 3. GLOBAL LIST / CATALOG QUERIES
        # ------------------------------------------------------------------
        if any(phrase in q_lower for phrase in ["list courses", "all courses", "courses offered", "course catalog", "available courses"]):
            c_list = [f"{c['course_id']}: {c['course_title']} ({c['credits']} credits, Instructor: {c['instructor']})" for c in courses_df]
            formatted_courses = " | ".join(c_list)
            return AgentResponse(
                query_id=query_id,
                query=query,
                answerable=1,
                category="Data Lookup",
                answer=formatted_courses,
                reasoning=f"Retrieved all {len(courses_df)} courses from course catalog.",
                tool_calls_count=1,
            )

        if any(phrase in q_lower for phrase in ["list students", "all students", "student list"]):
            s_list = [f"{s['student_id']}: {s['name']} ({s['cohort']}, {s['status']})" for s in students_df]
            formatted_students = " | ".join(s_list)
            return AgentResponse(
                query_id=query_id,
                query=query,
                answerable=1,
                category="Data Lookup",
                answer=formatted_students,
                reasoning=f"Retrieved all {len(students_df)} students from student database.",
                tool_calls_count=1,
            )

        # ------------------------------------------------------------------
        # 4. DATE & CALENDAR METRICS (Anchored to reference date)
        # ------------------------------------------------------------------
        if any(w in q_lower for w in ["weeks", "enrolled as of", "reference date", "elapsed", "days elapsed", "progress"]):
            if "reference date" in q_lower and not student_id:
                return AgentResponse(
                    query_id=query_id,
                    query=query,
                    answerable=1,
                    category="Rule-Based",
                    answer=ref_date,
                    reasoning=f"The active reference benchmark date for all evaluations is {ref_date}.",
                    tool_calls_count=1,
                )

            if student_id:
                s_data = lookup_student_data("students", filters={"student_id": student_id})
                if s_data.get("status") == "success" and s_data.get("data"):
                    enroll_date = s_data["data"][0]["enrollment_date"]
                    date_res = calculate_date_metrics(enroll_date, reference_date=ref_date)
                    weeks = date_res["weeks_elapsed"]
                    if "progress" in q_lower:
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=f"{date_res['expected_progress_pct']}%",
                            reasoning=f"Student {student_id} enrolled on {enroll_date}. As of {ref_date}, {weeks} weeks elapsed out of 16 weeks ({date_res['expected_progress_pct']}% expected progress).",
                            tool_calls_count=2,
                        )
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Rule-Based",
                        answer=weeks,
                        reasoning=f"Student {student_id} enrolled on {enroll_date}. As of reference date {ref_date}, {weeks} full weeks elapsed (floor of {date_res['days_elapsed']}/7).",
                        tool_calls_count=2,
                    )

        # ------------------------------------------------------------------
        # 5. WEIGHTED ATTENDANCE & THRESHOLDS
        # ------------------------------------------------------------------
        if any(term in q_lower for term in ["weighted attendance", "attendance requirement", "minimum 75%", "attendance percentage", "attendance standing", "debarred"]):
            if student_id and course_id:
                att_res = calculate_weighted_attendance(student_id, course_id)
                if att_res.get("status") == "success":
                    if any(w in q_lower for w in ["minimum 75%", "meet", "satisfy", "eligible"]):
                        meets = "Yes" if att_res["meets_75_pct_minimum"] else "No"
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=meets,
                            reasoning=f"Weighted attendance is {att_res['weighted_attendance_pct']}%. Minimum threshold is 75.0%. Meets requirement: {meets}.",
                            tool_calls_count=1,
                        )
                    if "standing" in q_lower or "status" in q_lower:
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=att_res["standing"],
                            reasoning=f"Weighted attendance is {att_res['weighted_attendance_pct']}%, placing student in {att_res['standing']}.",
                            tool_calls_count=1,
                        )
                    val = att_res["weighted_attendance_pct"]
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Rule-Based",
                        answer=val,
                        reasoning=f"Calculated using policy formula (Lecture {att_res['lecture_attendance_pct']}% * 0.40) + (Lab {att_res['lab_attendance_pct']}% * 0.60) = {val}%.",
                        tool_calls_count=1,
                    )

        # ------------------------------------------------------------------
        # 6. COMPOSITE SCORE, LETTER GRADE & PASSING CRITERIA
        # ------------------------------------------------------------------
        if any(term in q_lower for term in ["composite", "final score", "letter grade", "pass", "passing"]):
            if student_id and course_id:
                grade_res = calculate_composite_grade(student_id, course_id)
                if grade_res.get("status") == "success":
                    if "letter grade" in q_lower:
                        ans = grade_res["letter_grade"]
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=ans,
                            reasoning=f"Composite score is {grade_res['composite_score']} (Continuous: {grade_res['continuous_score']}*0.3 + Midterm: {grade_res['midterm_score']}*0.3 + Final: {grade_res['final_project_score']}*0.4), mapping to Letter Grade {ans}.",
                            tool_calls_count=1,
                        )
                    if any(w in q_lower for w in ["pass", "passing"]):
                        p_ans = "Yes" if grade_res["passed"] else "No"
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=p_ans,
                            reasoning=f"Composite score: {grade_res['composite_score']} (requires >=50.0%) and Final Exam: {grade_res['final_project_score']} (requires >=40.0%). Passed: {p_ans}.",
                            tool_calls_count=1,
                        )
                    ans = grade_res["composite_score"]
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Rule-Based",
                        answer=ans,
                        reasoning=f"Composite score computed via (Continuous {grade_res['continuous_score']}*0.30) + (Midterm {grade_res['midterm_score']}*0.30) + (Final {grade_res['final_project_score']}*0.40) = {ans}.",
                        tool_calls_count=1,
                    )

        # ------------------------------------------------------------------
        # 7. ACADEMIC ALERT & PROBATION
        # ------------------------------------------------------------------
        if any(term in q_lower for term in ["alert", "probation", "warning", "level 1", "level 2"]):
            if student_id:
                alert_res = evaluate_academic_alert(student_id, course_id)
                if alert_res.get("status") == "success":
                    if "level 2" in q_lower or "critical" in q_lower:
                        ans = "Yes" if alert_res["level_2_triggered"] else "No"
                        reason = "; ".join(alert_res["reasons"]) if alert_res["reasons"] else "No Level 2 triggers met."
                        return AgentResponse(
                            query_id=query_id,
                            query=query,
                            answerable=1,
                            category="Rule-Based",
                            answer=ans,
                            reasoning=f"Level 2 Critical Probation status: {ans}. Details: {reason}",
                            tool_calls_count=1,
                        )
                    ans = "Yes" if (alert_res["level_1_triggered"] or alert_res["level_2_triggered"]) else "No"
                    reason = "; ".join(alert_res["reasons"]) if alert_res["reasons"] else "Student is in normal standing."
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Rule-Based",
                        answer=ans,
                        reasoning=f"Alert Status: {alert_res['alert_status']}. Reasons: {reason}",
                        tool_calls_count=1,
                    )

        # ------------------------------------------------------------------
        # 8. DIRECT ASSESSMENT & ATTENDANCE RECORD LOOKUPS
        # ------------------------------------------------------------------
        if student_id and course_id:
            ass_data = lookup_student_data("assessments", filters={"student_id": student_id, "course_id": course_id})
            if ass_data.get("status") == "success" and ass_data.get("data"):
                ass_row = ass_data["data"][0]
                if "midterm" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=ass_row["midterm_score"],
                        reasoning=f"Retrieved midterm score ({ass_row['midterm_score']}) from assessments records for student {student_id} in {course_id}.",
                        tool_calls_count=1,
                    )
                if "continuous" in q_lower or "quiz" in q_lower or "homework" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=ass_row["continuous_score"],
                        reasoning=f"Retrieved continuous assessment score ({ass_row['continuous_score']}) for student {student_id} in {course_id}.",
                        tool_calls_count=1,
                    )
                if "final project" in q_lower or "final exam" in q_lower or "capstone" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=ass_row["final_project_score"],
                        reasoning=f"Retrieved final project score ({ass_row['final_project_score']}) for student {student_id} in {course_id}.",
                        tool_calls_count=1,
                    )
                if "gpa" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=ass_row["current_gpa"],
                        reasoning=f"Current GPA for student {student_id} in {course_id} is {ass_row['current_gpa']}.",
                        tool_calls_count=1,
                    )

            att_data = lookup_student_data("attendance", filters={"student_id": student_id, "course_id": course_id})
            if att_data.get("status") == "success" and att_data.get("data"):
                att_row = att_data["data"][0]
                if "lab attendance" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=att_row.get("lab_attendance_pct", 0.0),
                        reasoning=f"Lab attendance: {att_row.get('lab_sessions_attended')}/{att_row.get('lab_sessions_total')} sessions ({att_row.get('lab_attendance_pct')}%) for {student_id} in {course_id}.",
                        tool_calls_count=1,
                    )
                if "lecture attendance" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=att_row.get("lecture_attendance_pct", 0.0),
                        reasoning=f"Lecture attendance: {att_row.get('lecture_sessions_attended')}/{att_row.get('lecture_sessions_total')} sessions ({att_row.get('lecture_attendance_pct')}%) for {student_id} in {course_id}.",
                        tool_calls_count=1,
                    )

        # ------------------------------------------------------------------
        # 9. STUDENT RECORD LOOKUPS & PROFILES
        # ------------------------------------------------------------------
        if student_id:
            s_data = lookup_student_data("students", filters={"student_id": student_id})
            if s_data.get("status") == "success" and s_data.get("data"):
                row = s_data["data"][0]
                if "enrollment date" in q_lower or "enrolled on" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["enrollment_date"],
                        reasoning=f"Student {row['name']} ({student_id}) enrolled on {row['enrollment_date']}.",
                        tool_calls_count=1,
                    )
                if "cohort" in q_lower or "status" in q_lower:
                    ans = f"{row['cohort']} ({row['status']})"
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=ans,
                        reasoning=f"Student {student_id} is in cohort {row['cohort']} with academic status {row['status']}.",
                        tool_calls_count=1,
                    )
                if "name" in q_lower or "who is" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["name"],
                        reasoning=f"Student ID {student_id} corresponds to {row['name']} ({row['cohort']}, Status: {row['status']}).",
                        tool_calls_count=1,
                    )

                # Look up student's GPA across courses
                all_ass = lookup_student_data("assessments", filters={"student_id": student_id}).get("data", [])
                if "gpa" in q_lower and all_ass:
                    gpas = [a.get("current_gpa", 0.0) for a in all_ass]
                    avg_gpa = round(sum(gpas) / len(gpas), 2)
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=avg_gpa,
                        reasoning=f"Student {student_id} has cumulative GPA {avg_gpa} across {len(all_ass)} course(s).",
                        tool_calls_count=1,
                    )

                # Look up enrolled courses
                all_att = lookup_student_data("attendance", filters={"student_id": student_id}).get("data", [])
                if any(w in q_lower for w in ["courses", "enrolled in", "classes"]) and all_att:
                    c_enrolled = [a.get("course_id") for a in all_att]
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=", ".join(c_enrolled),
                        reasoning=f"Student {student_id} is actively taking: {', '.join(c_enrolled)}.",
                        tool_calls_count=1,
                    )

                # General student profile summary
                courses_taken = [a.get("course_id") for a in all_att]
                gpas = [a.get("current_gpa", 0.0) for a in all_ass]
                avg_gpa = round(sum(gpas) / len(gpas), 2) if gpas else "N/A"
                profile_summary = (
                    f"Name: {row['name']} | ID: {student_id} | Cohort: {row['cohort']} | "
                    f"Status: {row['status']} | Enrolled: {row['enrollment_date']} | "
                    f"Courses: {', '.join(courses_taken) if courses_taken else 'None'} | GPA: {avg_gpa}"
                )
                return AgentResponse(
                    query_id=query_id,
                    query=query,
                    answerable=1,
                    category="Data Lookup",
                    answer=profile_summary,
                    reasoning=f"Full academic profile for {row['name']} ({student_id}).",
                    tool_calls_count=2,
                )

        # ------------------------------------------------------------------
        # 10. COURSE RECORD LOOKUPS
        # ------------------------------------------------------------------
        if course_id:
            c_data = lookup_student_data("courses", filters={"course_id": course_id})
            if c_data.get("status") == "success" and c_data.get("data"):
                row = c_data["data"][0]
                if "credit" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["credits"],
                        reasoning=f"Course {course_id} ({row['course_title']}) is worth {row['credits']} credits.",
                        tool_calls_count=1,
                    )
                if "instructor" in q_lower or "teacher" in q_lower or "professor" in q_lower or "faculty" in q_lower or "who teaches" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["instructor"],
                        reasoning=f"Course {course_id} ({row['course_title']}) is taught by {row['instructor']}.",
                        tool_calls_count=1,
                    )
                if "title" in q_lower or "name" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["course_title"],
                        reasoning=f"Course ID {course_id} title is '{row['course_title']}'.",
                        tool_calls_count=1,
                    )
                if "department" in q_lower:
                    return AgentResponse(
                        query_id=query_id,
                        query=query,
                        answerable=1,
                        category="Data Lookup",
                        answer=row["department"],
                        reasoning=f"Course {course_id} belongs to Department of {row['department']}.",
                        tool_calls_count=1,
                    )
                # General course summary
                course_summary = f"{course_id}: {row['course_title']} | Credits: {row['credits']} | Instructor: {row['instructor']} | Department: {row['department']}"
                return AgentResponse(
                    query_id=query_id,
                    query=query,
                    answerable=1,
                    category="Data Lookup",
                    answer=course_summary,
                    reasoning=f"Catalog record for {course_id}.",
                    tool_calls_count=1,
                )

        # ------------------------------------------------------------------
        # 11. POLICY DOCUMENT SEMANTIC SEARCH
        # ------------------------------------------------------------------
        if any(term in q_lower for term in ["policy", "rule", "formula", "grading", "attendance", "probation", "alert", "how is", "what is the weight", "threshold"]):
            pol_res = retrieve_policy_documents(query, top_k=2)
            if pol_res.get("status") == "success" and pol_res.get("results"):
                best_match = pol_res["results"][0]
                summary_content = best_match["content"].replace("\n\n", " ").strip()
                # Extract first 250 characters
                short_summary = summary_content[:250] + "..." if len(summary_content) > 250 else summary_content
                return AgentResponse(
                    query_id=query_id,
                    query=query,
                    answerable=1,
                    category="Rule-Based",
                    answer=short_summary,
                    reasoning=f"Retrieved from institutional policy section '{best_match['section']}' in {best_match['source']}.",
                    tool_calls_count=1,
                )

        # ------------------------------------------------------------------
        # 12. MEANINGFUL GENERAL ANSWER
        # ------------------------------------------------------------------
        # If no specific entity was matched, search across policies or provide helpful summary
        pol_res = retrieve_policy_documents(query, top_k=1)
        if pol_res.get("status") == "success" and pol_res.get("results"):
            m = pol_res["results"][0]
            clean_text = m["text"][:300].replace("\n", " ").strip()
            return AgentResponse(
                query_id=query_id,
                query=query,
                answerable=1,
                category="Rule-Based",
                answer=clean_text,
                reasoning=f"Matched policy guidelines under section: {m['section_title']}.",
                tool_calls_count=1,
            )

        return AgentResponse(
            query_id=query_id,
            query=query,
            answerable=1,
            category="Data Lookup",
            answer=f"Academic query received for '{query}'. Please specify student ID (e.g. STU001) or course code (e.g. CS101) for detailed metrics.",
            reasoning="Query matched general academic scope but requires specific student or course parameters for deeper evaluation.",
            tool_calls_count=0,
        )

    def answer_query(self, query: str, query_id: Optional[str] = None, reference_date: Optional[str] = None) -> AgentResponse:
        """
        Executes the agent tool-calling loop to answer a natural language academic query.
        Automatically handles Groq key rotation on rate limits and supports dynamic reference dates.
        """
        ref_date = reference_date or REFERENCE_DATE_STR

        # If no Groq API keys available, use the deterministic heuristic agent
        if not config.groq_keys.has_keys:
            logger.info("Operating in offline heuristic mode (no Groq keys configured).")
            return self._heuristic_fallback(query, query_id, reference_date=ref_date)

        sys_content = SYSTEM_PROMPT
        if ref_date != REFERENCE_DATE_STR:
            sys_content = sys_content.replace(REFERENCE_DATE_STR, ref_date)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": query},
        ]

        turn = 0
        tool_calls_count = 0
        retries = 0
        max_retries = 3

        while turn < self.max_turns:
            turn += 1
            try:
                client = config.groq_keys.get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                choice = response.choices[0]
                message = choice.message

                # Case 1: Model requested tool calls
                if message.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    })

                    for tc in message.tool_calls:
                        tool_name = tc.function.name
                        tool_args = tc.function.arguments
                        tool_calls_count += 1
                        logger.info(f"[Turn {turn}] Invoking tool: {tool_name} with {tool_args}")
                        
                        tool_result = execute_tool(tool_name, tool_args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": tool_result,
                        })

                    # Reset retry counter on successful tool step
                    retries = 0
                    continue

                # Case 2: Model returned final text answer
                final_content = message.content or ""
                parsed = self._parse_llm_json(final_content)

                return AgentResponse(
                    query_id=query_id,
                    query=query,
                    answerable=int(parsed.get("answerable", 1)),
                    category=str(parsed.get("category", "Data Lookup")),
                    answer=parsed.get("answer", "Out-of-scope" if parsed.get("answerable") == 0 else ""),
                    reasoning=str(parsed.get("reasoning", "")),
                    tool_calls_count=tool_calls_count,
                )

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "quota" in err_str
                if is_rate_limit:
                    logger.warning(f"Groq Rate Limit encountered: {e}. Rotating key...")
                    config.groq_keys.rotate_key(cooldown_seconds=45)
                    retries += 1
                    if retries > max_retries:
                        logger.error("Max retries exceeded across Groq keys. Falling back to heuristic mode.")
                        return self._heuristic_fallback(query, query_id, reference_date=ref_date)
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Error during agent turn {turn}: {e}")
                    # Fallback to heuristic
                    return self._heuristic_fallback(query, query_id, reference_date=ref_date)

        # If turn limit reached without final answer
        logger.warning(f"Agent reached max turns ({self.max_turns}). Producing fallback answer.")
        return self._heuristic_fallback(query, query_id, reference_date=ref_date)
