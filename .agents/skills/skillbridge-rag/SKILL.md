---
name: skillbridge-rag
description: SkillBridge EdTech Agentic RAG Challenge helper. Provides guidelines, reference formulas, and execution rules for student data and policy reasoning.
---

# SkillBridge EdTech Agentic RAG Skill

Use this skill when developing, testing, or submitting solutions for the **SkillBridge EdTech Agentic RAG Challenge**.

## Core Problem Rules & Constants

1. **Reference Date Benchmark**:
   - The reference date for all calculations is **January 15, 2025** (`2025-01-15`).
   - Every elapsed date calculation (weeks since enrollment, progress) must be anchored to this date:
     $$\text{Weeks Elapsed} = \lfloor \frac{\text{Date(2025-01-15)} - D_{\text{start}}}{7} \rfloor$$

2. **Weighted Attendance**:
   - Theory/Lecture: 40% weight (`0.40`).
   - Lab/Practical: 60% weight (`0.60`).
   - Formula:
     $$\text{Weighted Attendance \%} = (\text{Lecture Attendance \%} \times 0.40) + (\text{Lab Attendance \%} \times 0.60)$$
   - Minimum threshold for Good Standing: `75.0%`.
   - Debarred threshold: `< 65.0%`.

3. **Composite Grade Calculation**:
   - Continuous Assessments (Quizzes): 30% weight (`0.30`).
   - Midterm Exam: 30% weight (`0.30`).
   - Final Exam / Project: 40% weight (`0.40`).
   - Formula:
     $$\text{Composite Score} = (\text{Continuous} \times 0.30) + (\text{Midterm} \times 0.30) + (\text{Final} \times 0.40)$$
   - Grade Cutoffs:
     * A: $90.0\% - 100.0\%$
     * B: $80.0\% - 89.9\%$
     * C: $70.0\% - 79.9\%$
     * D: $50.0\% - 69.9\%$
     * F: $< 50.0\%$
   - Dual-Passing Condition: Final Exam must be $\ge 40.0\%$ in addition to composite score $\ge 50.0\%$.

4. **Query Categories**:
   - `Data Lookup`: Answer directly from CSV tables.
   - `Rule-Based`: Apply policy formula to data.
   - `Out-of-Scope`: Cannot be answered from provided data (return `answerable = 0`, `answer = "Out-of-scope"`).

## Starter Notebook Cell Mapping

| Local File | Kaggle Starter Notebook Cell | Purpose |
|------------|------------------------------|---------|
| `src/vector_store.py` | **Cell 6** | Policy chunking, Gemini embeddings (`text-embedding-004`), FAISS index |
| `src/tools.py` | **Cell 8** | CSV lookup, math, date, and weighted attendance tools + OpenAI schemas |
| `src/agent.py` | **Cell 9** | System prompt & multi-turn tool calling agent loop |
| `run_eval.py` | **Cell 10** | Evaluation on `train.csv` & error logging |
| `generate_submission.py` | **Cell 11–12** | Test query processing & submission validation |
