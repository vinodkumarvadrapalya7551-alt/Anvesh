# SkillBridge EdTech - Agentic RAG Challenge Solution

Welcome to the **SkillBridge EdTech Agentic RAG** project repository. This project implements a production-grade Agentic RAG pipeline designed to reason over structured student academic records and unstructured institutional policy documents to answer complex academic queries.

---

## 📁 Repository Structure

```
Rag-agent/
├── .env                       # Local API keys (never commit or paste into Kaggle)
├── .env.example               # Template for environment variables
├── .gitignore                 # Excludes data, keys, cache, and logs
├── requirements.txt           # faiss-cpu, groq, google-genai, pandas, numpy, etc.
├── data/                      # Local copy of Kaggle CSVs & policy docs (gitignored)
│   ├── policies/              # Policy .md / .txt / .pdf files
│   │   └── academic_policies.md
│   ├── students.csv           # Student enrollment & profile records
│   ├── courses.csv            # Course catalog & instructor metadata
│   ├── attendance.csv         # Lecture & lab attendance logs
│   ├── assessments.csv        # Scores (quizzes, midterms, finals, GPA)
│   ├── train.csv              # Labeled training queries with ground truth
│   └── test.csv               # Test evaluation queries
├── src/
│   ├── __init__.py
│   ├── config.py              # Reference date (2025-01-15), paths, Groq key rotation
│   ├── vector_store.py        # Cell 6 logic: chunking, Gemini embeddings, FAISS index
│   ├── tools.py               # Cell 8 logic: CSV/date/math tools + OpenAI schemas
│   └── agent.py               # Cell 9 logic: system prompt + tool-calling loop
├── run_eval.py                # Cell 10 equivalent: test on train.csv & log errors
├── generate_submission.py     # Cell 11–12 equivalent: build & validate submission.csv
├── .agents/
│   └── skills/
│       └── skillbridge-rag/
│           └── SKILL.md       # Project guidelines and rules
└── README.md
```

---

## 🎯 The Problem & Mission

SkillBridge EdTech manages 800+ students across 20 courses. Queries about academic standing, attendance warnings, and grades cannot be solved with naive lookup or standalone LLMs because:
1. **Data Lookup**: Data is partitioned across multiple structured CSVs (`students`, `courses`, `attendance`, `assessments`).
2. **Rule-Based Reasoning**: Academic policies dictate exact formulas (e.g. weighted attendance: 40% lecture / 60% lab, composite scores: 30% continuous / 30% midterm / 40% final, alert probation triggers).
3. **Out-of-Scope Detection**: Extraneous questions (fees, dorm rooms, sports, contact details) must be classified as unanswerable with `answerable = 0`.
4. **Anchor Date Constraint**: All date calculations are strictly anchored to **January 15, 2025** (`2025-01-15`).

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate into the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and fill in your keys:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY_1=gsk_your_first_groq_key
GROQ_API_KEY_2=gsk_your_second_groq_key
```
> **Key Rotation**: The pipeline automatically detects all `GROQ_API_KEY*` variables and seamlessly rotates keys when encountering rate limits (HTTP 429).

### 3. Run Evaluation (Cell 10 Equivalent)
Run the pipeline against the labeled training set:
```bash
python run_eval.py
```
This prints an evaluation summary table detailing Answerable Accuracy and Exact/Numeric Match percentages, and saves detailed error logs to `logs/eval_results.json`.

### 4. Generate & Validate Submission (Cell 11–12 Equivalent)
Process the test queries and generate `submission.csv`:
```bash
python generate_submission.py
```
This automatically runs all Kaggle validation checks:
- Verifies row count matches `test.csv`.
- Confirms columns: `query_id`, `answerable`, `answer`.
- Asserts `answerable` contains strictly binary values (0 or 1).
- Checks for zero null / NaN values.

---

## 🧩 Kaggle Starter Notebook Cell Mapping

If working directly in Kaggle, copy code from this repository into the corresponding notebook cells:

| Kaggle Cell | Corresponding Local File | What It Does |
|-------------|--------------------------|--------------|
| **Cell 6**  | `src/vector_store.py`    | Document chunking, Gemini embeddings (`text-embedding-004`), and FAISS index retrieval |
| **Cell 8**  | `src/tools.py`           | Tools for CSV lookup, date calculation, weighted attendance, grades, and OpenAI tool schemas |
| **Cell 9**  | `src/agent.py`           | System prompt, multi-turn tool calling loop with Groq key rotation |
| **Cell 10** | `run_eval.py`            | Test agent on `train.csv`, compute accuracy metrics, and log errors |
| **Cell 11** | `generate_submission.py` | Run agent on `test.csv` to generate `submission.csv` |
| **Cell 12** | `generate_submission.py` | Validate submission format before uploading to Kaggle |

---

## 📝 Key Formulas Implemented

- **Reference Date**: `2025-01-15`
- **Elapsed Weeks**:
  $$\text{Weeks} = \lfloor \frac{\text{Date(2025-01-15)} - D_{\text{start}}}{7} \rfloor$$
- **Weighted Attendance**:
  $$\text{Weighted Attendance \%} = (\text{Lecture Attendance \%} \times 0.40) + (\text{Lab Attendance \%} \times 0.60)$$
- **Composite Course Grade**:
  $$\text{Composite Score} = (\text{Continuous} \times 0.30) + (\text{Midterm} \times 0.30) + (\text{Final Exam} \times 0.40)$$
- **Dual Passing Rule**: Composite Score $\ge 50.0\%$ AND Final Exam $\ge 40.0\%$.
