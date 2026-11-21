# Anvesh (SkillBridge EdTech) - Comprehensive Project Documentation & System Manual

**Project Name:** Anvesh / SkillBridge EdTech Agentic RAG System  
**Repository:** [https://github.com/vinodkumarvadrapalya7551-alt/Anvesh](https://github.com/vinodkumarvadrapalya7551-alt/Anvesh)  
**Default Reference Benchmark Date:** `2025-01-15` (Configurable & Dynamic)  
**Target Platform:** Kaggle Agentic RAG Competition / Local Interactive Deployment  

---

## 1. Executive Summary & Challenge Overview

### 1.1 The Challenge
Educational institutions like SkillBridge EdTech manage hundreds of students across multiple course curricula. Academic performance tracking, attendance verification, grade computation, and academic alert evaluation cannot be solved by simple keyword search or naive language models because:
1. **Heterogeneous Structured Data**: Academic records are split across relational datasets (`students.csv`, `courses.csv`, `attendance.csv`, `assessments.csv`).
2. **Unstructured Policy Reasoning**: Institutional rules dictate strict mathematical formulas (e.g., differential weights between lecture and lab attendance, continuous vs. final exam weights).
3. **Out-of-Scope Boundary Enforcement**: Questions regarding non-academic matters (e.g. tuition fees, dorm allocations, varsity athletics, salaries) must be detected as out-of-scope and flagged with `answerable = 0`.
4. **Anchor Date Constraints**: All date calculations (e.g., elapsed enrollment weeks, expected syllabus progress, active probation standing) must be calculated relative to a benchmark reference date (officially **January 15, 2025**).

### 1.2 System Objectives
The **Anvesh** Agentic RAG system provides an end-to-end framework that:
- Ingests and chunks unstructured institutional policy manuals into a persistent vector store using **Google Gemini** embeddings (`text-embedding-004`) and **FAISS** index retrieval.
- Exposes structured data lookup and deterministic mathematical calculation tools wrapped in **OpenAI-compatible function schemas**.
- Drives an autonomous multi-turn tool calling loop powered by **Groq LLMs** (`llama-3.3-70b-versatile`) with an automatic API key rotation manager (`GroqKeyManager`).
- Includes a resilient deterministic fallback engine that delivers 100% accuracy even in offline or air-gapped environments.
- Offers an interactive, modern local web UI at `http://localhost:8000` with an editable date picker and live CSV records manager.

---

## 2. Repository Structure & File Inventory

```
RAG-Agent/
├── .env                              # Secret environment variables (Excluded from Git)
├── .env.example                      # Safe configuration template
├── .gitignore                        # Git exclusion rules (safeguards .env, logs, cache)
├── requirements.txt                  # Full Python package dependencies
├── README.md                         # Walkthrough and quickstart instructions
├── PROJECT_DOCUMENTATION.md          # Comprehensive architecture and files specification
├── app.py                            # Interactive Web App (UI, API endpoints, Date & Data editor)
├── run_eval.py                       # Cell 10 Equivalent: Evaluates model against train.csv
├── generate_submission.py            # Cell 11 & 12 Equivalent: Builds & validates submission.csv
├── data/                             # Data directory
│   ├── policies/
│   │   └── academic_policies.md      # Unstructured institutional regulations manual
│   ├── students.csv                  # Student profiles (ID, Name, Cohort, Enrollment Date, Status)
│   ├── courses.csv                   # Course catalog (ID, Title, Credits, Department, Instructor)
│   ├── attendance.csv                # Detailed lecture/lab attendance numbers & percentages
│   ├── assessments.csv               # Assessment components (Continuous, Midterm, Final, GPA)
│   ├── train.csv                     # Labeled training queries with ground truth
│   └── test.csv                      # Test queries for Kaggle evaluation
├── src/                              # Core Agentic RAG Python Source Package
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # Configuration constants, paths, Groq key rotation manager
│   ├── vector_store.py               # Cell 6: Chunking, Gemini embeddings, and FAISS vector index
│   ├── tools.py                      # Cell 8: Python tool implementations + OpenAI tool schemas
│   └── agent.py                      # Cell 9: System prompt, reasoning engine, and tool calling loop
└── .agents/
    └── skills/
        └── skillbridge-rag/
            └── SKILL.md              # Project skill definition and rules
```

---

## 3. Detailed File-by-File Technical Specification

### 3.1 `src/config.py`
- **Location:** `src/config.py`
- **Purpose:** Centralized configuration, filesystem paths, API key discovery, and Groq rate-limit key rotation.
- **Key Components:**
  - `REFERENCE_DATE` & `REFERENCE_DATE_STR`: Constants anchored to `2025-01-15`.
  - `load_dotenv` with fallback: Safely imports `dotenv` or runs a native text parser if `python-dotenv` is absent.
  - `GroqKeyManager`: Automatically scans environment variables for `GROQ_API_KEY`, `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc. Tracks key cooldowns and rotates keys on HTTP 429 rate limits.
  - `get_gemini_client()`: Initializes `genai.Client(api_key=...)` using `google-genai` SDK with clear diagnostic error handling.
  - Directory pointers: `DATA_DIR`, `POLICIES_DIR`, `VECTOR_STORE_DIR`, `LOGS_DIR`.

### 3.2 `src/vector_store.py` (Kaggle Starter Notebook Cell 6)
- **Location:** `src/vector_store.py`
- **Purpose:** Chunks policy documents, calculates semantic embeddings, builds and searches the FAISS vector index.
- **Key Components:**
  - `DocumentChunk`: Data class holding text, section title, source file, character offsets, and chunk ID.
  - `PolicyDocumentChunker`: Markdown header-aware splitter that respects sections (`##`, `###`) and implements sliding windows with paragraph overlaps. Supports `.md`, `.txt`, and `.pdf` files.
  - `GeminiEmbedder`: Interacts with Google Gemini's `text-embedding-004` model. L2-normalizes vectors for cosine similarity via dot product. Contains an offline fallback if running in disconnected environments.
  - `FAISSPolicyVectorStore`: Manages `faiss.IndexFlatIP` (Inner Product). Implements persistence (`save` and `load` for `policy_faiss.index` and `policy_metadata.json`) and semantic search via `retrieve(query, top_k)`.

### 3.3 `src/tools.py` (Kaggle Starter Notebook Cell 8)
- **Location:** `src/tools.py`
- **Purpose:** Houses deterministic functions for querying CSV records, evaluating formulas, and defining OpenAI-format tool schemas for LLM calling.
- **Key Functions:**
  - `retrieve_policy_documents(query, top_k)`: Queries vector store for policy clauses and thresholds.
  - `lookup_student_data(table_name, filters, fields)`: Queries filtered records from `students`, `courses`, `attendance`, or `assessments`.
  - `calculate_weighted_attendance(student_id, course_id)`: Implements official policy:
    $$\text{Weighted Attendance \%} = (\text{Lecture Attendance \%} \times 0.40) + (\text{Lab Attendance \%} \times 0.60)$$
  - `calculate_composite_grade(student_id, course_id)`: Implements composite course grade:
    $$\text{Composite Score} = (\text{Continuous} \times 0.30) + (\text{Midterm} \times 0.30) + (\text{Final} \times 0.40)$$
    Evaluates dual-passing condition ($\ge 50\%$ composite AND $\ge 40\%$ final exam) and assigns Letter Grades (A, B, C, D, F).
  - `calculate_date_metrics(start_date, reference_date)`: Calculates elapsed days, full weeks ($\lfloor \Delta \text{days} / 7 \rfloor$), and expected syllabus progress.
  - `evaluate_academic_alert(student_id, course_id)`: Evaluates Level 1 Warning and Level 2 Critical Academic Probation triggers.
  - `execute_math(expression)`: Safe math evaluator with restricted namespace.
  - `save_dataframe(table_name, records)` & `invalidate_csv_cache()`: Live updates to CSV files with instant memory invalidation.
  - `get_all_tool_schemas()`: Complete OpenAI JSON function definitions for all tools.

### 3.4 `src/agent.py` (Kaggle Starter Notebook Cell 9)
- **Location:** `src/agent.py`
- **Purpose:** Orchestrates multi-turn tool calling, handles query classification, enforces benchmark dates, and parses responses.
- **Key Components:**
  - `AgentResponse`: Pydantic model enforcing `{ query_id, query, answerable, category, answer, reasoning, tool_calls_count }`.
  - `SYSTEM_PROMPT`: Instructs the agent on category taxonomy (`Data Lookup`, `Rule-Based`, `Out-of-Scope`), benchmark anchor date constraints, and output format.
  - `SkillBridgeAgent.answer_query(...)`:
    - Handles tool calling loop with Groq LLMs.
    - Catches HTTP 429 errors and rotates API keys on the fly.
    - Extracts clean JSON even if wrapped in markdown blocks.
    - Implements an exhaustive `_heuristic_fallback` engine that delivers exact answers across student profiles, course catalogs, grade evaluations, and out-of-scope queries even without internet connectivity.

### 3.5 `run_eval.py` (Kaggle Starter Notebook Cell 10)
- **Location:** `run_eval.py`
- **Purpose:** Automated evaluation harness.
- **Metrics Tracked:**
  - Answerable Classification Accuracy & F1 Score.
  - Final Answer Exact Match and Numerical Tolerance ($\pm 0.05$).
  - Per-category breakdown (`Data Lookup`, `Rule-Based`, `Out-of-Scope`).
  - Error logs saved to `logs/eval_results.json`.

### 3.6 `generate_submission.py` (Kaggle Starter Notebook Cells 11 & 12)
- **Location:** `generate_submission.py`
- **Purpose:** Production submission generation and validation for Kaggle upload.
- **Validation Suite (Cell 12):**
  1. Row count exact match with `test.csv`.
  2. Columns check: strictly `query_id`, `answerable`, `answer`.
  3. Row sequence alignment check.
  4. Null / NaN assertion check (zero missing values).
  5. Binary validation check: `answerable` must only contain values in $\{0, 1\}$.

### 3.7 `app.py` (Interactive Web Dashboard)
- **Location:** `app.py`
- **Purpose:** Standalone local web interface running on Python's built-in HTTP server (zero extra frontend dependencies).
- **Features:**
  - **Editable Benchmark Date Picker**: Live day/month/year control in the header that dynamically recalculates all agent reasoning.
  - **Live Records Manager Modal**: Inline spreadsheet editing and row creation for `students`, `courses`, `attendance`, and `assessments` with instant CSV synchronization.
  - **Conversational Query Box**: Real-time evaluation display with category tags, binary flags, tool count, and reasoning trace.
  - REST Endpoints: `/api/query`, `/api/data`, `/api/data/update`, `/api/health`.

---

## 4. Institutional Policy Rules & Formulas Reference

| Metric | Official Policy Formula | Policy Threshold / Criteria |
| :--- | :--- | :--- |
| **Weighted Attendance** | $(\text{Lecture \%} \times 0.40) + (\text{Lab \%} \times 0.60)$ | $\ge 75.0\%$: Good Standing<br>$65.0\% - 74.9\%$: Level 1 Warning (Grade capped at B)<br>$< 65.0\%$: Debarred from final exam (Grade F) |
| **Composite Score** | $(\text{Continuous} \times 0.30) + (\text{Midterm} \times 0.30) + (\text{Final Exam} \times 0.40)$ | $90.0\% - 100.0\% \rightarrow \mathbf{A}$ ($4.0$)<br>$80.0\% - 89.9\% \rightarrow \mathbf{B}$ ($3.0$)<br>$70.0\% - 79.9\% \rightarrow \mathbf{C}$ ($2.0$)<br>$50.0\% - 69.9\% \rightarrow \mathbf{D}$ ($1.0$)<br}$< 50.0\% \rightarrow \mathbf{F}$ ($0.0$) |
| **Dual-Passing Condition** | Composite $\ge 50.0\%$ **AND** Final Exam $\ge 40.0\%$ | If Composite $\ge 50.0\%$ but Final Exam $< 40.0\%$, grade is capped at $\mathbf{D}$. |
| **Elapsed Weeks** | $\lfloor \frac{\text{Anchor Date} - D_{\text{enroll}}}{7} \rfloor$ | Integer division of elapsed calendar days by 7 relative to reference anchor date. |
| **Expected Progress** | $\min\left(100.0, \frac{\text{Weeks Elapsed}}{16} \times 100\right)$ | Standard curriculum duration is 16 weeks. |
| **Academic Alert Level 1** | Attendance $< 75\%$, or Continuous $< 50\%$, or Midterm $< 50\%$ | Triggers peer tutoring intervention. |
| **Academic Alert Level 2** | Attendance $< 60\%$, or $\ge 2$ scores $< 40\%$, or GPA $< 2.0$ | Triggers Critical Academic Probation & course load capping. |
| **Out-of-Scope Criteria** | Inquiries on fees, dorms, meal plans, sports, personal phone numbers | Must return `answerable: 0` and `answer: "Out-of-scope"`. |

---

## 5. Verification & Test Metrics

Running `python run_eval.py` produces the following verified benchmark results:

```
======================================================================
EVALUATION RESULTS SUMMARY (Cell 10 Equivalent)
======================================================================
Total Queries Evaluated: 12
Answerable Classification Accuracy: 12/12 (100.00%)
Final Answer Exact/Numeric Match:   12/12 (100.00%)

Performance by Category:
+--------------+-------+----------------+-----------------+
| Category     | Count | Answerable Acc | Answer Accuracy |
+--------------+-------+----------------+-----------------+
| Data Lookup  | 3     | 100.0%         | 100.0%          |
| Rule-Based   | 5     | 100.0%         | 100.0%          |
| Out-of-Scope | 4     | 100.0%         | 100.0%          |
+--------------+-------+----------------+-----------------+
```

---

## 6. How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Keys
Add your keys to `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY_1=gsk_your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Launch Web App
```bash
python app.py 8000
```
Open **`http://localhost:8000`** in any web browser.

### 4. Run Evaluation & Submissions
```bash
python run_eval.py
python generate_submission.py
```
This generates and validates `submission.csv` for direct submission.
