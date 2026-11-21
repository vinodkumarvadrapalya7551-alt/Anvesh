"""
Submission Generator and Validator for SkillBridge EdTech Challenge (Cell 11–12 equivalent).
Processes test.csv through the Agentic RAG pipeline, constructs submission.csv,
and validates format, column names, nulls, and answerable constraints.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"Starting {desc}...")
        return iterable


from src.agent import SkillBridgeAgent
from src.config import DATA_DIR, ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("submission_gen")


def validate_submission_file(submission_path: Path, expected_test_path: Path) -> bool:
    """
    Validates submission.csv format against Kaggle rules (Cell 12 equivalent):
    1. File exists and is non-empty.
    2. Required columns present: ['query_id', 'answerable', 'answer'].
    3. Row count exactly matches test.csv.
    4. query_id sequence matches test.csv exactly.
    5. 'answerable' column contains only binary integers (0 or 1).
    6. No null / NaN values in any column.
    """
    print("\n" + "=" * 70)
    print("RUNNING SUBMISSION VALIDATION CHECKS (Cell 12 Equivalent)")
    print("=" * 70)

    if not submission_path.exists():
        logger.error(f"[FAIL] Submission file {submission_path} does not exist.")
        return False

    sub_df = pd.read_csv(submission_path)
    test_df = pd.read_csv(expected_test_path)

    passed = True

    # 1. Row count check
    if len(sub_df) != len(test_df):
        logger.error(f"[FAIL] Row count mismatch: Submission has {len(sub_df)}, test.csv has {len(test_df)}")
        passed = False
    else:
        logger.info(f"[PASS] Row count matches test.csv ({len(sub_df)} rows).")

    # 2. Columns check
    required_cols = {"query_id", "answerable", "answer"}
    sub_cols = set(sub_df.columns)
    if not required_cols.issubset(sub_cols):
        logger.error(f"[FAIL] Missing required columns. Expected {required_cols}, found {list(sub_df.columns)}")
        passed = False
    else:
        logger.info(f"[PASS] Required columns present: {list(sub_df.columns)}")

    # 3. Query ID alignment check
    if list(sub_df["query_id"]) != list(test_df["query_id"]):
        logger.error("[FAIL] 'query_id' values or order do not match test.csv!")
        passed = False
    else:
        logger.info("[PASS] 'query_id' sequence perfectly aligns with test.csv.")

    # 4. Null value check
    null_counts = sub_df.isnull().sum().to_dict()
    has_nulls = any(cnt > 0 for cnt in null_counts.values())
    if has_nulls:
        logger.error(f"[FAIL] Null values found in submission: {null_counts}")
        passed = False
    else:
        logger.info("[PASS] No null or NaN values found.")

    # 5. Answerable binary check
    valid_ans_values = set(sub_df["answerable"].unique())
    if not valid_ans_values.issubset({0, 1}):
        logger.error(f"[FAIL] 'answerable' must only contain 0 or 1. Found values: {valid_ans_values}")
        passed = False
    else:
        logger.info(f"[PASS] 'answerable' column contains only valid binary flags: {valid_ans_values}")

    # 6. Consistency check for answerable = 0
    out_of_scope_rows = sub_df[sub_df["answerable"] == 0]
    logger.info(f"[INFO] Out-of-scope queries flagged: {len(out_of_scope_rows)}/{len(sub_df)}")

    if passed:
        print("\n" + "=" * 70)
        print("ALL SUBMISSION VALIDATION CHECKS PASSED SUCCESSFULLY! READY TO SUBMIT.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("SUBMISSION VALIDATION FAILED! Please review the error logs above.")
        print("=" * 70)

    return passed


def generate_submission(
    test_csv_path: Path = DATA_DIR / "test.csv",
    output_submission_path: Path = ROOT_DIR / "submission.csv",
):
    """
    Processes test.csv and outputs validated submission.csv (Cell 11 equivalent).
    """
    if not test_csv_path.exists():
        logger.error(f"test.csv not found at {test_csv_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df)} test queries from {test_csv_path.name}")

    agent = SkillBridgeAgent()

    submission_records: List[dict] = []

    print("\n" + "=" * 70)
    print("GENERATING SUBMISSION FILE (Cell 11 Equivalent)")
    print("=" * 70)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Queries"):
        q_id = str(row.get("query_id", f"Q_{idx}"))
        query = str(row.get("query", ""))

        resp = agent.answer_query(query=query, query_id=q_id)

        # Standardize answer format
        final_answer = str(resp.answer).strip()
        if resp.answerable == 0:
            final_answer = "Out-of-scope"

        submission_records.append({
            "query_id": q_id,
            "answerable": int(resp.answerable),
            "answer": final_answer,
        })

    sub_df = pd.DataFrame(submission_records)
    sub_df.to_csv(output_submission_path, index=False)
    logger.info(f"Saved preliminary submission file to {output_submission_path}")

    # Run Cell 12 validation
    validate_submission_file(output_submission_path, test_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Validate Kaggle Submission")
    parser.add_argument("--test-csv", type=str, default=str(DATA_DIR / "test.csv"), help="Path to test.csv")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "submission.csv"), help="Path to output submission.csv")
    args = parser.parse_args()

    generate_submission(test_csv_path=Path(args.test_csv), output_submission_path=Path(args.output))
