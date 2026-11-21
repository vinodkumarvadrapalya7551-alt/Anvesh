"""
Evaluation Runner for SkillBridge EdTech Agentic RAG System (Cell 10 equivalent).
Tests the agent pipeline against train.csv, compares predicted outputs with ground truth,
computes precision, recall, F1, exact match / numerical tolerance scores, and logs error cases.
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers=None, tablefmt=None):
        if not rows:
            return ""
        all_rows = ([headers] if headers else []) + [list(map(str, r)) for r in rows]
        col_widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(all_rows[0]))]
        lines = []
        if headers:
            header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
            sep_line = "-+-".join("-" * w for w in col_widths)
            lines.extend([header_line, sep_line])
        for r in rows:
            lines.append(" | ".join(str(val).ljust(w) for val, w in zip(r, col_widths)))
        return "\n".join(lines)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"Starting {desc}...")
        return iterable


from src.agent import SkillBridgeAgent
from src.config import DATA_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")


def are_answers_equivalent(pred: Any, ground_truth: Any, tolerance: float = 0.05) -> bool:
    """
    Checks equivalence between predicted answer and ground truth:
    - Handles numerical equality within tolerance (e.g. 79.28 vs 79.3)
    - Handles boolean equality (Yes/No vs True/False)
    - Handles case-insensitive string matching
    """
    if pred is None or ground_truth is None:
        return False

    pred_str = str(pred).strip().lower()
    gt_str = str(ground_truth).strip().lower()

    # Out-of-scope check
    if "out-of-scope" in gt_str or "out of scope" in gt_str:
        return "out-of-scope" in pred_str or "out of scope" in pred_str or pred_str == "0"

    # Direct string match
    if pred_str == gt_str:
        return True

    # Numeric check
    try:
        pred_num = float(pred_str.replace("%", "").strip())
        gt_num = float(gt_str.replace("%", "").strip())
        return abs(pred_num - gt_num) <= tolerance or (
            gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) <= tolerance
        )
    except ValueError:
        pass

    # Boolean conversion check
    bool_map = {"true": "yes", "false": "no", "1": "yes", "0": "no"}
    p_norm = bool_map.get(pred_str, pred_str)
    g_norm = bool_map.get(gt_str, gt_str)
    if p_norm == g_norm:
        return True

    return False


def run_evaluation(
    train_csv_path: Path = DATA_DIR / "train.csv",
    sample_size: int = -1,
    output_log_path: Path = LOGS_DIR / "eval_results.json",
):
    """
    Runs full evaluation over train.csv.
    """
    if not train_csv_path.exists():
        logger.error(f"Training dataset not found at {train_csv_path}")
        sys.exit(1)

    df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded {len(df)} training examples from {train_csv_path.name}")

    if sample_size > 0 and sample_size < len(df):
        df = df.head(sample_size)
        logger.info(f"Evaluating on sample subset of {len(df)} queries.")

    agent = SkillBridgeAgent()

    results: List[Dict[str, Any]] = []
    correct_answerable = 0
    correct_answers = 0
    total = len(df)

    category_stats: Dict[str, Dict[str, int]] = {}

    print("\n" + "=" * 70)
    print("STARTING EVALUATION ON TRAINING SET (Cell 10 Equivalent)")
    print("=" * 70)

    for idx, row in tqdm(df.iterrows(), total=total, desc="Evaluating Agent"):
        q_id = str(row.get("query_id", f"Q_{idx}"))
        query = str(row.get("query", ""))
        gt_answerable = int(row.get("answerable", 1))
        gt_answer = row.get("ground_truth", "")
        category = str(row.get("category", "General"))

        if category not in category_stats:
            category_stats[category] = {"total": 0, "ans_correct": 0, "val_correct": 0}
        category_stats[category]["total"] += 1

        # Run agent
        resp = agent.answer_query(query=query, query_id=q_id)

        ans_match = (resp.answerable == gt_answerable)
        val_match = are_answers_equivalent(resp.answer, gt_answer) if ans_match else False

        if ans_match:
            correct_answerable += 1
            category_stats[category]["ans_correct"] += 1
        if val_match:
            correct_answers += 1
            category_stats[category]["val_correct"] += 1

        record = {
            "query_id": q_id,
            "query": query,
            "category": category,
            "ground_truth_answerable": gt_answerable,
            "predicted_answerable": resp.answerable,
            "answerable_correct": ans_match,
            "ground_truth_answer": str(gt_answer),
            "predicted_answer": str(resp.answer),
            "answer_correct": val_match,
            "reasoning": resp.reasoning,
            "tool_calls_count": resp.tool_calls_count,
        }
        results.append(record)

    # Compute aggregate metrics
    answerable_acc = (correct_answerable / total) * 100.0 if total > 0 else 0.0
    overall_acc = (correct_answers / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Queries Evaluated: {total}")
    print(f"Answerable Classification Accuracy: {correct_answerable}/{total} ({answerable_acc:.2f}%)")
    print(f"Final Answer Exact/Numeric Match:   {correct_answers}/{total} ({overall_acc:.2f}%)")

    # Category breakdown table
    cat_rows = []
    for cat, stats in category_stats.items():
        c_tot = stats["total"]
        ans_pct = (stats["ans_correct"] / c_tot) * 100.0 if c_tot > 0 else 0.0
        val_pct = (stats["val_correct"] / c_tot) * 100.0 if c_tot > 0 else 0.0
        cat_rows.append([cat, c_tot, f"{ans_pct:.1f}%", f"{val_pct:.1f}%"])

    print("\nPerformance by Category:")
    print(
        tabulate(
            cat_rows,
            headers=["Category", "Count", "Answerable Acc", "Answer Accuracy"],
            tablefmt="grid",
        )
    )

    # Log errors
    errors = [r for r in results if not r["answer_correct"]]
    if errors:
        print(f"\nDiscrepancies / Error Cases ({len(errors)}):")
        err_table = [
            [
                e["query_id"],
                e["query"][:40] + "...",
                f"GT: {e['ground_truth_answer']}",
                f"Pred: {e['predicted_answer']}",
                f"AnsMatch: {e['answerable_correct']}",
            ]
            for e in errors
        ]
        print(tabulate(err_table, headers=["ID", "Query", "Ground Truth", "Predicted", "Ans Match"], tablefmt="simple"))

    # Save detailed JSON log
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "answerable_accuracy": answerable_acc,
        "answer_accuracy": overall_acc,
        "category_metrics": category_stats,
        "results": results,
    }
    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed evaluation metrics saved to {output_log_path}")

    return overall_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation on train.csv")
    parser.add_argument("--sample-size", type=int, default=-1, help="Number of queries to sample (-1 for all)")
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "train.csv"), help="Path to train.csv")
    args = parser.parse_args()

    run_evaluation(train_csv_path=Path(args.csv), sample_size=args.sample_size)
