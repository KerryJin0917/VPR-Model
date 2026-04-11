#!/usr/bin/env python3
"""
COMP560 Visual Place Recognition Evaluation Script

Evaluates student-submitted prediction files against ground truth.
Students submit a CSV file with ranked database indices for each query.

Submission format (CSV):
    query_index,ranked_database_indices
    0,"45,12,78,3,99,..."
    1,"102,5,67,23,11,..."
    ...

- query_index: integer index of the query (0-based, matching sorted query image order)
- ranked_database_indices: comma-separated database indices sorted by similarity (most similar first)
  At least top-20 indices should be provided.

Usage:
    python evaluate.py --student_id <your_id> --prediction <rankings.csv>
    python evaluate.py --student_id <your_id> --prediction <rankings.csv> --datasets dataset_a dataset_b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_recall_at_k(
    predictions: np.ndarray,
    positives_per_query: List[np.ndarray],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Compute Recall@K metrics."""
    recalls = {}
    num_queries = len(positives_per_query)

    for k in k_values:
        correct = 0
        for q_idx, positives in enumerate(positives_per_query):
            if len(positives) > 0 and q_idx < predictions.shape[0]:
                preds_at_k = predictions[q_idx, :k]
                if np.any(np.isin(preds_at_k, positives)):
                    correct += 1
        recalls[f"R@{k}"] = (correct / num_queries) * 100

    return recalls


def compute_map_at_k(
    predictions: np.ndarray,
    positives_per_query: List[np.ndarray],
    k: int = 20,
) -> float:
    """Compute Mean Average Precision at K."""
    aps = []
    for q_idx, positives in enumerate(positives_per_query):
        if len(positives) == 0 or q_idx >= predictions.shape[0]:
            continue

        preds = predictions[q_idx, :k]
        score = 0.0
        num_hits = 0.0

        for i, pred in enumerate(preds):
            if pred in positives:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        aps.append(score / min(len(positives), k))

    return np.mean(aps) * 100 if aps else 0.0


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset_a_gt(root: str):
    """Load dataset_a ground truth: 1-to-1 index matching."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))

    db_df = df[df["split"] == "database"].sort_values("image_path")
    q_df = df[df["split"] == "queries"].sort_values("image_path")

    n_db = len(db_df)
    n_q = len(q_df)

    # 1-to-1 matching: query[i] matches database[i]
    positives_per_query = [np.array([i]) for i in range(n_q)]

    return positives_per_query, n_db, n_q


def load_dataset_b_gt(root: str, positive_threshold: float = 25.0):
    """Load dataset_b ground truth: GPS-based matching."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))

    db_df = df[df["role"] == "database"]
    q_df = df[df["role"] == "queries"]

    db_coords = db_df[["utm_east", "utm_north"]].values
    q_coords = q_df[["utm_east", "utm_north"]].values

    n_db = len(db_df)
    n_q = len(q_df)

    # Compute positives based on GPS distance
    positives_per_query = []
    for q_coord in q_coords:
        distances = np.linalg.norm(db_coords - q_coord, axis=1)
        pos_indices = np.where(distances <= positive_threshold)[0]
        positives_per_query.append(pos_indices)

    return positives_per_query, n_db, n_q


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_dataset(
    prediction_path: str,
    dataset_root: str,
    dataset_name: str,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict:
    """Evaluate predictions against ground truth."""

    if dataset_name == "dataset_a":
        positives_per_query, n_db, n_q = load_dataset_a_gt(dataset_root)
    else:
        positives_per_query, n_db, n_q = load_dataset_b_gt(dataset_root)

    # Load predictions
    pred_df = pd.read_csv(prediction_path)
    required_cols = {"query_index", "ranked_database_indices"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV must have columns: {required_cols}. Got: {set(pred_df.columns)}")

    max_k = max(k_values)
    predictions = np.zeros((n_q, max_k), dtype=np.int64)

    for _, row in pred_df.iterrows():
        q_idx = int(row["query_index"])
        if q_idx >= n_q:
            continue
        indices_str = str(row["ranked_database_indices"]).strip('"').strip("'")
        indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        length = min(len(indices), max_k)
        predictions[q_idx, :length] = indices[:length]

    predicted_queries = set(pred_df["query_index"].astype(int))
    missing = n_q - len(predicted_queries.intersection(range(n_q)))
    if missing > 0:
        print(f"  WARNING: {missing}/{n_q} queries have no prediction")

    # Compute metrics
    recalls = compute_recall_at_k(predictions, positives_per_query, k_values)
    map_at_k = compute_map_at_k(predictions, positives_per_query, k=max_k)

    results = {
        "performance": {
            **recalls,
            f"mAP@{max_k}": map_at_k,
        },
        "submission_info": {
            "num_database": n_db,
            "num_queries": n_q,
            "num_predicted_queries": len(predicted_queries.intersection(range(n_q))),
            "num_missing_queries": missing,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="COMP560 Visual Place Recognition Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student_id", type=str, required=True, help="Your student ID")
    parser.add_argument(
        "--prediction", type=str, required=True,
        help="Path to prediction CSV (or directory with dataset_a.csv, dataset_b.csv)",
    )
    parser.add_argument("--datasets_root", type=str, default="./datasets", help="Root directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["dataset_a"],
        choices=["dataset_a", "dataset_b"],
        help="Datasets to evaluate on",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("COMP560 Visual Place Recognition Evaluation")
    print("=" * 60)
    print(f"Student ID: {args.student_id}")
    print(f"Prediction: {args.prediction}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    all_results = {
        "student_id": args.student_id,
        "timestamp": timestamp,
        "prediction_path": args.prediction,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print("=" * 60)

        dataset_root = os.path.join(args.datasets_root, dataset_name)
        if not os.path.exists(os.path.join(dataset_root, "test.parquet")):
            print(f"  ERROR: Ground truth not found at {dataset_root}")
            all_results["datasets"][dataset_name] = {"error": "ground truth not found"}
            continue

        if os.path.isdir(args.prediction):
            pred_path = os.path.join(args.prediction, f"{dataset_name}.csv")
        else:
            pred_path = args.prediction

        if not os.path.exists(pred_path):
            print(f"  ERROR: Prediction file not found at {pred_path}")
            all_results["datasets"][dataset_name] = {"error": "prediction file not found"}
            continue

        try:
            results = evaluate_dataset(pred_path, dataset_root, dataset_name)
            all_results["datasets"][dataset_name] = results

            print(f"\nPerformance Metrics:")
            for metric, value in results["performance"].items():
                print(f"  {metric}: {value:.2f}%")

            print(f"\nSubmission Info:")
            for key, value in results["submission_info"].items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][dataset_name] = {"error": str(e)}

    output_file = output_dir / f"{args.student_id}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    summary_file = output_dir / f"{args.student_id}_{timestamp}_summary.csv"
    with open(summary_file, "w") as f:
        f.write("dataset,R@1,R@5,R@10,R@20,mAP@20\n")
        for dataset_name, results in all_results["datasets"].items():
            if "error" not in results:
                perf = results["performance"]
                f.write(f"{dataset_name},{perf.get('R@1', 0):.2f},{perf.get('R@5', 0):.2f},"
                       f"{perf.get('R@10', 0):.2f},{perf.get('R@20', 0):.2f},"
                       f"{perf.get('mAP@20', 0):.2f}\n")
    print(f"Summary saved to: {summary_file}")

    return all_results


if __name__ == "__main__":
    main()
