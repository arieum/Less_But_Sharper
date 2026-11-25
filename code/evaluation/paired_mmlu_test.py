#!/usr/bin/env python3
"""Question-level paired tests for MMLU evaluator outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_rel


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return {
            row.get("pair_uid") or row.get("doc_hash"): int(float(row["correct"]))
            for row in rows
            if row.get("correct") not in (None, "", "None")
        }

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return {str(key): int(score) for key, score in value.items()}
    if isinstance(value, list):
        records = {}
        for index, row in enumerate(value):
            key = row.get("pair_uid") or row.get("doc_hash") or str(index)
            records[key] = int(row["correct"])
        return records
    raise TypeError(f"Unsupported content in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Baseline CSV or JSON")
    parser.add_argument(
        "--conditions", nargs="+", required=True, help="Condition CSV/JSON files"
    )
    args = parser.parse_args()

    baseline = load_records(args.base)
    for condition_path in args.conditions:
        condition = load_records(condition_path)
        if set(baseline) != set(condition):
            raise ValueError(
                f"{condition_path}: expected the same question IDs as {args.base}"
            )
        keys = sorted(baseline)
        base_values = np.array([baseline[key] for key in keys], dtype=int)
        condition_values = np.array([condition[key] for key in keys], dtype=int)
        differences = condition_values - base_values
        base_only = int(np.sum((base_values == 1) & (condition_values == 0)))
        condition_only = int(np.sum((base_values == 0) & (condition_values == 1)))
        discordant = base_only + condition_only
        mcnemar_p = (
            1.0
            if discordant == 0
            else float(binomtest(condition_only, discordant, p=0.5).pvalue)
        )
        t_result = ttest_rel(condition_values, base_values)
        print(
            json.dumps(
                {
                    "condition": condition_path,
                    "n": len(keys),
                    "accuracy_difference": float(differences.mean()),
                    "condition_only_correct": condition_only,
                    "baseline_only_correct": base_only,
                    "paired_t": float(t_result.statistic),
                    "paired_t_p": float(t_result.pvalue),
                    "mcnemar_exact_p": mcnemar_p,
                }
            )
        )


if __name__ == "__main__":
    main()
