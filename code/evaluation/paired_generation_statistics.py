#!/usr/bin/env python3
"""Paired Top-K/MMR tests for generation outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


LANGS = ["zh", "fr", "es", "vi", "id", "ja"]


def load_items(path):
    with open(path, encoding="utf-8") as handle:
        try:
            value = json.load(handle)
            return value if isinstance(value, list) else [value]
        except json.JSONDecodeError:
            handle.seek(0)
            return [json.loads(line) for line in handle if line.strip()]


def keyed(items, path):
    result = {}
    for item in items:
        key = item.get("question") or item.get("input")
        if not key:
            raise ValueError(f"Missing question/input key in {path}")
        if key in result:
            raise ValueError(f"Duplicate prompt in {path}: {key[:80]!r}")
        result[key] = item
    return result


def judge_score(item):
    score = item.get("content_score", item.get("judge_score"))
    if score is None or float(score) < 0:
        raise ValueError("Missing or failed content/judge score")
    return float(score)


def bootstrap_ci(differences, rng, samples):
    differences = np.asarray(differences, dtype=float)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[draws].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance tests for Top-K versus MMR generation."
    )
    parser.add_argument("--topk_template", required=True, help="Path containing {lang}")
    parser.add_argument("--mmr_template", required=True, help="Path containing {lang}")
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []

    for lang in args.langs:
        topk_path = args.topk_template.format(lang=lang)
        mmr_path = args.mmr_template.format(lang=lang)
        topk = keyed(load_items(topk_path), topk_path)
        mmr = keyed(load_items(mmr_path), mmr_path)
        if set(topk) != set(mmr):
            raise ValueError(f"{lang}: methods do not contain identical prompts")

        prompts = sorted(topk)
        topk_lid = np.array([int(topk[q]["lid_match"]) for q in prompts])
        mmr_lid = np.array([int(mmr[q]["lid_match"]) for q in prompts])
        topk_quality = np.array([judge_score(topk[q]) for q in prompts])
        mmr_quality = np.array([judge_score(mmr[q]) for q in prompts])

        lid_diff = mmr_lid - topk_lid
        quality_diff = mmr_quality - topk_quality
        topk_only = int(np.sum((topk_lid == 1) & (mmr_lid == 0)))
        mmr_only = int(np.sum((topk_lid == 0) & (mmr_lid == 1)))
        discordant = topk_only + mmr_only
        mcnemar_p = (
            1.0
            if discordant == 0
            else float(binomtest(mmr_only, discordant, p=0.5).pvalue)
        )
        wilcoxon_p = (
            1.0
            if np.allclose(quality_diff, 0)
            else float(wilcoxon(quality_diff, zero_method="wilcox").pvalue)
        )

        results.append(
            {
                "language": lang,
                "n": len(prompts),
                "topk_language_accuracy": float(topk_lid.mean()),
                "mmr_language_accuracy": float(mmr_lid.mean()),
                "language_accuracy_difference": float(lid_diff.mean()),
                "language_accuracy_ci95": bootstrap_ci(lid_diff, rng, args.bootstrap),
                "mcnemar_p": mcnemar_p,
                "topk_content_quality": float(topk_quality.mean()),
                "mmr_content_quality": float(mmr_quality.mean()),
                "content_quality_difference": float(quality_diff.mean()),
                "content_quality_ci95": bootstrap_ci(
                    quality_diff, rng, args.bootstrap
                ),
                "wilcoxon_p": wilcoxon_p,
            }
        )

    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
