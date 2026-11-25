import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]
MMLU_FEWSHOT = 5


def patch_model(model, mask_path, mask_langs=None):
    """Apply neuron pruning to a Hugging Face Llama model."""
    obj = torch.load(mask_path, map_location="cpu")

    if isinstance(obj, dict) and "neuron" in obj:
        activation_masks = obj["neuron"]
    else:
        activation_masks = obj

    if mask_langs is None:
        lang_indices = list(range(len(LANGS)))
    else:
        lang_indices = [LANGS.index(l) for l in mask_langs]

    num_layers = model.config.num_hidden_layers
    merged_masks = [set() for _ in range(num_layers)]

    for lang_idx in lang_indices:
        if lang_idx >= len(activation_masks):
            continue
        per_layer = activation_masks[lang_idx]
        if per_layer is None:
            continue
        for layer_idx, neuron_ids in enumerate(per_layer):
            if neuron_ids is not None and len(neuron_ids) > 0:
                merged_masks[layer_idx].update(neuron_ids.tolist())

    total_pruned = sum(len(s) for s in merged_masks)
    print(f"Total neurons to prune: {total_pruned}")

    for layer_idx in range(num_layers):
        if not merged_masks[layer_idx]:
            continue

        mlp = model.model.layers[layer_idx].mlp
        layer_device = next(mlp.parameters()).device
        mask_tensor = torch.tensor(
            sorted(merged_masks[layer_idx]), dtype=torch.long
        ).to(layer_device)

        def make_forward(mask):
            def forward(self, x):
                gate = self.act_fn(self.gate_proj(x))
                gate.index_fill_(2, mask, 0)
                return self.down_proj(gate * self.up_proj(x))
            return forward

        mlp.forward = MethodType(make_forward(mask_tensor), mlp)

    return total_pruned


def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def normalize_to_str(x):
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def first_present(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def infer_correct(sample: dict):
    """Read correctness from supported lm-eval sample schemas."""
    for key in [
        "acc",
        "acc,none",
        "acc_norm",
        "acc_norm,none",
        "exact_match",
        "correct",
    ]:
        if key in sample and isinstance(sample[key], (int, float, bool)):
            return int(bool(sample[key]))

    metrics = sample.get("metrics")
    if isinstance(metrics, dict):
        for key in [
            "acc",
            "acc,none",
            "acc_norm",
            "acc_norm,none",
            "exact_match",
            "correct",
        ]:
            if key in metrics and isinstance(metrics[key], (int, float, bool)):
                return int(bool(metrics[key]))

    return None


def infer_subject(task_name: str, sample: dict):
    if task_name.startswith("mmlu_"):
        return task_name[len("mmlu_"):]

    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        for key in ["subject", "subtask", "task_name"]:
            if key in doc:
                return str(doc[key])

    return task_name


def infer_doc_id(sample: dict, fallback_idx: int):
    for key in ["doc_id", "id", "sample_id", "idx"]:
        if key in sample:
            return sample[key]
    return fallback_idx


def infer_target(sample: dict):
    for key in ["target", "gold", "answer", "label"]:
        if key in sample:
            return sample[key]

    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        for key in ["target", "gold", "answer", "label"]:
            if key in doc:
                return doc[key]

    return None


def infer_prediction(sample: dict):
    for key in ["prediction", "pred", "resps", "filtered_resps"]:
        if key in sample:
            return sample[key]

    return None


def extract_rows_from_samples(results: dict):
    """Convert MMLU samples to per-question rows."""
    rows = []

    samples_block = results.get("samples", {})
    if not isinstance(samples_block, dict):
        raise ValueError(
            "results['samples'] must be a dictionary. "
            "Check the lm-eval version and log_samples setting."
        )

    mmlu_task_keys = [k for k in samples_block.keys() if k == "mmlu" or k.startswith("mmlu_")]
    if not mmlu_task_keys:
        raise ValueError(
            "No MMLU samples found in results['samples']."
        )

    for task_key in sorted(mmlu_task_keys):
        sample_list = samples_block[task_key]
        if not isinstance(sample_list, list):
            continue

        for idx, sample in enumerate(sample_list):
            if not isinstance(sample, dict):
                continue

            doc = sample.get("doc", {})
            doc_str = normalize_to_str(doc)
            doc_hash = hash_text(doc_str) if doc_str else ""

            doc_id = infer_doc_id(sample, idx)
            subject = infer_subject(task_key, sample)
            correct = infer_correct(sample)
            target = infer_target(sample)
            prediction = infer_prediction(sample)

            pair_uid = f"{task_key}::{doc_id}"

            rows.append({
                "task": task_key,
                "subject": subject,
                "doc_id": doc_id,
                "pair_uid": pair_uid,
                "sample_index_within_task": idx,
                "correct": correct,
                "target": normalize_to_str(target),
                "prediction": normalize_to_str(prediction),
                "doc_hash": doc_hash,
                "doc_json": doc_str,
                "raw_sample_json": normalize_to_str(sample),
            })

    return rows


def run_mmlu_eval(model_name, mask_path, mask_langs, tag, batch_size, out_dir):
    print(f"\n{'='*70}")
    print(f"Tag: {tag}")
    print(f"Mask: {mask_path or 'None (baseline)'}")
    print(f"Mask langs: {mask_langs or 'all'}")
    print(f"{'='*70}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_pruned = 0
    if mask_path:
        n_pruned = patch_model(model, mask_path, mask_langs)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    print(f"Running MMLU ({MMLU_FEWSHOT}-shot) with log_samples=True ...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=MMLU_FEWSHOT,
        batch_size=batch_size,
        log_samples=True,
    )

    mmlu_results = results.get("results", {})
    agg_mmlu_acc = None
    if "mmlu" in mmlu_results and isinstance(mmlu_results["mmlu"], dict):
        agg_mmlu_acc = mmlu_results["mmlu"].get("acc,none", None)

    print(f"MMLU aggregate acc: {agg_mmlu_acc}")
    print(f"Neurons pruned: {n_pruned}")

    raw_json_path = out_dir / f"{tag}_mmlu_full_results.json"
    safe_json_dump(results, raw_json_path)
    print(f"Saved raw full results to: {raw_json_path}")

    rows = extract_rows_from_samples(results)

    csv_path = out_dir / f"{tag}_mmlu_per_question.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "subject",
                "doc_id",
                "pair_uid",
                "sample_index_within_task",
                "correct",
                "target",
                "prediction",
                "doc_hash",
                "doc_json",
                "raw_sample_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-question CSV to: {csv_path}")
    print(f"Saved {len(rows)} question rows.")

    n_correct = sum(int(r["correct"]) for r in rows if r["correct"] is not None)
    n_known = sum(1 for r in rows if r["correct"] is not None)
    print(f"Per-question known correctness rows: {n_known}")
    if n_known > 0:
        print(f"Per-question accuracy from CSV: {n_correct / n_known:.6f}")

    del model, lm
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "n_pruned": n_pruned,
        "aggregate_mmlu_acc": agg_mmlu_acc,
        "n_rows": len(rows),
        "csv_path": str(csv_path),
        "raw_json_path": str(raw_json_path),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None,
                        help="Neuron mask file; omit for the baseline.")
    parser.add_argument("--mask_langs", type=str, nargs="*", default=None,
                        help="Languages to merge from the mask; default: all.")
    parser.add_argument("--tag", type=str, required=True,
                        help="Experiment identifier.")
    parser.add_argument("--batch_size", type=str, default="2",
                        help="Evaluation batch size or 'auto'.")
    parser.add_argument("--out_dir", type=str, default="results_per_question")
    args = parser.parse_args()

    bs = args.batch_size
    if bs != "auto":
        bs = int(bs)

    summary = run_mmlu_eval(
        model_name=args.model,
        mask_path=args.mask,
        mask_langs=args.mask_langs,
        tag=args.tag,
        batch_size=bs,
        out_dir=args.out_dir,
    )

    summary_path = Path(args.out_dir) / f"{args.tag}_summary.json"
    safe_json_dump(summary, summary_path)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
