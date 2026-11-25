import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path

import fasttext
from openai import OpenAI
from tqdm import tqdm


LANG_FULL_NAME = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ja": "Japanese",
}

client = None
ft_model = None
judge_model_name = None


JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator.

The response was required to be written in {lang_name} ({target_lang}).
Evaluate LANGUAGE QUALITY and CONTENT QUALITY separately.

Language Score:
5 = Entirely in the requested language, fluent and natural, with no meaningful language mixing.
4 = Correct requested language with only minor grammatical, lexical, or fluency issues.
3 = Mostly understandable in the requested language, but with noticeable errors or language mixing.
2 = Substantial language errors or extensive mixing with another language.
1 = Completely wrong language, nearly unintelligible, or empty.

Content Score:
5 = Fully relevant, accurate, clear, and helpful.
4 = Mostly correct and helpful, with only minor omissions or inaccuracies.
3 = Partially correct or relevant, but incomplete, vague, or containing noticeable errors.
2 = Mostly incorrect, irrelevant, or unhelpful.
1 = Completely incorrect, nonsensical, unrelated, or empty.

Assess content independently from target-language compliance whenever the semantic content can still be understood.

[User Question]
{question}

[AI Response]
{response}

Output strictly in this format:

[[Language Score: X]]
[[Content Score: Y]]
Brief explanation in English.

X and Y must each be an integer from 1 to 5.
"""


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)

            if isinstance(data, list):
                return data

            return [data]

        except json.JSONDecodeError:
            f.seek(0)

            data = []

            for line in f:
                if line.strip():
                    data.append(json.loads(line))

            return data


def detect_language(response, target_lang):
    if not response.strip():
        return "empty", 0

    try:
        clean_text = re.sub(
            r"\s+",
            " ",
            response,
        ).strip()

        predictions = ft_model.predict(
            clean_text,
            k=1,
        )

        detected = (
            predictions[0][0]
            .replace("__label__", "")
        )

        lid_match = int(detected == target_lang)

        return detected, lid_match

    except Exception:
        return "error", 0


def parse_judge_output(judge_output):
    language_match = re.search(
        r"\[\[\s*Language\s+Score\s*:\s*([1-5])\s*\]\]",
        judge_output,
        re.IGNORECASE,
    )

    content_match = re.search(
        r"\[\[\s*Content\s+Score\s*:\s*([1-5])\s*\]\]",
        judge_output,
        re.IGNORECASE,
    )

    if not language_match or not content_match:
        return None, None

    return (
        int(language_match.group(1)),
        int(content_match.group(1)),
    )


def evaluate_single_response(
    item_idx,
    item,
    target_lang,
    seed,
    max_retries,
):
    result = dict(item)

    question = (
        result.get("question")
        or result.get("input")
        or ""
    )

    response = result.get("output", "")

    detected, lid_match = detect_language(
        response=response,
        target_lang=target_lang,
    )

    result["detected_lang"] = detected
    result["lid_match"] = lid_match

    if not response.strip():
        result["language_score"] = 1
        result["content_score"] = 1
        result["judge_explanation"] = (
            "Empty response."
        )

        return item_idx, result

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        target_lang=target_lang,
        lang_name=LANG_FULL_NAME[target_lang],
        question=question,
        response=response,
    )

    for attempt in range(max_retries):
        try:
            api_response = (
                client.chat.completions.create(
                    model=judge_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    seed=seed,
                    extra_body={
                        "repetition_penalty": 1.1
                    },
                )
            )

            judge_output = (
                api_response
                .choices[0]
                .message
                .content
            )

            language_score, content_score = (
                parse_judge_output(judge_output)
            )

            if (
                language_score is None
                or content_score is None
            ):
                raise ValueError(
                    "Judge output parsing failed: "
                    f"{judge_output}"
                )

            result["language_score"] = (
                language_score
            )
            result["content_score"] = (
                content_score
            )
            result["judge_explanation"] = (
                judge_output
            )

            return item_idx, result

        except Exception as error:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

            result["language_score"] = -1
            result["content_score"] = -1
            result["judge_explanation"] = (
                f"API Error: {error}"
            )

            return item_idx, result

    return item_idx, result


def evaluate_file(
    input_path,
    output_path,
    target_lang,
    workers,
    seed,
    max_retries,
):
    data = load_json_or_jsonl(input_path)

    print(
        f"\nEvaluating: {input_path}",
        flush=True,
    )
    print(
        f"Target language: {target_lang}",
        flush=True,
    )
    print(
        f"Items: {len(data)}",
        flush=True,
    )

    evaluated_data = [None] * len(data)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        future_to_idx = {}

        for item_idx, item in enumerate(data):
            future = executor.submit(
                evaluate_single_response,
                item_idx,
                item,
                target_lang,
                seed,
                max_retries,
            )

            future_to_idx[future] = item_idx

        for future in tqdm(
            concurrent.futures.as_completed(
                future_to_idx
            ),
            total=len(future_to_idx),
            desc="Evaluation",
        ):
            item_idx, result = future.result()
            evaluated_data[item_idx] = result

    valid_language_scores = [
        item["language_score"]
        for item in evaluated_data
        if item["language_score"] != -1
    ]

    valid_content_scores = [
        item["content_score"]
        for item in evaluated_data
        if item["content_score"] != -1
    ]

    lid_values = [
        item["lid_match"]
        for item in evaluated_data
    ]

    avg_language_score = (
        sum(valid_language_scores)
        / len(valid_language_scores)
        if valid_language_scores
        else 0.0
    )

    avg_content_score = (
        sum(valid_content_scores)
        / len(valid_content_scores)
        if valid_content_scores
        else 0.0
    )

    lid_accuracy = (
        100.0 * sum(lid_values) / len(lid_values)
        if lid_values
        else 0.0
    )

    parse_failures = sum(
        1
        for item in evaluated_data
        if (
            item["language_score"] == -1
            or item["content_score"] == -1
        )
    )

    summary = {
        "input_file": str(input_path),
        "target_lang": target_lang,
        "num_items": len(evaluated_data),
        "avg_language_score": avg_language_score,
        "avg_content_score": avg_content_score,
        "lid_accuracy": lid_accuracy,
        "parse_or_api_failures": parse_failures,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "items": evaluated_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("-" * 60)
    print(
        f"Language score : "
        f"{avg_language_score:.3f} / 5"
    )
    print(
        f"Content score  : "
        f"{avg_content_score:.3f} / 5"
    )
    print(
        f"FastText LID   : "
        f"{lid_accuracy:.2f}%"
    )
    print(
        f"Failures       : "
        f"{parse_failures}"
    )
    print(f"Saved          : {output_path}")
    print("-" * 60)

    return summary


def main():
    global client
    global ft_model
    global judge_model_name

    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(
        required=True
    )

    input_group.add_argument(
        "--input_file",
        type=str,
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target_lang",
        choices=LANG_FULL_NAME.keys(),
        default=None,
        help="Target language for a single file; inferred from the filename otherwise.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ft_model_path",
        type=str,
        default="lid.176.bin",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8001/v1",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="llama-judge",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ft_model_path):
        raise FileNotFoundError(
            f"FastText model not found: "
            f"{args.ft_model_path}"
        )

    ft_model = fasttext.load_model(
        args.ft_model_path
    )

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    judge_model_name = args.judge_model

    output_root = Path(args.output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.input_file:
        input_paths = [
            Path(args.input_file)
        ]

        input_root = Path(args.input_file).parent

    else:
        input_root = Path(args.input_dir)

        input_paths = sorted([
            *input_root.rglob("*.json"),
            *input_root.rglob("*.jsonl"),
        ])

    if not input_paths:
        raise FileNotFoundError(
            "No JSON/JSONL files found"
        )

    summaries = []

    for input_path in input_paths:
        if args.target_lang is not None:
            target_lang = args.target_lang
        else:
            target_lang = input_path.name.split(".")[0]

        if target_lang not in LANG_FULL_NAME:
            print(
                f"Skip unknown language file: "
                f"{input_path}",
                flush=True,
            )
            continue

        relative_path = input_path.relative_to(
            input_root
        )

        output_path = (
            output_root
            / relative_path
        )

        if (
            args.skip_existing
            and output_path.exists()
        ):
            print(
                f"Skip existing: {output_path}",
                flush=True,
            )
            continue

        summary = evaluate_file(
            input_path=input_path,
            output_path=output_path,
            target_lang=target_lang,
            workers=args.workers,
            seed=args.seed,
            max_retries=args.max_retries,
        )

        summaries.append(summary)

    summary_path = (
        output_root
        / "_evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nOverall summary saved: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
