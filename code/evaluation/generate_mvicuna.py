import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]

MODEL_SPECS = {
    "llama2-7b": {
        "model": "meta-llama/Llama-2-7b-hf",
        "use_fast": True,
    },
    "llama3.1-8b": {
        "model": "meta-llama/Llama-3.1-8B",
        "use_fast": True,
    },
    "opt-6.7b": {
        "model": "facebook/opt-6.7b",
        "use_fast": False,
    },
    "bloom-7b1": {
        "model": "bigscience/bloom-7b1",
        "use_fast": True,
    },
    "mistral-7b-v0.2": {
        "model": "mistral-community/Mistral-7B-v0.2",
        "use_fast": True,
    },
}

ANSWER_LANG = {
    "zh": " 请用中文回答。",
    "en": " Answer in English.",
    "fr": " Veuillez répondre en français.",
    "es": " Por favor responda en español.",
    "id": " Tolong dijawab dalam bahasa Indonesia.",
    "ja": " 日本語で答えてください。",
    "vi": " Hãy trả lời bằng tiếng Việt.",
}


def load_dataset(lang: str, data_dir: str):
    """Load mVicuna questions and construct model prompts."""
    path = os.path.join(data_dir, f"{lang}.txt")

    with open(path, "r", encoding="utf-8") as f:
        questions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    records = []

    for question in questions:
        prompt = (
            f"Q: {question}"
            f"{ANSWER_LANG[lang]}"
            f"\nA:"
        )

        records.append({
            "question": question,
            "prompt": prompt,
        })

    return records


def get_intervention_modules(model):
    """Return the final MLP projections used for intervention."""
    model_type = model.config.model_type

    if model_type == "opt":
        layers = model.model.decoder.layers
        modules = [
            layer.fc2
            for layer in layers
        ]
        intervention_point = "fc2 input"

    elif model_type == "bloom":
        layers = model.transformer.h
        modules = [
            layer.mlp.dense_4h_to_h
            for layer in layers
        ]
        intervention_point = "dense_4h_to_h input"

    elif model_type in {"mistral", "llama"}:
        layers = model.model.layers
        modules = [
            layer.mlp.down_proj
            for layer in layers
        ]
        intervention_point = "down_proj input"

    else:
        raise NotImplementedError(
            f"Unsupported model_type: {model_type}"
        )

    return modules, intervention_point


def register_intervention(
    model,
    layer_masks,
    mode: str,
    scale: float,
):
    """Register pruning or steering hooks on selected neurons."""
    modules, intervention_point = (
        get_intervention_modules(model)
    )

    if len(layer_masks) != len(modules):
        raise ValueError(
            "Mask/model layer mismatch: "
            f"mask_layers={len(layer_masks)}, "
            f"model_layers={len(modules)}"
        )

    handles = []

    for layer_idx, (module, layer_mask) in enumerate(
        zip(modules, layer_masks)
    ):
        index_cpu = torch.as_tensor(
            layer_mask,
            dtype=torch.long,
            device="cpu",
        )

        if index_cpu.numel() == 0:
            continue

        if torch.any(index_cpu < 0):
            raise ValueError(
                f"Negative neuron index at layer {layer_idx}"
            )

        if hasattr(module, "in_features"):
            if int(index_cpu.max()) >= module.in_features:
                raise ValueError(
                    f"Neuron index out of range at layer "
                    f"{layer_idx}: max={int(index_cpu.max())}, "
                    f"in_features={module.in_features}"
                )

        index_cache = {}

        def make_pre_hook(
            indices_cpu,
            cache,
            selected_mode,
            selected_scale,
        ):
            def pre_hook(module, inputs):
                intermediate = inputs[0]

                device_key = str(intermediate.device)

                if device_key not in cache:
                    cache[device_key] = indices_cpu.to(
                        intermediate.device,
                        non_blocking=True,
                    )

                indices = cache[device_key]

                modified = intermediate.clone()

                if selected_mode == "prune":
                    modified.index_fill_(
                        dim=-1,
                        index=indices,
                        value=0.0,
                    )

                elif selected_mode == "steer":
                    selected_values = modified.index_select(
                        dim=-1,
                        index=indices,
                    )

                    selected_values = (
                        selected_values * selected_scale
                    )

                    modified.index_copy_(
                        dim=-1,
                        index=indices,
                        source=selected_values,
                    )

                else:
                    raise ValueError(
                        f"Unsupported mode: {selected_mode}"
                    )

                return (
                    modified,
                    *inputs[1:],
                )

            return pre_hook

        handle = module.register_forward_pre_hook(
            make_pre_hook(
                indices_cpu=index_cpu,
                cache=index_cache,
                selected_mode=mode,
                selected_scale=scale,
            )
        )

        handles.append(handle)

    return handles, intervention_point


def remove_intervention(handles):
    for handle in handles:
        handle.remove()


def trim_generated_text(text: str):
    """Remove a trailing question-answer turn from generated text."""
    stop_strings = [
        "\nQ:",
        "\nA:",
        "\nQuestion:",
        "\nAnswer:",
    ]

    stop_positions = [
        text.find(stop)
        for stop in stop_strings
        if stop in text
    ]

    if stop_positions:
        text = text[:min(stop_positions)]

    return text.strip()


def generate_texts(
    model,
    tokenizer,
    records,
    input_device,
    batch_size,
    max_new_tokens,
    repetition_penalty,
    max_prompt_tokens,
):
    outputs_all = []

    for start in range(0, len(records), batch_size):
        batch_records = records[
            start:start + batch_size
        ]

        prompts = [
            record["prompt"]
            for record in batch_records
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )

        inputs = {
            key: value.to(
                input_device,
                non_blocking=True,
            )
            for key, value in inputs.items()
        }

        prompt_width = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = output_ids[:, prompt_width:]

        decoded_outputs = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        outputs_all.extend([
            trim_generated_text(text)
            for text in decoded_outputs
        ])

        completed = min(
            start + len(batch_records),
            len(records),
        )

        print(
            f"  [{completed}/{len(records)}] done",
            flush=True,
        )

        del inputs
        del output_ids
        del generated_ids

    return outputs_all


def save_generation(
    output_path,
    records,
    outputs,
    model_name,
    model_tag,
    target_lang,
    condition,
    mode,
    scale,
    mask_path=None,
):
    results = []

    for item_idx, (record, output) in enumerate(
        zip(records, outputs)
    ):
        results.append({
            "id": item_idx,
            "question": record["question"],
            "prompt": record["prompt"],
            "output": output,
            "target_lang": target_lang,
            "model": model_name,
            "model_tag": model_tag,
            "condition": condition,
            "mode": mode if mask_path else "baseline",
            "scale": scale if mask_path else None,
            "activation_mask": (
                os.path.basename(mask_path)
                if mask_path
                else None
            ),
        })

    tmp_output_path = f"{output_path}.tmp"

    with open(
        tmp_output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tmp_output_path, output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_tag",
        required=True,
        choices=MODEL_SPECS.keys(),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--activation_masks",
        nargs="*",
        default=[],
        help="Evaluate one or more Top-K/MMR mask files.",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
    )
    parser.add_argument(
        "--mode",
        choices=["prune", "steer"],
        default="steer",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["zh", "fr", "es", "vi", "id", "ja"],
        choices=LANGS,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mvicuna",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.include_baseline and not args.activation_masks:
        raise ValueError(
            "Nothing to generate. Use --include_baseline "
            "or --activation_masks."
        )

    if args.mode == "steer" and args.scale <= 0:
        raise ValueError("--scale must be positive")

    spec = MODEL_SPECS[args.model_tag]
    model_name = args.model or spec["model"]

    if args.output_root is None:
        output_folder = os.path.join(
            f"generation_result_{args.mode}",
            args.model_tag,
            "mvicuna",
        )
    else:
        output_folder = args.output_root

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    print(f"Loading model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=spec["use_fast"],
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token"
            )

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.config.use_cache = True

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )

    input_device = (
        model.get_input_embeddings().weight.device
    )

    modules, intervention_point = (
        get_intervention_modules(model)
    )

    num_layers = len(modules)

    print(f"model_type         : {model.config.model_type}")
    print(f"num_layers         : {num_layers}")
    print(f"intervention_point : {intervention_point}")
    print(f"input_device       : {input_device}")
    print(f"output_folder      : {output_folder}")

    if args.include_baseline:
        for lang in args.langs:
            output_path = os.path.join(
                output_folder,
                f"{lang}.baseline.json",
            )

            if (
                args.skip_existing
                and os.path.exists(output_path)
            ):
                print(
                    f"Skip existing: {output_path}",
                    flush=True,
                )
                continue

            print(
                f"\n[baseline] Generating {lang}",
                flush=True,
            )

            records = load_dataset(
                lang=lang,
                data_dir=args.data_dir,
            )

            outputs = generate_texts(
                model=model,
                tokenizer=tokenizer,
                records=records,
                input_device=input_device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                max_prompt_tokens=args.max_prompt_tokens,
            )

            save_generation(
                output_path=output_path,
                records=records,
                outputs=outputs,
                model_name=model_name,
                model_tag=args.model_tag,
                target_lang=lang,
                condition="baseline",
                mode=args.mode,
                scale=args.scale,
            )

            print(f"Saved: {output_path}", flush=True)

    for mask_path in args.activation_masks:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        activation_masks = torch.load(
            mask_path,
            map_location="cpu",
        )

        if len(activation_masks) != len(LANGS):
            raise ValueError(
                f"{mask_path}: expected {len(LANGS)} "
                f"language masks, got {len(activation_masks)}"
            )

        mask_name = Path(mask_path).stem
        scale_name = format(args.scale, "g")

        print("\n" + "=" * 70)
        print(f"Mask: {mask_path}")
        print("=" * 70)

        for lang in args.langs:
            lang_idx = LANGS.index(lang)
            layer_mask = activation_masks[lang_idx]

            output_name = (
                f"{lang}."
                f"{args.mode}_scale{scale_name}."
                f"{mask_name}.json"
            )

            output_path = os.path.join(
                output_folder,
                output_name,
            )

            if (
                args.skip_existing
                and os.path.exists(output_path)
            ):
                print(
                    f"Skip existing: {output_path}",
                    flush=True,
                )
                continue

            selected_count = sum(
                len(layer_indices)
                for layer_indices in layer_mask
            )

            print(
                f"\n[{mask_name}] "
                f"lang={lang}, "
                f"neurons={selected_count}",
                flush=True,
            )

            handles = []

            try:
                handles, _ = register_intervention(
                    model=model,
                    layer_masks=layer_mask,
                    mode=args.mode,
                    scale=args.scale,
                )

                records = load_dataset(
                    lang=lang,
                    data_dir=args.data_dir,
                )

                outputs = generate_texts(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    input_device=input_device,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    max_prompt_tokens=args.max_prompt_tokens,
                )

                save_generation(
                    output_path=output_path,
                    records=records,
                    outputs=outputs,
                    model_name=model_name,
                    model_tag=args.model_tag,
                    target_lang=lang,
                    condition=mask_name,
                    mode=args.mode,
                    scale=args.scale,
                    mask_path=mask_path,
                )

                print(
                    f"Saved: {output_path}",
                    flush=True,
                )

            finally:
                remove_intervention(handles)

    print("\nAll generation completed.", flush=True)


if __name__ == "__main__":
    main()
