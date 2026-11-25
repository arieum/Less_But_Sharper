import argparse
import os

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]

MODEL_SPECS = {
    "llama2-7b": {
        "model": "meta-llama/Llama-2-7b-hf",
        "default_seq_len": 4096,
    },
    "llama3.1-8b": {
        "model": "meta-llama/Llama-3.1-8B",
        "default_seq_len": 4096,
    },
    "opt-6.7b": {
        "model": "facebook/opt-6.7b",
        "default_seq_len": 2048,
    },
    "bloom-7b1": {
        "model": "bigscience/bloom-7b1",
        "default_seq_len": 2048,
    },
    "mistral-7b-v0.2": {
        "model": "mistral-community/Mistral-7B-v0.2",
        "default_seq_len": 4096,
    },
    "mistral-7b-v0.1": {
        "model": "mistralai/Mistral-7B-v0.1",
        "default_seq_len": 4096,
    }
}

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_mlp_structure(model):
    """Resolve the MLP projection and activation for a model family."""
    model_type = model.config.model_type

    if model_type == "opt":
        layers = model.model.decoder.layers
        intermediate_size = model.config.ffn_dim

        def projection_getter(layer):
            return layer.fc1

        def activation_getter(layer):
            return layer.activation_fn

        activation_definition = "activation_fn(fc1(x))"

    elif model_type == "bloom":
        layers = model.transformer.h
        intermediate_size = 4 * model.config.hidden_size

        def projection_getter(layer):
            return layer.mlp.dense_h_to_4h

        def activation_getter(layer):
            return layer.mlp.gelu_impl

        activation_definition = "gelu(dense_h_to_4h(x))"

    elif model_type in {"llama", "mistral"}:
        layers = model.model.layers
        intermediate_size = model.config.intermediate_size

        def projection_getter(layer):
            return layer.mlp.gate_proj

        def activation_getter(layer):
            return layer.mlp.act_fn

        activation_definition = "act_fn(gate_proj(x))"

    elif model_type == "phi":
        layers = model.model.layers
        intermediate_size = model.config.intermediate_size

        def projection_getter(layer):
            return layer.mlp.fc1

        def activation_getter(layer):
            return layer.mlp.activation_fn

        activation_definition = "activation_fn(fc1(x))"

    else:
        raise NotImplementedError(
            f"Unsupported model_type: {model_type}"
        )

    return (
        layers,
        intermediate_size,
        projection_getter,
        activation_getter,
        activation_definition,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_tag",
        required=True,
        choices=MODEL_SPECS.keys(),
    )
    parser.add_argument(
        "-l",
        "--lang",
        required=True,
        choices=LANGS,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="float16",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of input tokens to process.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )

    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_tag]
    model_name = spec["model"]
    dtype = DTYPE_MAP[args.dtype]

    print("=" * 70)
    print(f"Model tag : {args.model_tag}")
    print(f"Model     : {model_name}")
    print(f"Language  : {args.lang}")
    print(f"Dtype     : {args.dtype}")
    print("=" * 70)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.config.use_cache = False

    (
        layers,
        intermediate_size,
        projection_getter,
        activation_getter,
        activation_definition,
    ) = resolve_mlp_structure(model)

    num_layers = len(layers)

    print(f"model_type        : {model.config.model_type}")
    print(f"num_layers        : {num_layers}")
    print(f"intermediate_size : {intermediate_size}")
    print(f"activation        : {activation_definition}")

    over_zero = torch.zeros(
        num_layers,
        intermediate_size,
        dtype=torch.int64,
        device="cpu",
    )

    def get_activation_hook(layer_idx, activation_fn):
        def hook(module, inputs, output):
            with torch.no_grad():
                activation = activation_fn(output)

                reduce_dims = tuple(range(activation.ndim - 1))

                active_count = (
                    activation > 0
                ).sum(
                    dim=reduce_dims,
                    dtype=torch.int64,
                ).cpu()

                over_zero[layer_idx].add_(active_count)

        return hook

    hooks = []

    for layer_idx, layer in enumerate(layers):
        projection = projection_getter(layer)
        activation_fn = activation_getter(layer)

        handle = projection.register_forward_hook(
            get_activation_hook(
                layer_idx=layer_idx,
                activation_fn=activation_fn,
            )
        )
        hooks.append(handle)

    print(f"Attached {len(hooks)} hooks.")

    ids_path = os.path.join(
        args.data_dir,
        f"id.{args.lang}.train.{args.model_tag}",
    )

    print(f"Loading token IDs: {ids_path}")
    ids = torch.load(ids_path, map_location="cpu")

    if not isinstance(ids, torch.Tensor):
        raise TypeError(
            f"Expected Tensor from {ids_path}, got {type(ids)}"
        )

    seq_len = (
        args.seq_len
        if args.seq_len is not None
        else spec["default_seq_len"]
    )

    available_tokens = ids.numel()

    if args.max_tokens is not None:
        available_tokens = min(
            available_tokens,
            args.max_tokens,
        )

    usable_tokens = (
        available_tokens // seq_len
    ) * seq_len

    if usable_tokens == 0:
        raise ValueError(
            f"Not enough tokens. "
            f"available={available_tokens}, seq_len={seq_len}"
        )

    input_ids = ids[:usable_tokens].reshape(-1, seq_len)

    num_sequences = input_ids.size(0)
    total_batches = (
        num_sequences + args.batch_size - 1
    ) // args.batch_size

    input_device = model.get_input_embeddings().weight.device

    print(f"seq_len          : {seq_len}")
    print(f"usable_tokens    : {usable_tokens:,}")
    print(f"num_sequences    : {num_sequences:,}")
    print(f"batch_size       : {args.batch_size}")
    print(f"total_batches    : {total_batches:,}")
    print(f"input_device     : {input_device}")

    backbone = model.base_model

    try:
        with torch.inference_mode():
            for start in tqdm(
                range(0, num_sequences, args.batch_size),
                desc=f"{args.model_tag}/{args.lang}",
            ):
                batch_ids = input_ids[
                    start:start + args.batch_size
                ].to(
                    input_device,
                    non_blocking=True,
                )

                _ = backbone(
                    input_ids=batch_ids,
                    use_cache=False,
                    return_dict=False,
                )

                del batch_ids

    finally:
        for handle in hooks:
            handle.remove()

    output_dict = {
        "model": model_name,
        "model_tag": args.model_tag,
        "model_type": model.config.model_type,
        "lang": args.lang,
        "n": usable_tokens,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "activation_definition": activation_definition,
        "over_zero": over_zero,
    }

    save_path = os.path.join(
        args.data_dir,
        f"activation.{args.lang}.train.{args.model_tag}",
    )

    torch.save(output_dict, save_path)

    print("=" * 70)
    print(f"Saved: {save_path}")
    print(f"n     : {usable_tokens:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
