import argparse
import os
import math

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]

MODEL_SPECS = {
    "llama2-7b": {
        "model": "meta-llama/Llama-2-7b-hf",
        "default_seq_len": 2000,
    },
    "llama3.1-8b": {
        "model": "meta-llama/Llama-3.1-8B",
        "default_seq_len": 2000,
    },
    "llama3-8b-base": {
        "model": "meta-llama/Llama-3.1-8B",
        "default_seq_len": 2000,
    },
    "opt-6.7b": {
        "model": "facebook/opt-6.7b",
        "default_seq_len": 2000,
    },
    "bloom-7b1": {
        "model": "bigscience/bloom-7b1",
        "default_seq_len": 2000,
    },
    "mistral-7b-v0.2": {
        "model": "mistral-community/Mistral-7B-v0.2",
        "default_seq_len": 2000,
    },
}

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_lape_candidates(
    in_neuron_path: str,
    lang: str,
):
    """Load candidate coordinates for one language."""
    print(f"Loading candidates: {in_neuron_path}")

    neuron_data = torch.load(
        in_neuron_path,
        map_location="cpu",
    )

    lang_idx = LANGS.index(lang)
    per_layer = neuron_data[lang_idx]

    coords = []

    for layer_idx, neuron_indices in enumerate(per_layer):
        for neuron_idx in neuron_indices.tolist():
            coords.append(
                (int(layer_idx), int(neuron_idx))
            )

    return coords


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
        intermediate_size = (
            4 * model.config.hidden_size
        )

        def projection_getter(layer):
            return layer.mlp.dense_h_to_4h

        def activation_getter(layer):
            return layer.mlp.gelu_impl

        activation_definition = (
            "gelu(dense_h_to_4h(x))"
        )

    elif model_type in {"llama", "mistral"}:
        layers = model.model.layers
        intermediate_size = (
            model.config.intermediate_size
        )

        def projection_getter(layer):
            return layer.mlp.gate_proj

        def activation_getter(layer):
            return layer.mlp.act_fn

        activation_definition = (
            "act_fn(gate_proj(x))"
        )

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


def build_tracking_maps(
    coords,
    num_layers,
    intermediate_size,
):
    """Map layer-local neuron indices to signature rows."""
    layer_entries = {}

    for global_row, (layer_idx, neuron_idx) in enumerate(coords):
        if not 0 <= layer_idx < num_layers:
            raise ValueError(
                f"Invalid layer index: {layer_idx}; "
                f"num_layers={num_layers}"
            )

        if not 0 <= neuron_idx < intermediate_size:
            raise ValueError(
                f"Invalid neuron index: "
                f"layer={layer_idx}, neuron={neuron_idx}; "
                f"intermediate_size={intermediate_size}"
            )

        layer_entries.setdefault(
            layer_idx,
            [],
        ).append(
            (neuron_idx, global_row)
        )

    layer_to_neuron_indices = {}
    layer_to_global_rows = {}

    for layer_idx, entries in layer_entries.items():
        entries.sort(key=lambda x: x[0])

        layer_to_neuron_indices[layer_idx] = torch.tensor(
            [neuron_idx for neuron_idx, _ in entries],
            dtype=torch.long,
        )

        layer_to_global_rows[layer_idx] = torch.tensor(
            [global_row for _, global_row in entries],
            dtype=torch.long,
        )

    return (
        layer_to_neuron_indices,
        layer_to_global_rows,
    )


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
        help="Override the model registered for model_tag.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=LANGS,
    )
    parser.add_argument(
        "--in_neuron",
        required=True,
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=30_000,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Length of each independent input sequence.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequences per forward pass.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="float16",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
    )

    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_tag]
    model_name = args.model or spec["model"]
    seq_len = (
        args.seq_len
        if args.seq_len is not None
        else spec["default_seq_len"]
    )
    dtype = DTYPE_MAP[args.dtype]

    if args.num_tokens <= 0:
        raise ValueError("--num_tokens must be positive")

    if seq_len <= 0:
        raise ValueError("--seq_len must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    os.makedirs(args.out_dir, exist_ok=True)

    data_path = os.path.join(
        args.data_dir,
        f"id.{args.lang}.train.{args.model_tag}",
    )

    out_path = os.path.join(
        args.out_dir,
        (
            f"signature.{args.lang}."
            f"T{args.num_tokens}.token."
            f"{args.model_tag}.pt"
        ),
    )

    print(f"Loading token IDs: {data_path}")

    full_ids = torch.load(
        data_path,
        map_location="cpu",
    )

    if not isinstance(full_ids, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(full_ids)}"
        )

    if full_ids.numel() < args.num_tokens:
        raise ValueError(
            f"Not enough tokens: "
            f"need={args.num_tokens:,}, "
            f"available={full_ids.numel():,}"
        )

    input_ids = full_ids[:args.num_tokens].long()

    print(f"Loading model: {model_name}")

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

    coords = load_lape_candidates(
        args.in_neuron,
        args.lang,
    )

    num_candidates = len(coords)

    print(
        f"Tracking {num_candidates:,} "
        f"candidate neurons"
    )

    (
        layer_to_neuron_indices,
        layer_to_global_rows,
    ) = build_tracking_maps(
        coords=coords,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
    )

    signature_matrix = torch.zeros(
        (
            num_candidates,
            args.num_tokens,
        ),
        dtype=torch.bool,
        device="cpu",
    )

    captured_activations = {}
    hook_handles = []

    def make_hook(
        layer_idx,
        activation_fn,
        target_indices_cpu,
    ):
        def hook(module, inputs, output):
            with torch.no_grad():
                target_indices = target_indices_cpu.to(
                    output.device,
                    non_blocking=True,
                )

                activation = activation_fn(output)

                selected_activation = activation.index_select(
                    dim=-1,
                    index=target_indices,
                )

                binary_activation = (
                    selected_activation > 0
                ).reshape(
                    -1,
                    target_indices.numel(),
                )

                captured_activations[layer_idx] = (
                    binary_activation.detach().cpu()
                )

        return hook

    for layer_idx, target_indices in (
        layer_to_neuron_indices.items()
    ):
        layer = layers[layer_idx]
        projection = projection_getter(layer)
        activation_fn = activation_getter(layer)

        handle = projection.register_forward_hook(
            make_hook(
                layer_idx=layer_idx,
                activation_fn=activation_fn,
                target_indices_cpu=target_indices,
            )
        )

        hook_handles.append(handle)

    input_device = (
        model.get_input_embeddings().weight.device
    )

    def process_batch(
        batch_ids_cpu,
        token_start,
        token_end,
    ):
        nonlocal captured_activations

        captured_activations = {}

        batch_ids = batch_ids_cpu.to(
            input_device,
            non_blocking=True,
        )

        with torch.inference_mode():
            outputs = model.base_model(
                input_ids=batch_ids,
                use_cache=False,
                return_dict=False,
            )

        expected_token_count = (
            token_end - token_start
        )

        for layer_idx, global_rows in (
            layer_to_global_rows.items()
        ):
            if layer_idx not in captured_activations:
                raise RuntimeError(
                    f"No activation captured "
                    f"for layer {layer_idx}"
                )

            binary_activation = (
                captured_activations[layer_idx]
            )

            if binary_activation.size(0) != expected_token_count:
                raise RuntimeError(
                    f"Activation token mismatch at "
                    f"layer {layer_idx}: "
                    f"expected={expected_token_count}, "
                    f"received={binary_activation.size(0)}"
                )

            signature_matrix[
                global_rows,
                token_start:token_end,
            ] = binary_activation.transpose(0, 1)

        del outputs
        del batch_ids

    num_full_sequences = (
        args.num_tokens // seq_len
    )

    full_token_count = (
        num_full_sequences * seq_len
    )

    if num_full_sequences > 0:
        sequences = input_ids[
            :full_token_count
        ].reshape(
            num_full_sequences,
            seq_len,
        )

        sequence_starts = range(
            0,
            num_full_sequences,
            args.batch_size,
        )

        for sequence_start in tqdm(
            sequence_starts,
            desc=f"{args.model_tag}/{args.lang}",
        ):
            sequence_end = min(
                sequence_start + args.batch_size,
                num_full_sequences,
            )

            batch_ids_cpu = sequences[
                sequence_start:sequence_end
            ]

            token_start = sequence_start * seq_len
            token_end = sequence_end * seq_len

            process_batch(
                batch_ids_cpu=batch_ids_cpu,
                token_start=token_start,
                token_end=token_end,
            )

    remainder = (
        args.num_tokens - full_token_count
    )

    if remainder > 0:
        remainder_ids = input_ids[
            full_token_count:args.num_tokens
        ].unsqueeze(0)

        process_batch(
            batch_ids_cpu=remainder_ids,
            token_start=full_token_count,
            token_end=args.num_tokens,
        )

    for handle in hook_handles:
        handle.remove()

    output_data = {
        "model": model_name,
        "model_tag": args.model_tag,
        "model_type": model.config.model_type,
        "lang": args.lang,
        "coords": coords,
        "signature": signature_matrix,
        "num_tokens": args.num_tokens,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "activation_definition": activation_definition,
        "context_reset_every": seq_len,
    }

    print(
        f"Saving signature: "
        f"{tuple(signature_matrix.shape)}"
    )
    print(f"Output: {out_path}")

    torch.save(
        output_data,
        out_path,
    )

    print("Done.")


if __name__ == "__main__":
    main()
