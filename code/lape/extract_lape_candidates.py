import argparse
import os

import torch


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]


def percentile_by_kthvalue(values, ratio):
    """Return a bounded kth-value percentile."""
    flattened = values.flatten()
    num_values = flattened.numel()

    k = round(num_values * ratio)
    k = max(1, min(k, num_values))

    return flattened.kthvalue(k).values.item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_tag",
        required=True,
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix for the neuron and metadata files.",
    )
    parser.add_argument(
        "--top_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--filter_rate",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--activation_bar_ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--largest",
        action="store_true",
        help="Select high-entropy neurons instead of low-entropy neurons.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )

    args = parser.parse_args()

    neuron_out_path = f"{args.out_prefix}.neuron.pth"
    meta_out_path = f"{args.out_prefix}.meta.pth"

    n_list = []
    over_zero_list = []
    activation_metadata = []

    for lang in LANGS:
        path = os.path.join(
            args.data_dir,
            f"activation.{lang}.train.{args.model_tag}",
        )

        print(f"load: {path}")

        data = torch.load(path, map_location="cpu")

        n_list.append(float(data["n"]))
        over_zero_list.append(
            data["over_zero"].float()
        )

        activation_metadata.append({
            "lang": lang,
            "n": int(data["n"]),
            "seq_len": data.get("seq_len"),
            "activation_definition": data.get(
                "activation_definition"
            ),
        })

    n = torch.tensor(
        n_list,
        dtype=torch.float32,
    ).view(1, 1, -1)

    over_zero = torch.stack(
        over_zero_list,
        dim=-1,
    )

    num_layers, intermediate_size, lang_num = (
        over_zero.shape
    )

    activation_probs = over_zero / n

    probability_sum = activation_probs.sum(
        dim=-1,
        keepdim=True,
    )

    normed_probs = torch.where(
        probability_sum > 0,
        activation_probs / probability_sum,
        torch.zeros_like(activation_probs),
    )

    normed_probs = torch.nan_to_num(
        normed_probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    log_probs = torch.where(
        normed_probs > 0,
        normed_probs.log(),
        torch.zeros_like(normed_probs),
    )

    raw_entropy = -(
        normed_probs * log_probs
    ).sum(dim=-1)

    if torch.isnan(raw_entropy).any():
        raise ValueError("raw_entropy contains NaN")

    flattened_probs = activation_probs.flatten()

    top_prob_value = percentile_by_kthvalue(
        activation_probs,
        args.filter_rate,
    )

    print(
        f"filter threshold (prob): "
        f"{top_prob_value:.8f}"
    )

    top_position = (
        activation_probs > top_prob_value
    ).sum(dim=-1)

    selection_entropy = raw_entropy.clone()

    if args.largest:
        selection_entropy[top_position == 0] = -torch.inf
    else:
        selection_entropy[top_position == 0] = torch.inf

    flattened_entropy = selection_entropy.flatten()

    selected_k = max(
        1,
        round(
            flattened_entropy.numel()
            * args.top_rate
        ),
    )

    _, flat_indices = flattened_entropy.topk(
        selected_k,
        largest=args.largest,
    )

    row_index = (
        flat_indices // selection_entropy.size(1)
    )
    col_index = (
        flat_indices % selection_entropy.size(1)
    )

    selected_probs = activation_probs[
        row_index,
        col_index,
    ]

    print(f"selected candidates: {selected_k:,}")
    print(
        "argmax bincount:",
        torch.bincount(
            selected_probs.argmax(dim=-1),
            minlength=lang_num,
        ).tolist(),
    )

    activation_bar = percentile_by_kthvalue(
        activation_probs,
        args.activation_bar_ratio,
    )

    print(
        f"activation_bar (prob): "
        f"{activation_bar:.8f}"
    )

    selected_probs_t = selected_probs.transpose(0, 1)

    per_lang_counts = (
        selected_probs_t > activation_bar
    ).sum(dim=1)

    print(
        "per-language counts over bar:",
        {
            lang: int(count)
            for lang, count in zip(
                LANGS,
                per_lang_counts.tolist(),
            )
        },
    )

    lang_ids, indices_in_selected = torch.where(
        selected_probs_t > activation_bar
    )

    merged_index = torch.stack(
        (row_index, col_index),
        dim=-1,
    )

    splits = torch.bincount(
        lang_ids,
        minlength=lang_num,
    ).tolist()

    final_indices = []
    cursor = 0

    for lang_idx, count in enumerate(splits):
        selected_local_indices = indices_in_selected[
            cursor:cursor + count
        ]
        cursor += count

        coordinates = [
            tuple(merged_index[idx].tolist())
            for idx in selected_local_indices
        ]

        coordinates.sort()

        layer_indices = [
            [] for _ in range(num_layers)
        ]

        for layer_idx, neuron_idx in coordinates:
            layer_indices[layer_idx].append(neuron_idx)

        layer_indices = [
            torch.tensor(
                indices,
                dtype=torch.long,
            )
            for indices in layer_indices
        ]

        final_indices.append(layer_indices)

    meta_data = {
        "model_tag": args.model_tag,
        "languages": LANGS,
        "normed_probs": normed_probs,
        "entropy": selection_entropy,
        "raw_entropy": raw_entropy,
        "activation_probs": activation_probs,
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "lang_num": lang_num,
        "top_rate": args.top_rate,
        "filter_rate": args.filter_rate,
        "activation_bar_ratio": args.activation_bar_ratio,
        "filter_threshold": top_prob_value,
        "activation_bar": activation_bar,
        "activation_metadata": activation_metadata,
    }

    torch.save(final_indices, neuron_out_path)
    torch.save(meta_data, meta_out_path)

    print(f"saved: {neuron_out_path}")
    print(f"saved: {meta_out_path}")


if __name__ == "__main__":
    main()
