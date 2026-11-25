import argparse
import math

import torch


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]


def baseline_lape_topk(
    neuron_indices,
    entropy,
    target_count,
):
    """Select the K lowest-entropy LAPE candidates."""
    num_layers = len(neuron_indices)

    candidates = []

    for layer_idx in range(num_layers):
        for neuron_idx in (
            neuron_indices[layer_idx].tolist()
        ):
            entropy_value = float(
                entropy[layer_idx, neuron_idx]
            )

            if math.isfinite(entropy_value):
                candidates.append(
                    (
                        layer_idx,
                        int(neuron_idx),
                        entropy_value,
                    )
                )

    if not candidates:
        return [
            torch.empty(
                0,
                dtype=torch.long,
            )
            for _ in range(num_layers)
        ]

    candidates.sort(
        key=lambda item: (
            item[2],
            item[0],
            item[1],
        )
    )

    selected = candidates[
        :min(target_count, len(candidates))
    ]

    result = [
        [] for _ in range(num_layers)
    ]

    for layer_idx, neuron_idx, _ in selected:
        result[layer_idx].append(neuron_idx)

    return [
        torch.tensor(
            sorted(layer_neurons),
            dtype=torch.long,
        )
        for layer_neurons in result
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_neuron",
        required=True,
    )
    parser.add_argument(
        "--meta_path",
        required=True,
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_neuron",
        required=True,
    )

    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("--K must be positive")

    print("=" * 70)
    print(f"LAPE Top-K baseline: K={args.K}")
    print("=" * 70)

    lang_specific_neurons = torch.load(
        args.input_neuron,
        map_location="cpu",
    )

    meta_data = torch.load(
        args.meta_path,
        map_location="cpu",
    )

    entropy = meta_data.get(
        "raw_entropy",
        meta_data["entropy"],
    )

    if len(lang_specific_neurons) != len(LANGS):
        raise ValueError(
            f"Expected {len(LANGS)} languages, "
            f"got {len(lang_specific_neurons)}"
        )

    all_results = []

    for lang_idx, lang in enumerate(LANGS):
        original_neurons = (
            lang_specific_neurons[lang_idx]
        )

        num_candidates = sum(
            len(layer)
            for layer in original_neurons
        )

        selected = baseline_lape_topk(
            neuron_indices=original_neurons,
            entropy=entropy,
            target_count=args.K,
        )

        num_selected = sum(
            len(layer)
            for layer in selected
        )

        print(
            f"[{lang}] "
            f"candidates={num_candidates}, "
            f"selected={num_selected}"
        )

        all_results.append(selected)

    torch.save(
        all_results,
        args.out_neuron,
    )

    print(f"Saved: {args.out_neuron}")


if __name__ == "__main__":
    main()
