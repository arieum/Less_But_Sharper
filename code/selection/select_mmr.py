import argparse
import math
import os

import torch


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]


def calculate_jaccard_similarity_matrix(
    binary_matrix,
    device,
):
    """Compute pairwise Jaccard similarity for an [N, T] binary matrix."""
    matrix = binary_matrix.to(
        device=device,
        dtype=torch.float32,
    )

    intersection = torch.matmul(
        matrix,
        matrix.transpose(0, 1),
    )

    active_counts = matrix.sum(
        dim=1,
        keepdim=True,
    )

    union = (
        active_counts
        + active_counts.transpose(0, 1)
        - intersection
    )

    jaccard = torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(union),
    )

    del matrix
    del intersection
    del active_counts
    del union

    return jaccard


def mmr_select_jaccard(
    relevance_scores,
    jaccard_matrix,
    K,
    lam,
):
    """Select candidates with maximal marginal relevance."""
    num_candidates = relevance_scores.numel()
    K = min(K, num_candidates)

    if K == 0:
        return []

    selected_mask = torch.zeros(
        num_candidates,
        dtype=torch.bool,
        device=relevance_scores.device,
    )

    first = int(
        torch.argmax(relevance_scores).item()
    )

    selected = [first]
    selected_mask[first] = True

    max_similarities = (
        jaccard_matrix[:, first].clone()
    )

    while len(selected) < K:
        scores = (
            lam * relevance_scores
            - (1.0 - lam) * max_similarities
        )

        scores[selected_mask] = -torch.inf

        best = int(
            torch.argmax(scores).item()
        )

        selected.append(best)
        selected_mask[best] = True

        max_similarities = torch.maximum(
            max_similarities,
            jaccard_matrix[:, best],
        )

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sig_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--model_tag",
        required=True,
    )
    parser.add_argument(
        "--meta_path",
        required=True,
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=30_000,
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--out_neuron",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("--K must be positive")

    if not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError(
            "--lambda_ must be between 0 and 1"
        )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable"
            )

    device = torch.device(args.device)

    print("=" * 70)
    print("Token-level Jaccard MMR")
    print(f"model_tag : {args.model_tag}")
    print(f"K         : {args.K}")
    print(f"lambda    : {args.lambda_}")
    print(f"device    : {device}")
    print("=" * 70)

    meta_data = torch.load(
        args.meta_path,
        map_location="cpu",
    )

    entropy = meta_data.get(
        "raw_entropy",
        meta_data["entropy"],
    )

    num_layers = int(
        meta_data["num_layers"]
    )

    log_num_languages = math.log(
        len(LANGS)
    )

    all_results = []

    for lang in LANGS:
        signature_path = os.path.join(
            args.sig_dir,
            (
                f"signature.{lang}."
                f"T{args.num_tokens}.token."
                f"{args.model_tag}.pt"
            ),
        )

        print(f"\n[{lang}]")
        print(f"Loading: {signature_path}")

        if not os.path.exists(signature_path):
            raise FileNotFoundError(
                signature_path
            )

        signature_data = torch.load(
            signature_path,
            map_location="cpu",
        )

        coords = signature_data["coords"]
        signature_matrix = (
            signature_data["signature"]
        )

        if signature_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2-D signature, "
                f"got {signature_matrix.shape}"
            )

        if signature_matrix.size(0) != len(coords):
            raise ValueError(
                f"Coordinate/signature mismatch: "
                f"coords={len(coords)}, "
                f"rows={signature_matrix.size(0)}"
            )

        num_candidates = len(coords)

        print(
            f"Candidates: {num_candidates:,}, "
            f"signature={tuple(signature_matrix.shape)}"
        )

        if num_candidates == 0:
            all_results.append([
                torch.empty(
                    0,
                    dtype=torch.long,
                )
                for _ in range(num_layers)
            ])
            continue

        relevance_values = []

        for layer_idx, neuron_idx in coords:
            entropy_value = float(
                entropy[layer_idx, neuron_idx]
            )

            if not math.isfinite(entropy_value):
                relevance = 0.0
            else:
                relevance = (
                    1.0
                    - entropy_value
                    / log_num_languages
                )

                relevance = max(
                    0.0,
                    min(1.0, relevance),
                )

            relevance_values.append(relevance)

        relevance_scores = torch.tensor(
            relevance_values,
            dtype=torch.float32,
            device=device,
        )

        print("Computing Jaccard matrix...")

        jaccard_matrix = (
            calculate_jaccard_similarity_matrix(
                binary_matrix=signature_matrix,
                device=device,
            )
        )

        print("Running MMR...")

        selected_indices = mmr_select_jaccard(
            relevance_scores=relevance_scores,
            jaccard_matrix=jaccard_matrix,
            K=args.K,
            lam=args.lambda_,
        )

        final_coords = [
            coords[index]
            for index in selected_indices
        ]

        layer_results = [
            [] for _ in range(num_layers)
        ]

        for layer_idx, neuron_idx in final_coords:
            layer_results[layer_idx].append(
                int(neuron_idx)
            )

        layer_results = [
            torch.tensor(
                sorted(layer_neurons),
                dtype=torch.long,
            )
            for layer_neurons in layer_results
        ]

        all_results.append(layer_results)

        print(
            f"Selected: "
            f"{sum(len(x) for x in layer_results)}"
        )

        del relevance_scores
        del jaccard_matrix

        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(
        all_results,
        args.out_neuron,
    )

    print(f"\nSaved: {args.out_neuron}")


if __name__ == "__main__":
    main()
