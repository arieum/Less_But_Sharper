import argparse

import torch


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]


def calculate_jaccard_similarity_matrix(binary_matrix):
    """Compute pairwise Jaccard similarity on GPU."""
    matrix = binary_matrix.float().cuda()
    intersection = torch.matmul(matrix, matrix.T)
    active_counts = matrix.sum(dim=1).unsqueeze(1)
    union = active_counts + active_counts.T - intersection
    return (intersection / (union + 1e-8)).cpu()


def mrmr_select_jaccard(candidates, relevance, similarity, K):
    """Select candidates by relevance minus mean redundancy."""
    num_candidates = len(candidates)
    K = min(K, num_candidates)
    selected = []
    remaining = set(range(num_candidates))

    first = max(remaining, key=lambda index: relevance[index])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < K:
        remaining_list = list(remaining)
        mean_similarity = similarity[remaining_list][:, selected].mean(dim=1)
        scores = relevance[remaining_list] - mean_similarity
        best = remaining_list[scores.argmax().item()]
        selected.append(best)
        remaining.remove(best)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig_dir", type=str, default="data")
    parser.add_argument("--model_tag", type=str, default="llama-7b")
    parser.add_argument("--meta_path", type=str, default="LLaMA-2-7B.meta.pth")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--out_neuron", type=str, required=True)
    args = parser.parse_args()

    print(f"Token-level Jaccard mRMR: K={args.K}")

    metadata = torch.load(args.meta_path, map_location="cpu")
    entropy = metadata["entropy"]
    num_layers = metadata["num_layers"]
    log_num_languages = torch.tensor(len(LANGS)).float().log().item()
    all_results = []

    for lang in LANGS:
        signature_path = (
            f"{args.sig_dir}/signature.{lang}.T30000.token.{args.model_tag}.pt"
        )
        print(f"\nProcessing {lang}...")

        try:
            data = torch.load(signature_path, map_location="cpu")
        except FileNotFoundError:
            print(f"Missing {signature_path}; skipping.")
            all_results.append([])
            continue

        coords = data["coords"]
        signature = data["signature"]
        print(f"Loaded signature: {signature.shape}")

        relevance_values = []
        for layer_idx, neuron_idx in coords:
            entropy_value = entropy[layer_idx, neuron_idx].item()
            relevance = (
                0.0
                if torch.isinf(torch.tensor(entropy_value))
                else 1.0 - entropy_value / log_num_languages
            )
            relevance_values.append(relevance)

        relevance = torch.tensor(relevance_values, dtype=torch.float32).cuda()
        similarity = calculate_jaccard_similarity_matrix(signature).cuda()
        selected_indices = mrmr_select_jaccard(
            range(len(coords)), relevance, similarity, args.K
        )
        selected_coords = [coords[index] for index in selected_indices]

        layer_results = [[] for _ in range(num_layers)]
        for layer_idx, neuron_idx in selected_coords:
            layer_results[layer_idx].append(neuron_idx)
        layer_results = [
            torch.tensor(sorted(indices), dtype=torch.long)
            for indices in layer_results
        ]

        all_results.append(layer_results)
        print(f"Selected {sum(len(indices) for indices in layer_results)} neurons.")

    torch.save(all_results, args.out_neuron)
    print(f"\nSaved: {args.out_neuron}")


if __name__ == "__main__":
    main()
