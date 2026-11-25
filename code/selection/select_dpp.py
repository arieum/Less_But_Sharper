import argparse
import math

import torch


LANGS = ["en", "zh", "fr", "es", "vi", "id", "ja"]


def dpp_greedy_map(kernel: torch.Tensor, K: int, eps: float = 1e-6):
    """Select a greedy MAP subset using incremental Cholesky updates."""
    num_candidates = kernel.size(0)
    K = min(K, num_candidates)
    kernel = kernel + eps * torch.eye(num_candidates, device=kernel.device)

    selected = []
    gains = kernel.diag().clone()
    factors = torch.zeros(num_candidates, K, device=kernel.device)

    for step in range(K):
        if selected:
            gains[selected] = -torch.inf
        candidate = gains.argmax().item()
        selected.append(candidate)

        if step == 0:
            factors[:, 0] = kernel[:, candidate] / math.sqrt(
                max(gains[candidate].item(), eps)
            )
        else:
            residual = kernel[:, candidate].clone()
            for column in range(step):
                residual -= factors[:, column] * factors[candidate, column]
            norm = math.sqrt(max(residual[candidate].item(), eps))
            factors[:, step] = residual / norm

        gains -= factors[:, step] ** 2

    return selected


def build_dpp_kernel(signature, relevance, alpha=1.0, eps=1e-6):
    """Build a quality-weighted Jaccard DPP kernel."""
    matrix = signature.float().cuda()
    intersection = torch.matmul(matrix, matrix.T)
    active_counts = matrix.sum(dim=1).unsqueeze(1)
    union = active_counts + active_counts.T - intersection
    similarity = (intersection / (union + 1e-8)).cpu()
    quality = torch.exp(alpha * relevance.cpu())
    kernel = quality.unsqueeze(1) * similarity * quality.unsqueeze(0)
    return kernel.cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig_dir", type=str, default="data")
    parser.add_argument("--model_tag", type=str, default="llama-7b")
    parser.add_argument("--meta_path", type=str, default="LLaMA-2-7B.meta.pth")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out_neuron", type=str, required=True)
    args = parser.parse_args()

    print(f"LAPE + DPP greedy MAP: alpha={args.alpha}, K={args.K}")

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

        relevance = torch.tensor(relevance_values, dtype=torch.float32)
        kernel = build_dpp_kernel(signature, relevance, alpha=args.alpha)
        selected_indices = dpp_greedy_map(kernel, K=args.K)
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
