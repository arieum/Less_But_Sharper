# Released artifacts

- `masks/`: per-language, per-layer neuron indices for the reported selectors
  and backbones;
- `results/`: paired comparison tables and aggregate MMLU outputs;
- `snmf/`: feature co-membership, rank-sweep, occupancy, and random-baseline
  summaries;
- `timing/`: reported pipeline timing summary.

PyTorch mask files contain either the historical nested list representation or
a dictionary with a `neuron` entry. All released evaluators accept both forms.
Use `sha256sum -c artifacts/MANIFEST.sha256` from the repository root to verify
the files.
