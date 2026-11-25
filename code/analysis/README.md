# Analysis notebooks

The notebooks are intentionally saved with execution outputs from the paper
analyses. They locate the repository root automatically, load released masks
from `artifacts/masks/`, and write caches and results under `work/analysis/`.

The causal notebook looks for regenerated token IDs and signatures in this
package first. When run inside the original `Language-Specific-Neurons`
workspace, it also recognizes the legacy files in the parent `data/` folder.

`snmf_feature_redundancy.ipynb` contains the semi-NMF decomposition and rank
sweep. `token_level_causal_analysis.ipynb` contains the Japanese script and
French token-level causal interventions described in the paper appendix.
