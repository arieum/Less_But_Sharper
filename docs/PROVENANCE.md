# Source provenance

This release was assembled from the `Language-Specific-Neurons` working tree
using only experiment sources whose original filesystem modification time was
before 2026-10-01 (Asia/Seoul). Generated logs, checkpoints, caches, unrelated
follow-up experiments, and post-cutoff scripts were excluded.

## File map

| Released path | Original path | Original date |
|---|---|---:|
| `code/data/prepare_wikipedia.py` | `01_collect_wiki_ids.py` | 2026-07-12 |
| `code/lape/collect_activations.py` | `02_collect_activations.py` | 2026-07-12 |
| `code/lape/extract_lape_candidates.py` | `03_extract_lape_neurons.py` | 2026-07-12 |
| `code/selection/collect_token_signatures.py` | `04_collect_token_signatures.py` | 2026-07-12 |
| `code/selection/select_topk.py` | `05_select_lape_topk.py` | 2026-07-12 |
| `code/selection/select_mmr.py` | `06_select_token_mmr.py` | 2026-07-12 |
| `code/selection/select_mrmr.py` | `mrmr_lape_jaccard.py` | 2026-06-02 |
| `code/selection/select_dpp.py` | `dpp_lape_jaccard.py` | 2026-06-08 |
| `code/evaluation/generate_mvicuna.py` | `07_generate_mvicuna.py` | 2026-07-13 |
| `code/evaluation/score_generation.py` | `08_evaluate_generation.py` | 2026-07-13 |
| `code/evaluation/evaluate_ppl.py` | `ppl.py` | 2026-01-04 |
| `code/evaluation/evaluate_mmlu.py` | `eval_mmlu_per_question.py` | 2026-04-10 |
| `code/evaluation/paired_mmlu_test.py` | `paired_test_mmlu.py` | 2026-08-20 |
| `code/evaluation/paired_generation_statistics.py` | `stat_generation_paired.py` | 2026-09-17 |
| `code/analysis/snmf_feature_redundancy.ipynb` | `EMNLP_Rebuttal/analysis1_snmf_comembership.ipynb` | 2026-08-03 |
| `code/analysis/token_level_causal_analysis.ipynb` | `causal_ablation_ppl.ipynb` | 2026-07-13 |

The mask, result, SNMF, and timing files under `artifacts/` are compact final
outputs associated with these experiments. Their content hashes are listed in
`artifacts/MANIFEST.sha256`.

## Packaging edits

The research logic is preserved. The release adds only portability and
repository-quality changes:

- descriptive directory and file names;
- explicit `llama2-7b` and `llama3.1-8b` model tags in shared model tables;
- configurable Wikipedia train/validation article ranges;
- configurable PPL data paths and model tags;
- a cleaned pre-cutoff paired-generation test compatible with both historical
  `judge_score` and released `content_score` fields;
- shell entry points, dependency files, documentation, and smoke checks.

The upstream mVicuna files and base LAPE repository are referenced rather than
copied. The recorded upstream revision is
`fdeca23acd53d7a4de5ca3f7a1b66ef94d7e19a7` from
`RUCAIBox/Language-Specific-Neurons`.

## Exclusion policy

Files were omitted when they were any of the following:

- dated 2026-10-01 or later;
- raw generation/evaluation logs or large intermediate activation tensors;
- downloaded model or dataset caches;
- duplicate exploratory notebooks superseded by the preserved analyses;
- unrelated NAACL/follow-up experiments;
- upstream files that can be fetched directly from the original repository.
