# Reproducibility protocol

Run all commands from the repository root. The examples below reproduce the
main Llama-2-7B experiment; replace `llama2-7b` with `opt-6.7b` or
`mistral-7b-v0.2` for the cross-backbone study. The appendix's Llama-3.1-8B
robustness setting uses `llama3.1-8b`; its released K=40 masks are also under
`artifacts/masks/`.

## 1. Prepare data

Fetch the upstream mVicuna prompts:

```bash
bash scripts/fetch_upstream.sh
```

Tokenize the November 1, 2023 Wikipedia snapshots. The paper's activation and
signature analyses use the training stream; the second command creates a
non-overlapping stream for PPL evaluation.

```bash
python code/data/prepare_wikipedia.py \
  --model_tag llama2-7b --wiki_date 20231101 \
  --split_name train --max_articles 2000 --data_dir data

python code/data/prepare_wikipedia.py \
  --model_tag llama2-7b --wiki_date 20231101 \
  --split_name valid --skip_articles 2000 --max_articles 2000 --data_dir data
```

The scripts consume contiguous tokens in dataset order. Exact token counts can
vary if the dataset or tokenizer revision changes; record the resolved Hugging
Face revisions for a strict rerun.

## 2. LAPE candidate pool

Collect positive-activation counts for each language, then select the 1% of
neurons with the lowest language entropy:

```bash
bash scripts/run_lape.sh
```

This creates:

- `work/llama2-7b/llama2-7b.neuron.pth`: per-language, per-layer candidates;
- `work/llama2-7b/llama2-7b.meta.pth`: entropy, activation, and architecture
  metadata required by all selectors.

This stage implements the method introduced by
[Language-Specific Neurons](https://github.com/RUCAIBox/Language-Specific-Neurons).
The generalized local wrappers add OPT and Mistral MLP support but do not
change the LAPE definition.

## 3. Token-level signatures and selection

The paper records a binary activation signature on 30,000 contiguous tokens
for every candidate neuron. Run Top-K and the main MMR configuration with:

```bash
bash scripts/run_main_selection.sh
```

Defaults are `K=30`, `lambda=0.6`, and CUDA. A CPU-only smoke run can set
`DEVICE=cpu`, although the full signature stage still requires enough memory
to load the model.

The paper sweeps `K` over `{10,20,...,90}` and `lambda` over
`{0.1,0.2,...,0.9}`. For example:

```bash
for k in 10 20 30 40 50 60 70 80 90; do
  for lambda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    K="$k" LAMBDA="$lambda" bash scripts/run_main_selection.sh
  done
done
```

Signatures are independent of `K` and `lambda`; for a production sweep,
collect them once and invoke `select_mmr.py` directly to avoid repeated model
forwards.

Alternative selectors use the same signatures and metadata:

```bash
python code/selection/select_mrmr.py \
  --sig_dir work/llama2-7b/signatures --model_tag llama2-7b \
  --meta_path work/llama2-7b/llama2-7b.meta.pth --K 30 \
  --out_neuron work/llama2-7b/masks/mrmr_k30.pth

python code/selection/select_dpp.py \
  --sig_dir work/llama2-7b/signatures --model_tag llama2-7b \
  --meta_path work/llama2-7b/llama2-7b.meta.pth --K 30 --alpha 1.5 \
  --out_neuron work/llama2-7b/masks/dpp_k30_alpha1.5.pth
```

## 4. Selective PPL ablation

The historical evaluator patches vLLM's internal MLP forward method and is
therefore pinned separately in `requirements-vllm.txt`.

```bash
python code/evaluation/evaluate_ppl.py \
  --model meta-llama/Llama-2-7b-hf --model_tag llama2-7b \
  --data_dir data \
  --activation_mask artifacts/masks/llama2_mmr_k30_lambda0.6.pth
```

Each row in the output corresponds to one ablated target-language mask and
each column to one evaluation language. A successful selector should increase
PPL most strongly for its target language.

## 5. mVicuna generation and evaluation

Generate with the released main masks:

```bash
python code/evaluation/generate_mvicuna.py \
  --model_tag llama2-7b \
  --activation_masks artifacts/masks/llama2_topk_k30.pth \
                     artifacts/masks/llama2_mmr_k30_lambda0.6.pth \
  --include_baseline --mode steer --scale 1.3 \
  --data_dir data/mvicuna --output_root work/generation
```

`generate_mvicuna.py` evaluates the six non-English target languages by
default. Use `--langs` to change this set. It supports zero ablation with
`--mode prune` and activation scaling with `--mode steer`.

Score output-language accuracy with fastText LID and content/language quality
with an OpenAI-compatible local judge:

```bash
python code/evaluation/score_generation.py \
  --input_dir work/generation --output_root work/evaluation \
  --ft_model_path /path/to/lid.176.bin \
  --base_url http://localhost:8001/v1 --judge_model llama-judge
```

Run paired per-language tests after arranging one scored file per language and
method:

```bash
python code/evaluation/paired_generation_statistics.py \
  --topk_template 'work/evaluation/{lang}.topk.json' \
  --mmr_template 'work/evaluation/{lang}.mmr.json' \
  --out_json work/evaluation/paired_topk_mmr.json
```

## 6. MMLU

The evaluator runs the 5-shot MMLU task through lm-eval and writes both
aggregate JSON and per-question records:

```bash
python code/evaluation/evaluate_mmlu.py \
  --model meta-llama/Llama-2-7b-hf \
  --tag baseline --batch_size 2 --out_dir work/mmlu

python code/evaluation/evaluate_mmlu.py \
  --model meta-llama/Llama-2-7b-hf \
  --mask artifacts/masks/llama2_mmr_k30_lambda0.6.pth \
  --tag mmr_k30_lambda0.6 --batch_size 2 --out_dir work/mmlu
```

Use `paired_mmlu_test.py` on the resulting per-question CSV files:

```bash
python code/evaluation/paired_mmlu_test.py \
  --base work/mmlu/baseline_mmlu_per_question.csv \
  --conditions work/mmlu/mmr_k30_lambda0.6_mmlu_per_question.csv
```

Released aggregate outputs are under `artifacts/results/`.

## 7. SNMF and causal analyses

The preserved notebooks are:

- `code/analysis/snmf_feature_redundancy.ipynb` — semi-NMF feature
  co-membership, rank sweep (`F=30,50,100`), WTA thresholding, and the random
  baseline;
- `code/analysis/token_level_causal_analysis.ipynb` — Japanese script-specific
  and French token-level causal interventions.

The notebooks are distributed without cached outputs or execution counts.
They detect the repository root, use masks from `artifacts/masks/`, and write
new outputs under `work/analysis/`. The causal notebook accepts either the
new data/signature names or the original workspace's legacy files. Released
summary tables are in `artifacts/snmf/`.

## Expected resource profile

The paper reports approximately 8.1 hours for activation collection, 31.3
seconds for signature processing, 0.36 seconds for a Jaccard matrix, and 19.1
seconds for an 81-point selection grid in the reported environment. Model
loading, generation, and judge inference dominate end-to-end wall time and
depend strongly on GPU type and serving configuration.
