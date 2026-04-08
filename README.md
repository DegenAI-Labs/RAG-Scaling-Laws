# To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining

[Karan Singh](https://karanps.com), Michael Yu, [Varun Gangal](https://vgtomahawk.github.io), Zhuofu Tao, [Sachin Kumar](https://sites.google.com/view/sachinkumar?pli=1), [Emmy Liu](https://nightingal3.github.io), [Steven Y. Feng](https://styfeng.github.io)

**Stanford University, Patronus AI, The Ohio State University, Carnegie Mellon University, and DegenAI Labs**

<hr>

![DCLM data ranging in quantity from 30M to 100B trains OLMo-2 series models augmented by retrieval augmented generation. We investigate the optimal RAG frontier, i.e. the amount of pretraining tokens vs. RAG tokens that you should use for a given data budget, showing that small models benefit more from RAG and exhibit consistent scaling as the amount of retrieval data is increased, whereas larger models saturate quickly.](figures/intro.png)

**Figure: Trade-off between pretraining and retrieval under a fixed data budget.** _Left_: We train OLMo-2 models ranging from 30M to 3B parameters on DCLM data while constructing retrieval stores from held-out portions of the same corpus. _Center_: We conceptualize this as an optimization problem over a 2D allocation space of pretraining and retrieval tokens. For a fixed data budget, feasible configurations lie along a constraint frontier, and performance varies smoothly; our goal is to identify the optimal allocation along this frontier. _Right_: Retrieval allocation trade-off at fixed pretraining scale. As the % of data used for retrieval increases, performance changes non-monotonically, with scale dependence: smaller models benefit most, while larger models exhibit diminishing returns and over-allocation sensitivity.

Preprint: [arXiv](https://arxiv.org/pdf/2604.00715)

## Abstract

Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. However, the relationship between parametric knowledge acquired during pretraining and non-parametric knowledge accessed via retrieval remains poorly understood, especially under fixed data budgets. In this work, we systematically study the trade-off between pretraining corpus size and retrieval store size across a wide range of model and data scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B tokens of DCLM data, while varying both pretraining data scale (1-150x the number of parameters) and retrieval store size (1-20x), and evaluate performance across a diverse suite of benchmarks spanning reasoning, scientific QA, and open-domain QA. We find that retrieval consistently improves performance over parametric-only baselines across model scales and introduce a three-dimensional scaling framework that models performance as a function of model size, pretraining tokens, and retrieval corpus size. This scaling manifold enables us to estimate optimal allocations of a fixed data budget between pretraining and retrieval, revealing that the marginal utility of retrieval depends strongly on model scale, task type, and the degree of pretraining saturation. Our results provide a quantitative foundation for understanding when and how retrieval should complement pretraining, offering practical guidance for allocating data resources in the design of scalable language modeling systems.

## Overview

This repository contains utilities for:
- preparing split datasets for pretraining-vs-retrieval experiments
- launching pretraining and evaluation jobs
- fitting and analyzing scaling-law trends from aggregated CSV metrics

## Repository Layout

- `data/scripts/`: data download, split, and optimization utilities
- `data/splits/`: canonical shard split CSVs for pretraining/retrieval
- `scripts/eval/`: evaluation orchestration and result aggregation
- `scripts/rag/`: RAG index build scripts and FAISS docs
- `scripts/pretraining/`: pretraining scheduler scripts (`.sbatch`)
- `scripts/litgpt_to_hf/`: LitGPT checkpoint conversion utilities
- `scripts/fit_scaling_law/`: scaling-law fitting + diagnostics

## Getting Started

### 1) Environment

Use Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you use Weights & Biases:

```bash
wandb login
```

### 2) Initialize submodules

```bash
git submodule update --init --recursive
```

### 3) Configure paths via environment variables

```bash
export MODELS_ROOT="/path/to/pretrained/models"
export INDICES_ROOT="/path/to/rag/indices"
export LITGPT_DIR="/path/to/litgpt"
export HF_HOME="/path/to/hf-cache"
```

---

## Workflows

### Dataset split and optimization

See `data/DATA_README.md` and the scripts in `data/scripts/`.

End-to-end helper:

```bash
bash data/scripts/create_datasets.sh
```

### Pretraining

Pretraining job templates are in `scripts/pretraining/`.

Example (submit a 30M pretraining sweep):

```bash
sbatch scripts/pretraining/olmo2_30m.sbatch
```

You can similarly launch other scales:
- `scripts/pretraining/olmo2_136m.sbatch`
- `scripts/pretraining/olmo2_136m_seed43.sbatch`
- `scripts/pretraining/olmo2_233m.sbatch`
- `scripts/pretraining/olmo2_728m.sbatch`
- `scripts/pretraining/olmo2_1b.sbatch`
- `scripts/pretraining/olmo2_3b.sbatch`

### RAG index generation

RAG index build utilities are in `scripts/rag/`.

Primary index build script:

```bash
bash scripts/rag/build_ratioed_indices.sh
```

Supporting files:
- `scripts/rag/build_ratioed_indices.py`
- `scripts/rag/FAISS_ON_GPU.md`

### Multi-index RAG evaluation

Main runner:

```bash
bash scripts/eval/run_all_evals.sh
```

This script:
- prepares eval task inputs
- generates retrieval JSONL files for each index/task
- runs baseline and RAG evaluation for each model scale

### Scaling-law fitting

```bash
python3 scripts/fit_scaling_law/fit_scaling_law.py \
  --dir /path/to/aggregated_csvs \
  --metric "perplexity,none" \
  --mode sequential \
  --retrieval_model log
```

Related analysis:
- Sigma/kappa analysis: `scripts/fit_scaling_law/detect_saturation.py`
- Robustness/stability to random seeds: `scripts/fit_scaling_law/seed_stability_study.py`

## Citation

If you found our work helpful for your own research or applications, please cite it using the following BibTeX:
```bibtex
@misc{RAGScalingLaws2026,
      title={To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining}, 
      author={Karan Singh and Michael Yu and Varun Gangal and Zhuofu Tao and Sachin Kumar and Emmy Liu and Steven Y. Feng},
      year={2026},
      eprint={2604.00715},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.00715}, 
}
```