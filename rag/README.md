# RAG retrieval corpus and FAISS indices

This directory implements **dense retrieval** over a DCLM-derived corpus: chunking with TikToken, embedding with a Qwen embedding model, **L2-normalized** vectors, and **FAISS** indices (including **IVFPQ** at scale). The tables below match **Appendix A.2** of the paper (embedding/chunking, global FAISS settings, and per-split scaling).

---

## Paper methodology (Appendix A.2)

### A.2.1 Retrieval corpus

- **Source:** Held-out portion of the DCLM dataset (documents used only for retrieval, not pretraining).
- **Pipeline:** Documents are chunked into overlapping token windows, embedded with a frozen pretrained encoder, **L2-normalized**, and indexed with FAISS. Separate indices are built for each retrieval scale (e.g. 5% / 10% / 20% / 30% of tokens allocated to RAG).

### A.2.2 Embedding, chunking, and tokenization

We compared several encoders before selecting **Qwen3-Embedding-8B** for strong semantic recall on public RAG benchmarks.

**Table 4 — Embedding and chunking (paper)**

| Component | Configuration |
|-----------|----------------|
| Embedding model | Qwen3-Embedding-8B |
| Embedding dimension | 7168 |
| Pooling | Last non-padded token |
| Normalization | L2-normalized |
| Tokenizer (chunking) | `cl100k_base` (TikToken) |
| Chunk length | 900 tokens |
| Stride | 256 tokens (~28% overlap) |

**Table 5 — FAISS index (paper)**

| Component | Configuration |
|-----------|----------------|
| Index type | IVFPQ (inner product on unit vectors) |
| Sub-quantizers | 128 |
| Bits per code | 8 |
| Search | `nprobe = 64` |
| Training set size | \(\max(n_{\mathrm{list}} \times 50, 10^5)\) |
| Batch size (add) | 500K vectors |

**Table 6 — Per-split FAISS configuration (paper)**

| Corpus split (RAG %) | \(n_{\mathrm{list}}\) | Shards |
|----------------------|------------------------|--------|
| 5% | 32,768 | 32 |
| 10% | 32,768 | 64 |
| 20% | 65,536 | 128 |
| 30% | 65,536 | 128 |

---

## Code layout

| File | Purpose |
|------|---------|
| [`build_index.py`](build_index.py) | End-to-end index build: read deduped JSONL from a directory, **TikToken** `cl100k_base` chunking (`chunk_len` / `stride`), embed, L2-normalize, build FAISS (`flat`, `ivf`, `ivfpq`, `hnsw`). Supports **parallel shard** runs (`--shard_id` / `--num_shards`) and **`--merge_shards`** to concatenate shard embeddings and build one index. |
| [`ratioed_rag_indices.py`](ratioed_rag_indices.py) | Build **multiple retrieval scales** from merged or sharded embedding outputs: token-budget targets (ratios or absolute token counts), `cl100k_base` token accounting, IVFPQ defaults aligned with large-scale runs (`nlist=32768`, `pq_m=128`, `pq_nbits=8`, `nprobe=64`), optional GPU FAISS and parallel jobs. |
| [`subsample_index.py`](subsample_index.py) | **Subsample** an existing merged index (e.g. derive smaller RAG % from a larger build) with a fixed random seed; rebuilds IVFPQ with configurable `nlist` / PQ / `nprobe`. |
| [`compare_embeddings.py`](compare_embeddings.py) | Benchmark **multiple embedding models** and FAISS index types on a subset of docs (timing, size, optional recall vs flat). Default chunking in this script is **512/256** unless overridden—use **`--chunk_len 900 --stride 256`** to match Table 4. |

---

## Analysis: embedding models and index types

This section summarizes how we **compare encoders and FAISS variants** before committing to full-scale DCLM builds. The paper reports **Qwen3-Embedding-8B**; the tooling below is what we used to justify that choice and to pick **IVFPQ** settings under storage and latency constraints.

### Embedding models (ablations)

**Script:** [`compare_embeddings.py`](compare_embeddings.py) sweeps Hugging Face encoders on the same chunked corpus (filtered by `--retrieval_manifest`), embeds all chunks, then builds indices. Default model list in code includes:

| Model | Typical dim | Speed | Quality / role |
|-------|-------------|-------|----------------|
| **Qwen3-Embedding-8B** | 7168 | Slower | Strong semantic recall; paper default |
| `BAAI/bge-base-en-v1.5` | 768 | Fast | Strong English baseline, smaller footprint |
| `BAAI/bge-small-en-v1.5` | 384 | Faster | Lightweight baseline |
| `google/embeddinggemma-300m` | (ST wrapper) | Fast | Good for latency / smaller-GPU ablations |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Fast | Classic dense baseline |

**What to look at:** embedding time (chunks/s), peak RAM, embedding dimension, and downstream **recall@k** of approximate indices vs a **flat** index on held-out query vectors (the script samples query embeddings from the same matrix). For publication-matched chunking, pass `--chunk_len 900 --stride 256`.

### FAISS index types: when to use what

All dense indices use **inner product** on **L2-normalized** vectors (equivalent to cosine similarity). Tradeoffs at a high level:

| Index type | Best for | Pros | Cons | Memory (rule of thumb) |
|------------|----------|------|------|-------------------------|
| **Flat** (`IndexFlatIP`) | Small corpora, ground truth | Exact nearest neighbor, no training | Slow at millions+ vectors | Full \(N \times d\) float32 store |
| **IVF** (`IndexIVFFlat`) | Medium scale | Fast search with `nprobe` | Needs training; approximate | Similar to flat for vectors; adds coarse structure |
| **HNSW** (`IndexHNSWFlat`) | Medium–large when recall matters | Very fast search, good recall | Higher RAM than IVF/IVFPQ; build cost | Higher than IVF; tune `M`, `efSearch` |
| **IVFPQ** | Large DCLM slices | Compressed; scalable | Lower recall unless tuned; needs training | Much smaller than flat (PQ compression) |

**Rule of thumb for DCLM-scale RAG %:** use **IVFPQ** for multi–billion-token retrieval slices (paper Tables 5–6). Use **flat** or **HNSW** on **small** dev subsets to debug embedding quality without PQ noise.

**Rough memory scaling:** index RAM scales with **number of vectors × dimension** for flat variants; **IVFPQ** is dominated by PQ codes and coarse quantizer. For the same \(N\), **7168-d** vectors need ~**9×** the raw storage of **768-d** (order-of-magnitude sanity check when budgeting).

### Running systematic comparisons

`compare_embeddings.py` requires a **retrieval manifest** (JSONL with `id` per document) and a **deduped** JSONL with matching `id` fields so only retrieval-held-out docs are chunked.

```bash
python3 rag/compare_embeddings.py \
  --retrieval_manifest data/splits/retrieval_manifest.jsonl \
  --deduped_data data/processed/dclm_deduped.jsonl \
  --out_dir data/rag/comparison \
  --chunk_len 900 \
  --stride 256 \
  --max_docs 5000 \
  --num_queries 100 \
  --models \
    Qwen/Qwen3-Embedding-8B \
    BAAI/bge-base-en-v1.5 \
    google/embeddinggemma-300m \
  --index_types flat ivf hnsw ivfpq
```

It reports **build time**, **search latency** (mean / percentiles), and **recall@k** of each approximate index relative to the flat index on the sampled queries. Use that to choose **encoder** and **index family** before launching full `build_index.py` / `ratioed_rag_indices.py` jobs.

### Scaling heuristics (DCLM)

Retrieval **token budget** and **chunk** hyperparameters determine vector count \(N\). Very roughly, for ~900-token chunks with overlap, total chunks scale with **retrieval tokens ÷ effective tokens per chunk** (overlap reduces effective stride). The paper builds **separate indices per RAG %** (Table 6: shards and \(n_{\mathrm{list}}\) grow with corpus size). For intermediate sizes, `ratioed_rag_indices.py` may **reduce** requested `nlist` when \(N\) is too small—watch the log lines for **effective nlist**.

---

## Implementation notes (paper vs CLI defaults)

- **Chunking for indices:** `build_index.py` uses **`tiktoken` `cl100k_base`**, default **`--chunk_len 900 --stride 256`**, matching Table 4.
- **Embedding model name:** The paper uses **Qwen3-Embedding-8B**. In code, pass e.g. `--encoder Qwen/Qwen3-Embedding-8B` (or your exact Hub id). `build_index.py` applies **last non-padded token** pooling and **L2 normalization** for Qwen3-style embedding checkpoints. Other models use masked mean pooling unless using the SentenceTransformer path (e.g. Gemma).
- **Default encoder in `build_index.py`:** The argparse default may differ from the paper (check `--help`); always set **`--encoder`** explicitly for publication parity.
- **`ratioed_rag_indices.py`:** Defaults target large IVFPQ builds (e.g. `nlist=32768`, `pq_m=128`, `nprobe=64`). For small corpora, `choose_effective_nlist` may **cap** \(n_{\mathrm{list}}\) based on vector count—see logs when building tiny test indices.
- **Query-time encoding:** At retrieval, encode queries with the **same** encoder and pooling as in `build_index.py`. Passage chunks use **`max_length=900`** in `embed_texts`; use a long enough query context window so evaluation questions are not truncated unintentionally.
- **Scripts / `data/` paths:** Older notes referred to `scripts/run_build_index.sh` and manifests under `data/splits/`; the current `build_index.py` expects **`--deduped_data_dir`** pointing at a folder of **`.jsonl`** files. Adapt paths to your layout.

---

## Dependencies

Install project requirements (e.g. `pip install -r requirements.txt`). Typical RAG stack pieces:

- `torch`, `transformers` (with `trust_remote_code=True` for Qwen embedding models)
- `faiss-cpu` or `faiss-gpu`
- `tiktoken`
- `pyyaml`, `tqdm`
- Optional: `sentence-transformers` (e.g. Gemma embedding), `psutil` (`compare_embeddings.py`)

---

## Quick start: build an index

From a directory of deduped DCLM-style JSONL (each line a JSON object with a `text` field):

```bash
python3 rag/build_index.py \
  --deduped_data_dir /path/to/deduped_jsonl_dir \
  --out_dir /path/to/index_out \
  --encoder Qwen/Qwen3-Embedding-8B \
  --chunk_len 900 \
  --stride 256 \
  --index_type ivfpq \
  --nlist 32768 \
  --pq_m 128 \
  --pq_nbits 8 \
  --nprobe 64 \
  --batch_size 64
```

Shard large corpora across jobs, then merge:

```bash
# Per-shard (example: shard 0 of 32)
python3 rag/build_index.py \
  --deduped_data_dir /path/to/jsonl \
  --out_dir /path/to/shard_work \
  --shard_id 0 --num_shards 32 \
  ... same embedding / chunk flags ...

# After all shards finish
python3 rag/build_index.py \
  --deduped_data_dir /path/to/jsonl \
  --out_dir /path/to/shard_work \
  --merge_shards \
  --index_type ivfpq \
  --nlist 32768 --pq_m 128 --pq_nbits 8 --nprobe 64
```

Artifacts per index directory include:

- `faiss.index`
- `chunks_meta.jsonl`, `chunk_texts.jsonl` (text for debugging / eval)
- `stats.yaml` (encoder id, dim, chunk hyperparameters, index type)

---

## Query-time retrieval (outside this folder)

Built artifacts are standard **FAISS** indices plus metadata. Load `faiss.index` with `faiss.read_index`, join `chunks_meta.jsonl` / `chunk_texts.jsonl` by `chunk_id`, and encode queries with the same Hugging Face model and pooling as in **`embed_texts`** in `build_index.py`. Downstream training and eval scripts handle batch retrieval and export to the harness.

---

## Troubleshooting

- **CUDA OOM during embedding:** Lower `--batch_size` in `build_index.py` (and in `compare_embeddings.py` if you edit batch size there).
- **Slow search:** Prefer IVF/HNSW/IVFPQ over flat at scale; for IVF/IVFPQ increase `nprobe` (recall–speed tradeoff); for HNSW increase `efSearch` in code if exposed.
- **IVFPQ recall low:** Increase `--nprobe`, or \(n_{\mathrm{list}}\) / PQ bits within budget; validate with `compare_embeddings.py` recall vs `flat` on a sample.
- **FAISS training size:** `build_faiss_index` uses at least \(\max(n_{\mathrm{list}} \times 50, 10^5)\) vectors for IVF/IVFPQ training when the corpus is large enough, consistent with Table 5.

---
