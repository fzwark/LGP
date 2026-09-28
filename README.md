# LGP: LLM-Guided Graph Pruning for Semantic Retrieval

![Graph ANNS](https://img.shields.io/badge/Graph_ANNS-yellow.svg)
![LLM-Guided Pruning](https://img.shields.io/badge/LLM--Guided_Pruning-blue.svg)
![Semantic Retrieval](https://img.shields.io/badge/Semantic_Retrieval-green.svg)

## Introduction

Graph-based approximate nearest neighbor search (ANNS) is widely used for large-scale semantic search. Its indices are constructed primarily based on geometric relationships among embeddings of an input dataset (e.g., documents or images), rather than explicitly optimizing for semantic relevance. However, when using these indices for downstream query retrieval, performance is evaluated based on the semantic relevance of the retrieved results to the query. This creates a fundamental "geometry-semantic" mismatch between how the indices are constructed and how their retrieval results are evaluated. While existing LLM-based reranking methods can partially mitigate this mismatch at query time, they leave this underlying structural problem in the graph unresolved. We therefore propose LLM-Guided Graph Pruning (LGP), a general framework that addresses this mismatch directly by leveraging LLM reasoning to refine an existing ANN graph index itself. LGP identifies structurally "low-value" neighbors of nodes and replaces them with LLM-selected alternatives that provide useful semantic information while retaining desired geometric structures of the original graph, including sparsity and efficient navigability. Experiments on representative semantic retrieval benchmarks show that LGP consistently improves end-to-end retrieval performance over both vanilla greedy graph search and LLM-based reranking across widely used graph-based ANN indices such as DiskANN and HNSW.

This repository contains the core implementation for graph construction, LGP refinement, graph search, and LLM/VLM reranking.

## Setup

LGP supports Python 3.10--3.12. Create an environment and install the package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[text]'
```

## Prepare Data and Embeddings

For a BRIGHT text-retrieval experiment, run:

```bash
python -m lgp prepare-bright --subset biology --output data/biology
python -m lgp encode --data data/biology --encoder diver --device cuda
```

You can also provide a prepared dataset directory containing `documents.jsonl`, `queries.jsonl`, `passages.npy`, and `queries.npy`. The embedding arrays must be row-aligned, use the `float32` data type, and correspond to the records in the JSONL files.

For M-BEIR multimodal experiments, download the task-local candidate pool, queries, and images, and then inspect the required preparation arguments:

```bash
python -m lgp prepare-mbeir --help
```

Install CLIP and encode the prepared data:

```bash
python -m pip install 'git+https://github.com/openai/CLIP.git'
python -m lgp encode --data data/webqa_task2 --encoder clip --device cuda
```

## Start the Model Server

Install [vLLM](https://docs.vllm.ai/) in a separate GPU environment. Start the text model server and keep it running while building or reranking:

```bash
vllm serve Qwen/Qwen3-32B --served-model-name lgp \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --seed 42
```

Increase `--tensor-parallel-size` when using multiple GPUs. For multimodal experiments, start a vision-language model server instead:

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --served-model-name lgp \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --seed 42 \
  --max-model-len 32768 --limit-mm-per-prompt '{"image":17}'
```

Add `--multimodal` to the corresponding build and rerank commands when using the vision-language model.

## Build Vanilla and LGP Graphs

The following example constructs paired vanilla and LGP DiskANN graphs:

```bash
for variant in vanilla lgp; do
  python -m lgp build --data data/biology --variant "$variant" \
    --index diskann --output runs/biology-${variant}.json
done
```

Use `--index hnsw` to build text-only HNSW graphs. For a controlled comparison, keep the inputs, random seed, and construction parameters identical across the vanilla and LGP variants. Graph operations run on the CPU, while LGP sends its model requests to the running server.

## Search and Rerank

Search each graph and optionally apply LLM reranking to its retrieved candidates:

```bash
for variant in vanilla lgp; do
  python -m lgp search --data data/biology \
    --graph runs/biology-${variant}.json --budgets 10,20,50,100,200 \
    --output runs/biology-${variant}-search.json

  python -m lgp rerank --data data/biology \
    --run runs/biology-${variant}-search.json \
    --output runs/biology-${variant}-rerank.json
done
```

The output reports NDCG@5/10 and document-level Recall@5/10 in percent. To sweep the number of distance computations, add the following arguments to the search command:

```bash
--mode computation --budgets 100,200,300,500,800,1600,3200 --pool-size 20
```

Output files are never overwritten, so use a fresh output path for every run. Run the following command to see all available commands and options:

```bash
python -m lgp --help
python -m lgp <command> --help
```

## Citation

> **TODO:** Add the paper citation after the anonymous review period.
