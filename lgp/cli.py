import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np

from .computation import budget_search, cosine_distance
from .data import fingerprint, load_data, validate_graph, write_json
from .graph import diskann_graph, point
from .hnsw import ShortcutConfig, hnsw_graph
from .metrics import evaluate
from .search import centroid_entry, diskann_search


@contextmanager
def build_environment(args):
    old = {k: v for k, v in os.environ.items() if k.startswith("LLM_INDEX_")}
    for key in old:
        del os.environ[key]
    values = {"SELECTOR": "llm_control", "MIN_2HOP_GAIN": args.min_gain,
              "REQUIRE_UNCOVERED": 0,
              "REPLACE_EDGES": 1, "REPLACE_POLICY": "legacy", "PROTECT_LOCAL_EDGES": 0,
              "REPLACE_MIN_GAIN_DELTA": 0, "SHORTCUT_B": args.edits,
              "SHORTCUT_CAND_CAP": args.shortlist, "SELECTED_CONTEXT_CAP": args.context,
              "CANDIDATE_SOURCE": args.shortlisting, "VICTIM_POLICY": args.removal,
              "TRUNCATE_TEXT": 1, "ANCHOR_CHARS": 1200, "KEEP_CHARS": 700,
              "CAND_CHARS": 900, "FAIL_ON_LLM_ERROR": 1,
              "MM_IMAGE_MAX_SIDE": 512, "MM_JPEG_QUALITY": 80}
    os.environ.update({"LLM_INDEX_"+k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith("LLM_INDEX_"):
                del os.environ[key]
        os.environ.update(old)


def client(args):
    from .clients import VLLMClient
    return VLLMClient(args.server, args.model, args.tokenizer,
                      args.max_tokens or (64 if args.multimodal else 256),
                      multimodal=args.multimodal)


def build(args):
    docs, vectors, _, _ = load_data(args.data, queries=False)
    if args.shortlist is None:
        args.shortlist = 12 if args.multimodal else 24
    if not 0 <= args.beta <= 1 or not 0 <= args.edits <= min(args.shortlist, args.degree):
        raise ValueError("Require beta in [0,1] and 0 <= edits <= min(shortlist, degree)")
    if args.context < 0 or args.min_gain < 0:
        raise ValueError("Context and coverage threshold must be nonnegative")
    if args.multimodal and args.index == "hnsw":
        raise ValueError("The HNSW adapter requires text inputs; use DiskANN for multimodal items")
    if args.variant == "lgp" and any(not row.get("text") and not row.get("image") for row in docs):
        raise ValueError("LGP requires text or image content for every document")
    llm = client(args) if args.variant == "lgp" and args.beta > 0 and args.edits > 0 else None
    config = ShortcutConfig(B=args.edits, cand_cap=args.shortlist,
                            min_2hop_gain=args.min_gain,
                            max_calls=math.floor(args.beta * len(docs)),
                            backbone=False, reverse_policy="bounded_hnsw")
    params = {"index": args.index, "variant": args.variant, "degree": args.degree,
              "construction_width": args.construction_width, "alpha": args.alpha,
              "seed": args.seed, "beta": args.beta, "context": args.context,
              "shortlisting": args.shortlisting, "removal": args.removal,
              "multimodal": args.multimodal, "selector": asdict(config),
              "acceptance_rule": "coverage_threshold_v2"}
    with build_environment(args):
        A = [point(row) for row in vectors]
        texts = [row.get("text", "") for row in docs]
        if args.index == "hnsw":
            graph = hnsw_graph(A, args.construction_width, args.degree, args.alpha,
                               texts=texts, seed=args.seed, config=config)
            graph.build_vanilla()
            if args.variant == "lgp" and config.max_calls > 0 and config.B > 0:
                graph = graph.clone_for_refinement(llm, config)
                graph.refine_layer0()
            layers, entry = graph.layers, graph.enter_point
        else:
            Graph = diskann_graph
            kwargs = {}
            if args.multimodal:
                from .multimodal import diskann_graph as Graph
                kwargs["items"] = docs
            graph = Graph(A, args.construction_width, args.degree, args.alpha,
                          llm=llm, texts=texts, fla=2 if args.variant == "lgp" else 0,
                          seed=args.seed, llm_max_calls_per_prune=config.max_calls, **kwargs)
            graph.start = centroid_entry(vectors)
            graph.indexing()
            layers, entry = [graph.edge], graph.start
    state = {"format": "lgp-graph-v1", "index": args.index,
             "data_fingerprint": fingerprint(docs, vectors), "entry": entry,
             "layers": layers, "parameters": params,
             "selector_attempts": graph.debug_stats["num_use_shortcut_calls"]}
    validate_graph(state, docs, vectors)
    if args.index == "diskann" and any(len(row) > args.degree for row in graph.edge):
        raise ValueError("Degree bound violated; graph not published")
    write_json(args.output, state)
    print(json.dumps({"nodes": len(docs), "edges": sum(map(len, layers[0])),
                      "selector_attempts": state["selector_attempts"]}))


def search(args):
    docs, vectors, queries, qvectors = load_data(args.data)
    state = json.loads(Path(args.graph).read_text())
    validate_graph(state, docs, vectors)
    budgets = sorted(set(int(x) for x in args.budgets.split(",")))
    if not budgets or min(budgets) < 1 or args.pool_size < 1:
        raise ValueError("Positive budgets and pool size required")
    pid = {row["id"]: i for i, row in enumerate(docs)}
    rankings, costs = {str(b): {} for b in budgets}, {str(b): [] for b in budgets}
    native = None
    if state["index"] == "hnsw" and args.mode == "width":
        native = object.__new__(hnsw_graph)
        native.v = [point(row) for row in vectors]
        native.layers, native.enter_point = state["layers"], state["entry"]
        native.max_level = len(native.layers)-1
    for query, vector in zip(queries, qvectors):
        excluded = {pid[x] for x in query.get("excluded_ids", []) if x in pid} if args.exclusions == "apply" else set()
        distance = cosine_distance(vector, vectors, hnsw=state["index"] == "hnsw")
        if args.mode == "computation":
            snapshots = budget_search(state["layers"], state["entry"], distance, budgets,
                                      pool_size=args.pool_size, upper_layers=state["index"] == "hnsw", excluded=excluded)
        for budget in budgets:
            if args.mode == "computation":
                found, stats = snapshots[budget]["pool"], snapshots[budget]
            elif native is not None:
                found, stats, _ = native.search(point(vector), k=budget, ef=budget)
                found = [x for x in found if x not in excluded]
            else:
                found, stats = diskann_search(state["layers"][0], state["entry"], distance, budget, excluded)
            rankings[str(budget)][query["id"]] = [docs[x]["id"] for x in found]
            costs[str(budget)].append(stats["distance_evals"])
    metrics = {b: {**evaluate(queries, run), "mean_distance_evals": float(np.mean(costs[b]))}
               for b, run in rankings.items()}
    write_json(args.output, {"format": "lgp-run-v1", "data_fingerprint": state["data_fingerprint"],
                             "query_fingerprint": fingerprint(queries, qvectors),
                             "index": state["index"], "mode": args.mode, "exclusions": args.exclusions,
                             "pool_size": args.pool_size if args.mode == "computation" else "search_width",
                             "rankings": rankings, "metrics": metrics})
    print(json.dumps(metrics, indent=2))


def rerank(args):
    from .rerank import llm_rank, sliding_rerank
    docs, vectors, queries, qvectors = load_data(args.data)
    run = json.loads(Path(args.run).read_text())
    if run.get("format") != "lgp-run-v1" or run.get("stage") == "reranked":
        raise ValueError("Expected an unreranked lgp-run-v1 search output")
    if run["data_fingerprint"] != fingerprint(docs, vectors) or run["query_fingerprint"] != fingerprint(queries, qvectors):
        raise ValueError("Run and input dataset do not match")
    llm = client(args)
    pid = {row["id"]: i for i, row in enumerate(docs)}
    texts = [row.get("text", "") for row in docs]
    cap = args.cap or (64 if args.multimodal else 200)
    if cap < 1:
        raise ValueError("Rerank cap must be positive")
    if args.multimodal:
        from .rerank_multimodal import llm_rank_mm
    outputs, calls = {}, 0
    for budget, rankings in run["rankings"].items():
        outputs[budget] = {}
        if set(rankings) != {q["id"] for q in queries}:
            raise ValueError("Incomplete or mismatched query set")
        for query in queries:
            candidates = [pid[x] for x in rankings[query["id"]][:cap]]
            if args.multimodal:
                rank = lambda batch: llm_rank_mm(llm, query, batch, docs, "task",
                                                 {"task": query.get("instruction", "")})
            else:
                rank = lambda batch: llm_rank(llm, query.get("text", ""), batch, texts)
            ordered, count = sliding_rerank(candidates, rank)
            calls += count
            outputs[budget][query["id"]] = [docs[x]["id"] for x in ordered]
    result = {**run, "stage": "reranked", "rerank_cap": cap, "window": 10, "stride": 5,
              "rerank_calls": calls, "rankings": outputs,
              "metrics": {b: evaluate(queries, r) for b, r in outputs.items()}}
    write_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2))

def main():
    parser = argparse.ArgumentParser(description="Anonymous LGP implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    def inference(p):
        p.add_argument("--server", default="http://127.0.0.1:8000/v1")
        p.add_argument("--model", default="lgp", help="Served model name")
        p.add_argument("--tokenizer", default="Qwen/Qwen3-32B", help="Tokenizer matching the text server")
        p.add_argument("--max-tokens", type=int, help="Default: text 256, multimodal 64")
        p.add_argument("--multimodal", action="store_true")

    p = sub.add_parser("build")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--index", choices=("diskann", "hnsw"), default="diskann")
    p.add_argument("--variant", choices=("vanilla", "lgp"), default="lgp")
    p.add_argument("--degree", type=int, default=16)
    p.add_argument("--construction-width", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edits", type=int, default=2)
    p.add_argument("--shortlist", type=int)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--min-gain", type=int, default=12)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--shortlisting", choices=("balanced", "near", "gain", "diverse"), default="balanced")
    p.add_argument("--removal", choices=("lexicographic", "redundancy", "coverage"), default="lexicographic")
    inference(p)
    p.set_defaults(func=build)
    p = sub.add_parser("search")
    p.add_argument("--data", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("width", "computation"), default="width")
    p.add_argument("--budgets", default="20", help="Comma-separated widths or distance-call caps")
    p.add_argument("--pool-size", type=int, default=20, help="Returned pool size for computation-cap search")
    p.add_argument("--exclusions", choices=("apply", "ignore"), default="apply")
    p.set_defaults(func=search)
    p = sub.add_parser("rerank")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cap", type=int)
    inference(p)
    p.set_defaults(func=rerank)
    from .prepare import add_commands
    add_commands(sub)
    args = parser.parse_args()
    if hasattr(args, "output") and Path(args.output).exists():
        parser.error("Output already exists; choose a new output path")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
