import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row.get("id") for row in rows]
    if not rows or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("JSONL requires nonempty, unique string IDs")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(rows, vectors):
    h = hashlib.sha256()
    h.update(json.dumps([row["id"] for row in rows], ensure_ascii=False).encode())
    h.update(str(vectors.shape).encode())
    for start in range(0, len(vectors), 4096):
        h.update(np.ascontiguousarray(vectors[start:start+4096], dtype="<f4").tobytes())
    return h.hexdigest()


def _vectors(path, rows, dim=None):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or len(array) != len(rows) or (dim is not None and array.shape[1] != dim):
        raise ValueError("Embedding dimensions/row counts do not match metadata")
    if array.dtype != np.float32:
        raise ValueError("Embeddings must be float32")
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start+4096]).all():
            raise ValueError("Embeddings contain non-finite values")
    return array


def load_data(directory, queries=True):
    root = Path(directory)
    docs = read_jsonl(root / "documents.jsonl")
    vectors = _vectors(root / "passages.npy", docs)
    query_rows, query_vectors = [], None
    if queries:
        query_rows = read_jsonl(root / "queries.jsonl")
        query_vectors = _vectors(root / "queries.npy", query_rows, vectors.shape[1])
        doc_ids = {row["id"] for row in docs}
        for row in query_rows:
            relevant = row.get("relevant_ids", [])
            excluded = row.get("excluded_ids", [])
            if not isinstance(relevant, list) or not isinstance(excluded, list):
                raise ValueError("relevant_ids/excluded_ids must be lists")
            if not set(relevant) <= doc_ids:
                raise ValueError("A relevant document is absent from this candidate pool; check task/qrels alignment")
            if set(relevant) & set(excluded):
                raise ValueError("Relevant and excluded document IDs overlap")
    for row in docs + query_rows:
        if row.get("image"):
            image = Path(row["image"])
            if not image.is_absolute():
                image = root / image
            if not image.is_file():
                raise FileNotFoundError("A declared image does not exist")
            row["image"] = str(image)
    return docs, vectors, query_rows, query_vectors


def validate_graph(state, docs, vectors):
    if state.get("format") != "lgp-graph-v1" or state.get("index") not in ("diskann", "hnsw"):
        raise ValueError("Unsupported graph format")
    if state.get("data_fingerprint") != fingerprint(docs, vectors):
        raise ValueError("Graph and document IDs/embeddings do not match")
    layers = state["layers"]
    if not layers or not 0 <= state["entry"] < len(docs):
        raise ValueError("Invalid graph entry/layers")
    for rows in layers:
        if len(rows) != len(docs):
            raise ValueError("Graph size mismatch")
        for node, row in enumerate(rows):
            if len(set(row)) != len(row) or any(type(x) is not int or x == node or not 0 <= x < len(docs) for x in row):
                raise ValueError("Graph has duplicate, self, or invalid edges")
