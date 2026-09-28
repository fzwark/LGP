import gzip
import json
from pathlib import Path
import random

import numpy as np

from .data import read_jsonl, write_json, write_jsonl

BRIGHT_INSTRUCTIONS = {
    "biology": "Represent this Biology post for searching relevant passages: ",
    "earth_science": "Represent this Earth Science post for searching relevant passages: ",
    "economics": "Represent this Economics post for searching relevant passages: ",
    "psychology": "Represent this Psychology post for searching relevant passages: ",
    "robotics": "Represent this Robotics post for searching relevant passages: ",
    "stackoverflow": "Represent this Stack Overflow post for searching relevant passages: ",
    "sustainable_living": "Represent this Sustainable Living post for searching relevant passages: ",
    "leetcode": "Represent this Coding problem for searching relevant examples: ",
    "pony": "Represent this Pony question for searching relevant passages: ",
    "aops": "Represent this Math Problem for searching relevant examples: ",
    "theoremqa_theorems": "Represent this Math Problem for searching relevant examples: ",
    "theoremqa_questions": "Represent this Math Problem for searching relevant examples: ",
}

MBEIR_INSTRUCTIONS = {
    "visualnews_task0": "Identify news-related image match with the description.",
    "mscoco_task0": "Find an everyday image match with caption.",
    "fashion200k_task0": "Based on fashion description, retrieve matched image.",
    "webqa_task1": "Find a paragraph from Wikipedia to answer the question.",
    "edis_task2": "Find a news image matching with the caption.",
    "webqa_task2": "Find a Wiki image that answers the question.",
    "visualnews_task3": "Provide a news-related caption for the displayed image.",
    "mscoco_task3": "Find a caption describing the image.",
    "fashion200k_task3": "Find a description for the fashion item in the image.",
    "nights_task4": "Find an image that is identical to the given image.",
    "oven_task6": "Retrieve a Wiki text that answers the given query about the image.",
    "infoseek_task6": "Find an article that answers the given question about the image.",
    "fashioniq_task7": "Find an image to match the fashion image and style note.",
    "cirr_task7": "I'm looking for a similar everyday image with the described changes.",
    "oven_task8": "Find a Wiki image-text pair to answer a question regarding an image.",
    "infoseek_task8": "Find a Wiki image-text pair to answer my question about this image.",
}


def prepare_bright(args):
    from datasets import load_dataset
    kwargs = {"revision": args.revision} if args.revision else {}
    documents = load_dataset("xlangai/BRIGHT", "documents", split=args.subset, **kwargs)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=args.subset, **kwargs)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    write_jsonl(root / "documents.jsonl", [{"id": str(row["id"]), "text": row["content"]} for row in documents])
    write_jsonl(root / "queries.jsonl", [{"id": str(row["id"]), "text": row["query"],
                 "relevant_ids": list(row["gold_ids"]), "excluded_ids": list(row.get("excluded_ids", []))}
                 for row in examples])
    write_json(root / "dataset.json", {"benchmark": "bright", "subset": args.subset,
                                       "dataset_revision": args.revision or "unversioned"})


def _raw_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_mbeir(args):
    candidates, queries = _raw_rows(args.candidates), _raw_rows(args.queries)
    image_root = Path(args.image_root).resolve()
    pool = {row["did"] for row in candidates}
    prefix = {"oven_task6": "5", "infoseek_task6": "6"}.get(args.task)
    if prefix and any(x.partition(":")[0] != prefix for x in pool):
        raise ValueError("Task 6 requires the matching dataset-local text candidate pool")

    def item(row, query=False):
        modality = str(row.get("query_modality" if query else "modality", "")).lower()
        if not modality:
            raise ValueError("M-BEIR item is missing its modality")
        image = row.get("query_img_path" if query else "img_path", "") or ""
        text = row.get("query_txt" if query else "txt", "") or ""
        result = {"id": row["qid" if query else "did"],
                  "text": text if "text" in modality else "", "has_text": "text" in modality}
        if "image" in modality:
            if not image or not (image_root / image).is_file():
                raise FileNotFoundError("Extract the declared M-BEIR images before preparing the task")
            result["image"] = str(image_root / image)
        return result

    if args.query_limit < 0:
        raise ValueError("Query limit must be nonnegative (0 means all)")
    if args.query_limit and len(queries) > args.query_limit:
        chosen = sorted(random.Random(args.seed).sample(range(len(queries)), args.query_limit))
        queries = [queries[i] for i in chosen]
    docs = [item(row) for row in candidates]
    if prefix and any(row.get("image") or not row.get("has_text") for row in docs):
        raise ValueError("OVEN/InfoSeek Task 6 candidates must be the released text corpus")
    qrows, dropped = [], 0
    for row in queries:
        q = item(row, query=True)
        positive = [x if isinstance(x, str) else x["did"] for x in row.get("pos_cand_list", [])]
        relevant = [x for x in positive if x in pool and (not prefix or x.partition(":")[0] == prefix)]
        dropped += len(positive) - len(relevant)
        q.update(relevant_ids=list(dict.fromkeys(relevant)), excluded_ids=[], instruction=MBEIR_INSTRUCTIONS[args.task])
        qrows.append(q)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    write_jsonl(root / "documents.jsonl", docs)
    write_jsonl(root / "queries.jsonl", qrows)
    write_json(root / "dataset.json", {"benchmark": "mbeir", "task": args.task,
               "qrels": "task_local_pool", "out_of_pool_positive_annotations": dropped,
               "sampling_seed": args.seed, "query_limit": args.query_limit})
    print(json.dumps({"documents": len(docs), "queries": len(qrows), "out_of_pool_positives": dropped}))


def encode(args):
    root = Path(args.data)
    targets = [root / "passages.npy", root / "queries.npy", root / "encoding.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Encoding outputs already exist; use a fresh data directory")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    documents, queries = read_jsonl(root / "documents.jsonl"), read_jsonl(root / "queries.jsonl")
    metadata = json.loads((root / "dataset.json").read_text()) if (root / "dataset.json").exists() else {}
    if args.encoder == "clip":
        arrays = _encode_clip(documents, queries, root, args)
        model_name = "ViT-B/32"
    else:
        from sentence_transformers import SentenceTransformer
        model_name = "AQ-MedAI/Diver-Retriever-4B-1020" if args.encoder == "diver" else "Qwen/Qwen3-Embedding-4B"
        kwargs = {"device": args.device}
        if args.revision:
            kwargs["revision"] = args.revision
        if args.encoder == "qwen3":
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        model = SentenceTransformer(model_name, **kwargs)
        passage_texts = [row.get("text", "") for row in documents]
        query_texts = [row.get("text", "") for row in queries]
        if metadata.get("benchmark") == "bright":
            passage_texts = [(" " + text).replace("[", "").replace("]", "") for text in passage_texts]
            query_texts = [BRIGHT_INSTRUCTIONS[metadata["subset"]] + text for text in query_texts]
        options = {"batch_size": args.batch_size, "normalize_embeddings": True,
                   "convert_to_numpy": True, "show_progress_bar": True}
        arrays = [model.encode(passage_texts, **options), model.encode(query_texts, prompt_name="query", **options)]
    for path, array in zip(targets, arrays):
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) <= 1e-12).any():
            raise ValueError("Encoder produced nonfinite or zero embeddings; no silent replacement is allowed")
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
    write_json(targets[2], {"encoder": args.encoder, "model": model_name,
               "revision": args.revision or "unversioned", "dtype": "float32",
               "normalized": True, "clip_fusion": "l2_each_then_sum_then_l2" if args.encoder == "clip" else None})


def _encode_clip(documents, queries, root, args):
    import clip
    import torch
    from PIL import Image
    model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    model.eval()
    arrays = []
    with torch.no_grad():
        for rows, is_query in ((documents, False), (queries, True)):
            text = np.zeros((len(rows), 512), dtype=np.float32)
            image = np.zeros_like(text)
            text_ids = [i for i, row in enumerate(rows) if row.get("has_text", bool(row.get("text")))]
            image_ids = [i for i, row in enumerate(rows) if row.get("image")]
            for start in range(0, len(text_ids), args.batch_size):
                ids = text_ids[start:start+args.batch_size]
                inputs = [(" ".join([rows[i].get("instruction", "").strip(), rows[i].get("text", "").strip()]).strip()
                           if is_query else rows[i].get("text", "")) for i in ids]
                tokens = clip.tokenize(inputs, truncate=True).to(args.device)
                text[ids] = model.encode_text(tokens).float().cpu().numpy()
            for start in range(0, len(image_ids), args.batch_size):
                ids = image_ids[start:start+args.batch_size]
                batch = []
                for i in ids:
                    path = Path(rows[i]["image"])
                    with Image.open(path if path.is_absolute() else root/path) as im:
                        batch.append(preprocess(im))
                image[ids] = model.encode_image(torch.stack(batch).to(args.device)).float().cpu().numpy()
            combined = np.zeros_like(text)
            text_set, image_set = set(text_ids), set(image_ids)
            for i in range(len(rows)):
                if i in text_set and i in image_set:
                    combined[i] = (text[i] / max(float(np.linalg.norm(text[i])), 1e-12)
                                   + image[i] / max(float(np.linalg.norm(image[i])), 1e-12))
                elif i in text_set:
                    combined[i] = text[i]
                elif i in image_set:
                    combined[i] = image[i]
            combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
            arrays.append(combined)
    return arrays


def add_commands(sub):
    p = sub.add_parser("prepare-bright")
    p.add_argument("--subset", choices=sorted(BRIGHT_INSTRUCTIONS), required=True)
    p.add_argument("--revision", help="Dataset revision to load")
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare_bright)
    p = sub.add_parser("prepare-mbeir")
    p.add_argument("--task", choices=sorted(MBEIR_INSTRUCTIONS), required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--query-limit", type=int, default=100, help="0 means all queries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare_mbeir)
    p = sub.add_parser("encode")
    p.add_argument("--data", required=True)
    p.add_argument("--encoder", choices=("diver", "qwen3", "clip"), required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--revision", help="Optional text embedding model revision")
    p.set_defaults(func=encode)
