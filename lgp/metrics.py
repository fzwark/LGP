import math


def evaluate(queries, rankings, cutoffs=(5, 10)):
    totals = {f"{metric}@{k}": 0.0 for k in cutoffs for metric in ("NDCG", "Recall")}
    count = 0
    for query in queries:
        relevant = set(query.get("relevant_ids", []))
        if not relevant:
            continue
        if query["id"] not in rankings:
            raise ValueError("Missing query ranking; refusing incomplete evaluation")
        ranked = rankings[query["id"]]
        if len(set(ranked)) != len(ranked):
            raise ValueError("Duplicate documents in ranking")
        count += 1
        for k in cutoffs:
            if k < 1:
                raise ValueError("Cutoffs must be positive")
            top = ranked[:k]
            dcg = sum(1.0 / math.log2(rank + 2) for rank, doc in enumerate(top) if doc in relevant)
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
            totals[f"NDCG@{k}"] += dcg / ideal
            totals[f"Recall@{k}"] += len(set(top) & relevant) / len(relevant)
    return {"evaluated_queries": count, "unlabeled_queries": len(queries)-count,
            **{key: 100 * value / count if count else 0.0 for key, value in totals.items()}}
