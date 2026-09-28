from __future__ import annotations

import heapq
import math


def distance_events(layers, entry, distance, *, upper_layers=False):
    if entry is None:
        return
    ep = int(entry)
    if upper_layers:
        for level in range(len(layers) - 1, 0, -1):
            seen = {ep}
            d = float(distance(ep))
            yield ep, d, level
            candidates, best = [(d, ep)], (d, ep)
            while candidates:
                dc, c = heapq.heappop(candidates)
                if dc > best[0]:
                    break
                for raw in layers[level][c]:
                    node = int(raw)
                    if node in seen:
                        continue
                    seen.add(node)
                    d = float(distance(node))
                    yield node, d, level
                    if (d, node) < best:
                        best = (d, node)
                        heapq.heappush(candidates, best)
            ep = best[1]
    seen = {ep}
    d = float(distance(ep))
    yield ep, d, 0
    queue = [(d, ep)]
    while queue:
        _, current = heapq.heappop(queue)
        for raw in layers[0][current]:
            node = int(raw)
            if node in seen:
                continue
            seen.add(node)
            d = float(distance(node))
            yield node, d, 0
            heapq.heappush(queue, (d, node))


def budget_search(layers, entry, distance, budgets, *, pool_size=20,
                  upper_layers=False, excluded=()):
    budgets = sorted(set(budgets))
    if (not budgets or any(type(b) is not int or b < 1 for b in budgets)
            or type(pool_size) is not int or pool_size < 1):
        raise ValueError("Positive integer budgets and pool_size required")
    excluded = set(excluded)
    scores, layer0, result = {}, set(), {}
    calls = upper_calls = 0

    def snapshot(budget, reason):
        ranked = sorted((d, node) for node, d in scores.items() if node not in excluded)
        pool = [node for _, node in ranked[:pool_size]]
        return dict(budget=budget, pool=pool, distances=[scores[n] for n in pool],
                    distance_evals=calls, upper_distance_evals=upper_calls,
                    unique_scored_nodes=len(scores), layer0_scored_nodes=len(layer0),
                    stop_reason=reason, candidate_shortfall=max(0, pool_size-len(pool)))

    stream = distance_events(layers, entry, distance, upper_layers=upper_layers)
    for node, d, level in stream:
        if not math.isfinite(d):
            raise ValueError("Non-finite query-document distance")
        calls += 1
        upper_calls += int(level > 0)
        scores[node] = d
        if level == 0:
            layer0.add(node)
        if calls in budgets:
            result[calls] = snapshot(calls, "budget_exhausted")
        if calls == budgets[-1]:
            stream.close()
            break
    else:
        for budget in budgets:
            if budget not in result:
                result[budget] = snapshot(budget, "reachable_frontier_exhausted")
    return result


def cosine_distance(query, passages, *, hnsw=False):
    import numpy as np
    qnorm = np.linalg.norm(query) + 1e-12

    def distance(node):
        vector = passages[node]
        similarity = np.dot(query, vector) / (qnorm * (np.linalg.norm(vector) + 1e-12))
        return float(1.0 - similarity) if hnsw else 1.0 - float(similarity)
    return distance
