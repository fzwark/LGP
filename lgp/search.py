import bisect
import heapq

import numpy as np

from .computation import cosine_distance


def centroid_entry(vectors):
    center = vectors.mean(axis=0, dtype=np.float32)
    center /= np.linalg.norm(center) + 1e-12
    similarity = (vectors @ center) / (np.linalg.norm(vectors, axis=1) + 1e-12)
    return int(np.argmax(similarity))


def diskann_search(edges, entry, distance, width=20, excluded=()):
    if width < 1:
        raise ValueError("Search width must be positive")
    excluded = set(excluded)
    queue = [(distance(entry), entry)]
    discovered = {entry}
    visited = []
    calls = 1
    expanded = 0
    while queue:
        d, node = heapq.heappop(queue)
        expanded += 1
        if node not in excluded:
            bisect.insort(visited, (d, node))
        for neighbor in edges[node]:
            if neighbor in discovered:
                continue
            discovered.add(neighbor)
            heapq.heappush(queue, (distance(neighbor), neighbor))
            calls += 1
        if len(visited) >= width and (not queue or queue[0][0] > visited[width-1][0]):
            break
    return [node for _, node in visited[:width]], {"distance_evals": calls, "expanded": expanded}


def hnsw_search(state, vectors, query, width=20, excluded=()):
    from .graph import point
    from .hnsw import hnsw_graph
    graph = object.__new__(hnsw_graph)
    graph.v = [point(row) for row in vectors]
    graph.layers = state["layers"]
    graph.enter_point = state["entry"]
    graph.max_level = len(graph.layers)-1
    found, stats, _ = graph.search(point(query), k=width, ef=width)
    excluded = set(excluded)
    return [node for node in found if node not in excluded], stats
