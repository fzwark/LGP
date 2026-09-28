from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import asdict, dataclass
import copy
import hashlib
import heapq
import math
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time

from tqdm import tqdm
from .graph import diskann_graph, point, dist

FORMAT = "lgp_hnsw_v2"
ADAPTATION = "layer0_exact_outdegree_coverage_gate_v2"
_ENV_LOCK = threading.RLock()


@dataclass(frozen=True)
class ShortcutConfig:
    B: int = 2
    cand_cap: int = 24
    min_2hop_gain: int = 12
    require_uncovered: bool = False
    max_calls: int = 500000
    anchor_chars: int = 1200
    keep_chars: int = 700
    cand_chars: int = 900
    backbone: bool = True
    reverse_policy: str = "none"
    budget_policy: str = "fixed"

    def __post_init__(self):
        if self.B < 0 or self.cand_cap < 1 or self.B > self.cand_cap:
            raise ValueError("Require 0 <= B <= cand_cap and cand_cap >= 1")
        if self.min_2hop_gain < 0 or self.max_calls < 0:
            raise ValueError("Gain and call budget must be nonnegative")
        if min(self.anchor_chars, self.keep_chars, self.cand_chars) < 1:
            raise ValueError("Text budgets must be positive")
        if self.reverse_policy not in {"none", "bounded_hnsw"}:
            raise ValueError("Unknown HNSW reverse policy")
        if self.budget_policy not in {"fixed", "legacy_cap"}:
            raise ValueError("Unknown HNSW proposal budget policy")

    def environment(self):
        return {
            "LLM_INDEX_SELECTOR": "llm_control",
            "LLM_INDEX_SHORTCUT_B": str(self.B),
            "LLM_INDEX_SHORTCUT_CAND_CAP": str(self.cand_cap),
            "LLM_INDEX_MIN_2HOP_GAIN": str(self.min_2hop_gain),
            "LLM_INDEX_REQUIRE_UNCOVERED": str(int(self.require_uncovered)),
            "LLM_INDEX_REPLACE_EDGES": "1",
            "LLM_INDEX_REPLACE_POLICY": "legacy",
            "LLM_INDEX_PROTECT_LOCAL_EDGES": "0",
            "LLM_INDEX_REPLACE_MIN_GAIN_DELTA": "0",
            "LLM_INDEX_TRUNCATE_TEXT": "1",
            "LLM_INDEX_ANCHOR_CHARS": str(self.anchor_chars),
            "LLM_INDEX_KEEP_CHARS": str(self.keep_chars),
            "LLM_INDEX_CAND_CHARS": str(self.cand_chars),
        }


@contextmanager
def shortcut_environment(config):
    with _ENV_LOCK:
        values = config.environment()
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class BuildInterrupted(RuntimeError):
    pass


class LegacyHNSWCacheError(ValueError):
    pass


def atomic_pickle_dump(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".hnsw-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class hnsw_graph(diskann_graph):
    def __init__(self, A, L=200, R=16, alpha=1.2, llm=None, texts=None,
                 fla=0, seed=42, config=None, **kwargs):
        if not A or R < 2 or L < R:
            raise ValueError("Require nonempty embeddings, M >= 2, efConstruction >= M")
        if fla not in (0, 2):
            raise ValueError("Use variant 0 (Vanilla) or 2 (LGP)")
        if texts is not None and len(texts) != len(A):
            raise ValueError("Text and embedding row counts differ")
        super().__init__(A, L, R, alpha, llm=llm, texts=texts, fla=fla, seed=seed, **kwargs)
        self.M = int(R)
        self.maxM, self.maxM0 = self.M, 2 * self.M
        self.ef_construction = int(L)
        self.ml = 1.0 / math.log(self.M)
        self.layers = []
        self.level_of = [-1] * self.n
        self.enter_point, self.max_level = None, -1
        self.config = config or ShortcutConfig()
        self.llm_max_calls_per_prune = self.config.max_calls
        self.progress = {"phase": "new", "next_i": 0}
        self.build_order = self.refine_order = None
        self.baseline_degrees = self.baseline_digest = self.baseline_upper_digest = None
        self.protected_backbone = None
        self.baseline_reachable = None
        self._require_shortcut_replacement = True
        self.adaptation_stats = {
            "budget_reduced_nodes": 0, "reverse_attempted": 0,
            "reverse_already_present": 0, "reverse_rejected": 0,
            "reverse_committed": 0,
        }

    def _sync_layer0_alias(self):
        self.edge = self.layers[0] if self.layers else [[] for _ in range(self.n)]
        self.start = self.enter_point

    def _ensure_layers(self, level):
        while len(self.layers) <= level:
            self.layers.append([[] for _ in range(self.n)])

    def _sample_level(self):
        return int(-math.log(max(1e-12, self._rng.random())) * self.ml)

    def _neighbor_budget(self, level):
        return self.maxM0 if level == 0 else self.maxM

    def _search_layer(self, q, ep_ids, ef, lc, stats=None):
        if ef < 1:
            raise ValueError("ef must be positive")
        if isinstance(ep_ids, int):
            ep_ids = [ep_ids]
        candidates, best, seen, expanded, distances = [], [], set(), [], {}

        def distance(node):
            if node not in distances:
                distances[node] = dist(self.v[node], q)
                if stats is not None:
                    stats["distance_evals"] += 1
            return distances[node]

        for ep in dict.fromkeys(ep_ids):
            if ep is None:
                continue
            d = distance(ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(best, (-d, -ep))
            seen.add(ep)
            if len(best) > ef:
                heapq.heappop(best)
        while candidates:
            dc, c = heapq.heappop(candidates)
            if len(best) >= ef and dc > -best[0][0]:
                break
            expanded.append(c)
            for e in self.layers[lc][c]:
                if e in seen:
                    continue
                seen.add(e)
                de = distance(e)
                if len(best) < ef or (de, e) < (-best[0][0], -best[0][1]):
                    heapq.heappush(candidates, (de, e))
                    heapq.heappush(best, (-de, -e))
                    if len(best) > ef:
                        heapq.heappop(best)
        result = sorted((-nid for _, nid in best), key=lambda x: (distances[x], x))
        return result, expanded

    def _select_neighbors_hnsw(self, base_id, cand_ids, M, lc=0):
        candidates = sorted(set(cand_ids) - {base_id},
                            key=lambda x: (dist(self.v[base_id], self.v[x]), x))
        if len(candidates) < M:
            return candidates
        selected = []
        for candidate in candidates:
            if len(selected) >= M:
                break
            d_base = dist(self.v[base_id], self.v[candidate])
            if all(dist(self.v[s], self.v[candidate]) >= d_base for s in selected):
                selected.append(candidate)
        return selected

    def _connect_new_element(self, qid, neighbors, lc):
        self.layers[lc][qid] = list(neighbors)
        budget = self._neighbor_budget(lc)
        for e in neighbors:
            current = list(self.layers[lc][e])
            if qid not in current:
                current.append(qid)
            if len(current) > budget:
                current = self._select_neighbors_hnsw(e, current, budget, lc)
            self.layers[lc][e] = current

    def _insert_one(self, qid):
        level = self._sample_level()
        self.level_of[qid] = level
        self._ensure_layers(level)
        if self.enter_point is None:
            self.enter_point, self.max_level = qid, level
            self._sync_layer0_alias()
            return
        ep, q = self.enter_point, self.v[qid]
        for lc in range(self.max_level, level, -1):
            found, _ = self._search_layer(q, ep, 1, lc)
            ep = found[0]
        for lc in range(min(self.max_level, level), -1, -1):
            found, _ = self._search_layer(q, ep, self.ef_construction, lc)
            neighbors = self._select_neighbors_hnsw(qid, found, self.M, lc)
            self._connect_new_element(qid, neighbors, lc)
            ep = found
        if level > self.max_level:
            self.enter_point, self.max_level = qid, level
        self._sync_layer0_alias()

    def search(self, q, k=10, ef=100):
        if k < 1 or ef < k:
            raise ValueError("Require 1 <= k <= efSearch; efSearch is not rerank quota")
        stats = {"distance_evals": 0, "upper_expanded": 0, "layer0_expanded": 0}
        if self.enter_point is None:
            return [], stats, []
        ep = self.enter_point
        for lc in range(self.max_level, 0, -1):
            found, expanded = self._search_layer(q, ep, 1, lc, stats)
            stats["upper_expanded"] += len(expanded)
            ep = found[0]
        found, expanded = self._search_layer(q, ep, ef, 0, stats)
        stats["layer0_expanded"] = len(expanded)
        return found[:k], stats, expanded

    def greedysearch(self, s, q, k, L):
        found, _, visited = self.search(q, k=k, ef=max(k, L))
        return found, visited

    def greedysearch_layer0(self, s, q, k, L):
        if s is None:
            return [], []
        found, expanded = self._search_layer(q, s, max(k, L), 0)
        return found[:k], expanded

    def structural_digest(self, upper_only=False):
        layers = self.layers[1:] if upper_only else self.layers
        payload = (layers, self.level_of, self.enter_point, self.max_level)
        return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()

    def reachability_backbone(self):
        protected = [set() for _ in range(self.n)]
        if self.enter_point is None:
            return protected, 0
        reached = {self.enter_point}
        queue = deque([self.enter_point])
        while queue:
            parent = queue.popleft()
            for child in self.edge[parent]:
                if child not in reached:
                    reached.add(child)
                    queue.append(child)
                    protected[parent].add(child)
        return protected, len(reached)

    def _pick_low_value_edge_to_replace(self, p, S, protected=None):
        protected = set(protected or ())
        if self.protected_backbone is not None:
            protected.update(self.protected_backbone[p])
        return super()._pick_low_value_edge_to_replace(p, S, protected=protected)

    def clone_for_refinement(self, llm, config=None):
        if self.progress["phase"] != "vanilla_done":
            raise ValueError("Method must start from a completed vanilla HNSW")
        g = hnsw_graph(self.v, self.L, self.M, self.alpha, llm=llm,
                       texts=self.texts, fla=2, seed=self.seed, config=config or self.config)
        g.layers = copy.deepcopy(self.layers)
        g.level_of = self.level_of.copy()
        g.enter_point, g.max_level = self.enter_point, self.max_level
        g._sync_layer0_alias()
        g._rng.setstate(self._rng.getstate())
        g.build_order = self.build_order.copy()
        g.baseline_degrees = list(map(len, self.edge))
        g.baseline_digest = self.structural_digest()
        g.baseline_upper_digest = self.structural_digest(upper_only=True)
        g.protected_backbone, g.baseline_reachable = self.reachability_backbone()
        if not g.config.backbone:
            g.protected_backbone = [set() for _ in range(g.n)]
        g.refine_order = list(range(self.n))
        g._rng.shuffle(g.refine_order)
        g.progress = {"phase": "refine", "next_i": 0}
        return g

    def _run_phase(self, phase, order, operation, checkpoint, every_nodes,
                   every_seconds, should_stop, show_progress):
        last_save, since_save = time.monotonic(), 0
        start = self.progress["next_i"]
        if every_nodes < 1 or every_seconds < 0:
            raise ValueError("Invalid checkpoint interval")
        for i in tqdm(range(start, self.n), initial=start, total=self.n,
                      desc=f"HNSW {phase}", disable=not show_progress):
            if should_stop and should_stop():
                if checkpoint is not None:
                    checkpoint(self)
                raise BuildInterrupted("Interrupted at a checkpoint-safe node boundary")
            try:
                operation(order[i])
            except BuildInterrupted:
                if checkpoint is not None:
                    checkpoint(self)
                raise
            self.progress = {"phase": phase, "next_i": i + 1}
            since_save += 1
            elapsed = time.monotonic() - last_save
            if checkpoint is not None and (since_save >= every_nodes or
                    (every_seconds > 0 and elapsed >= every_seconds)):
                checkpoint(self)
                last_save, since_save = time.monotonic(), 0
        self.progress = {"phase": "vanilla_done" if phase == "build" else "method_done",
                         "next_i": self.n}
        self.validate()
        if checkpoint is not None:
            checkpoint(self)

    def build_vanilla(self, checkpoint=None, every_nodes=500, every_seconds=300,
                      should_stop=None, show_progress=True):
        if self.progress["phase"] == "vanilla_done":
            return
        if self.progress["phase"] == "new":
            self.build_order = list(range(self.n))
            self._rng.shuffle(self.build_order)
            self.progress = {"phase": "build", "next_i": 0}
        if self.progress["phase"] != "build":
            raise ValueError("Cannot rebuild a method graph as vanilla")
        self._run_phase("build", self.build_order, self._insert_one, checkpoint,
                        every_nodes, every_seconds, should_stop, show_progress)

    def _apply_reverse_updates(self, node, added, undo):
        if self.config.reverse_policy == "none":
            return
        for destination in sorted(added):
            self.adaptation_stats["reverse_attempted"] += 1
            row = self.edge[destination]
            if node in row:
                self.adaptation_stats["reverse_already_present"] += 1
                continue
            locked = self.protected_backbone[destination]
            selected = self._select_neighbors_hnsw(destination, row + [node], len(row))
            removable = set(row) - set(selected) - locked
            if not row or node not in selected or not removable:
                self.adaptation_stats["reverse_rejected"] += 1
                continue
            drop = max(removable, key=lambda x: (dist(self.v[destination], self.v[x]), x))
            undo.setdefault(destination, row.copy())
            self.edge[destination] = [x for x in row if x != drop] + [node]
            self.adaptation_stats["reverse_committed"] += 1

    def _refine_one(self, node):
        if (self.config.B == 0 or self.llm is None or
                self.debug_stats["num_use_shortcut_calls"] >= self.config.max_calls):
            return
        before = self.edge[node].copy()
        if not before:
            return
        locked = self.protected_backbone[node] if self.protected_backbone is not None else set()
        replaceable = len(set(before) - locked)
        effective_B = (min(self.config.B, max(0, replaceable - 1))
                       if self.config.budget_policy == "legacy_cap" else self.config.B)
        if effective_B == 0:
            return
        _, visited = self.greedysearch(self.start, self.v[node], 1, self.L)
        original_R = self.R
        stats_before = self.debug_stats.copy()
        logs_before = len(self.edit_logs)
        cache_size_before = len(self._llm_judge_cache)
        adaptation_before = self.adaptation_stats.copy()
        undo = {node: before}
        try:
            if effective_B < self.config.B:
                self.adaptation_stats["budget_reduced_nodes"] += 1
            self.R = len(before)
            super().refine_node_use_shortcuts(
                node, visited, B=effective_B, cand_cap=self.config.cand_cap)
            if getattr(self, "_stop_requested", None) and self._stop_requested():
                raise BuildInterrupted("Interrupted refinement node rolled back for replay")
            after = self.edge[node]
            if len(after) != len(before) or len(set(after)) != len(after) or node in after:
                raise RuntimeError("Shared selector violated exact-outdegree replacement")
            self._apply_reverse_updates(node, set(after) - set(before), undo)
            for changed, original in undo.items():
                row = self.edge[changed]
                if (len(row) != len(original) or len(set(row)) != len(row) or changed in row
                        or not self.protected_backbone[changed].issubset(set(row))):
                    raise RuntimeError("HNSW reverse update violated structural invariants")
            if getattr(self, "_stop_requested", None) and self._stop_requested():
                raise BuildInterrupted("Interrupted reverse updates rolled back for replay")
        except BaseException:
            for changed, original in undo.items():
                self.edge[changed] = original
            self.debug_stats = stats_before
            self.adaptation_stats = adaptation_before
            del self.edit_logs[logs_before:]
            while len(self._llm_judge_cache) > cache_size_before:
                self._llm_judge_cache.popitem()
            raise
        finally:
            self.R = original_R

    def refine_layer0(self, checkpoint=None, every_nodes=500, every_seconds=300,
                      should_stop=None, show_progress=True):
        if self.progress["phase"] == "method_done":
            return
        if self.progress["phase"] != "refine":
            raise ValueError("Use clone_for_refinement() on the completed baseline first")
        if (self.llm is None and self.config.B > 0 and
                self.debug_stats["num_use_shortcut_calls"] < self.config.max_calls):
            raise ValueError("llm_control refinement requires an LLM client")
        self._stop_requested = should_stop
        try:
            with shortcut_environment(self.config):
                self._run_phase("refine", self.refine_order, self._refine_one, checkpoint,
                                every_nodes, every_seconds, should_stop, show_progress)
        finally:
            self._stop_requested = None

    def indexing(self, **kwargs):
        if self.fla != 0:
            raise ValueError("Build vanilla once, then clone_for_refinement().refine_layer0()")
        return self.build_vanilla(**kwargs)

    def validate(self):
        if self.max_level != max(self.level_of) or len(self.layers) != self.max_level + 1:
            raise ValueError("Inconsistent HNSW layer/level metadata")
        if self.max_level >= 0:
            if self.enter_point is None or self.level_of[self.enter_point] != self.max_level:
                raise ValueError("Entry point must belong to the highest layer")
        if self.layers and self.edge is not self.layers[0]:
            raise ValueError("Layer-zero alias is detached")
        for level, rows in enumerate(self.layers):
            if len(rows) != self.n:
                raise ValueError("Wrong layer row count")
            for node, row in enumerate(rows):
                if len(row) > self._neighbor_budget(level) or len(row) != len(set(row)):
                    raise ValueError("Degree overflow or duplicate edge")
                if row and self.level_of[node] < level:
                    raise ValueError("Edges on an absent node level")
                if any(not isinstance(x, int) or x < 0 or x >= self.n or
                       x == node or self.level_of[x] < level for x in row):
                    raise ValueError("Invalid neighbor ID or level")
        phase, position = self.progress["phase"], self.progress["next_i"]
        if phase not in {"new", "build", "vanilla_done", "refine", "method_done"} or not 0 <= position <= self.n:
            raise ValueError("Invalid build progress")
        if phase in {"vanilla_done", "method_done"} and position != self.n:
            raise ValueError("Completed graph has an incomplete build position")
        if phase != "new":
            if self.build_order is None or sorted(self.build_order) != list(range(self.n)):
                raise ValueError("Invalid insertion order")
            count = position if phase == "build" else self.n
            inserted = set(self.build_order[:count])
            if any((level >= 0) != (i in inserted) for i, level in enumerate(self.level_of)):
                raise ValueError("Insertion position disagrees with stored levels")
        if phase in {"refine", "method_done"}:
            if self.refine_order is None or sorted(self.refine_order) != list(range(self.n)):
                raise ValueError("Invalid refinement order")
            if list(map(len, self.edge)) != self.baseline_degrees:
                raise ValueError("Method changed original node out-degrees")
            if self.structural_digest(upper_only=True) != self.baseline_upper_digest:
                raise ValueError("Method changed HNSW upper layers or entry metadata")
            if self.protected_backbone is None or len(self.protected_backbone) != self.n:
                raise ValueError("Missing reachability protection state")
            if any(not locked.issubset(set(row)) for locked, row in zip(self.protected_backbone, self.edge)):
                raise ValueError("Method removed an original reachability-tree edge")
            expected_locked = self.baseline_reachable - 1 if self.config.backbone else 0
            if sum(map(len, self.protected_backbone)) != expected_locked:
                raise ValueError("Inconsistent reachability-tree metadata")

    def save(self, path, metadata=None):
        state = {"format": FORMAT, "adaptation": ADAPTATION, "metadata": metadata or {},
                 "n": self.n, "dim": self.v[0].dim, "L": self.L, "M": self.M,
                 "alpha": self.alpha, "seed": self.seed, "flag": self.fla,
                 "config": asdict(self.config), "layers": self.layers,
                 "level_of": self.level_of, "enter_point": self.enter_point,
                 "max_level": self.max_level, "progress": self.progress,
                 "build_order": self.build_order, "refine_order": self.refine_order,
                 "rng_state": self._rng.getstate(), "baseline_degrees": self.baseline_degrees,
                 "baseline_digest": self.baseline_digest,
                 "baseline_upper_digest": self.baseline_upper_digest,
                 "protected_backbone": self.protected_backbone,
                 "baseline_reachable": self.baseline_reachable,
                 "adaptation_stats": self.adaptation_stats,
                 "debug_stats": self.debug_stats, "edit_logs": self.edit_logs,
                 "llm_judge_cache": self._llm_judge_cache}
        atomic_pickle_dump(state, path)

    @classmethod
    def load(cls, path, A, texts=None, llm=None, expected_metadata=None):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or state.get("format") != FORMAT:
            raise LegacyHNSWCacheError("Unsupported HNSW cache format; rebuild the index")
        if state.get("adaptation") != ADAPTATION:
            raise ValueError("Wrong HNSW adaptation version")
        if expected_metadata is not None and state["metadata"] != expected_metadata:
            raise ValueError("Cache dataset/model/config/source fingerprint mismatch")
        if len(A) != state["n"] or any(x.dim != state["dim"] for x in A):
            raise ValueError("Embedding shape does not match HNSW cache")
        g = cls(A, state["L"], state["M"], state["alpha"], llm=llm, texts=texts,
                fla=state["flag"], seed=state["seed"], config=ShortcutConfig(**state["config"]))
        for name in ("layers", "level_of", "enter_point", "max_level", "progress",
                     "build_order", "refine_order", "baseline_degrees", "baseline_digest",
                     "baseline_upper_digest", "protected_backbone", "baseline_reachable",
                     "debug_stats", "edit_logs", "adaptation_stats"):
            setattr(g, name, state[name])
        g._rng.setstate(state["rng_state"])
        g._llm_judge_cache = state["llm_judge_cache"]
        g._sync_layer0_alias()
        g.validate()
        return g
