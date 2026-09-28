from __future__ import annotations
import bisect
from collections import Counter
import heapq
import math
import functools
import os
import random
import re
import numpy as np
from tqdm import tqdm
from .candidate_sources import compose_candidate_shortlist, normalize_candidate_source

class point:

    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float32)
        self.dim = self.x.shape[0]
        self._cosine_norm = np.linalg.norm(self.x) + 1e-12

    def __getstate__(self):
        return {k: v for (k, v) in self.__dict__.items() if k != '_cosine_norm'}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.pop('_cosine_norm', None)

def dist(A: point, B: point, metric='cosine'):
    if metric == 'l2':
        return float(np.linalg.norm(A.x - B.x))
    elif metric == 'cosine':
        a = A.x
        b = B.x
        norm_a = getattr(A, '_cosine_norm', None)
        norm_b = getattr(B, '_cosine_norm', None)
        if norm_a is None:
            norm_a = A._cosine_norm = np.linalg.norm(a) + 1e-12
        if norm_b is None:
            norm_b = B._cosine_norm = np.linalg.norm(b) + 1e-12
        return float(1.0 - a @ b / (norm_a * norm_b))
    else:
        raise ValueError(metric)

class diskann_graph:

    def __init__(self, A, L=200, R=16, alpha=1.2, llm=None, texts=None, fla=0, llm_anchor_chars=1200, llm_keep_chars=700, llm_cand_chars=900, llm_cache=True, llm_max_calls_per_prune=500000, seed=42):
        if not A or fla not in (0, 2) or R < 2 or (L < R):
            raise ValueError('Require nonempty vectors, variant 0/2, R >= 2, and L >= R')
        (self.n, self.v, self.fla) = (len(A), A, fla)
        self.edge = [[] for _ in A]
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        (self.start, self.now, self.perm) = (None, None, None)
        (self.L, self.R, self.alpha) = (L, R, alpha)
        self._qflag = 0
        self._in_Q = np.zeros(self.n, dtype=np.int32)
        (self.llm, self.texts) = (llm, texts)
        self.llm_anchor_chars = int(llm_anchor_chars)
        self.llm_keep_chars = int(llm_keep_chars)
        self.llm_cand_chars = int(llm_cand_chars)
        self.llm_cache = bool(llm_cache)
        self.llm_max_calls_per_prune = int(llm_max_calls_per_prune)
        self._llm_judge_cache = {}
        self.debug_stats = Counter()
        self.edit_logs = []

    def indexing(self):
        if self.start is None:
            raise ValueError('Set the entry node before construction')
        self.random_init()
        for iteration in range(2):
            for node in tqdm(self.perm[iteration], desc=f'Build pass {iteration + 1}'):
                (_, visited) = self.greedysearch(self.start, self.v[node], 1, self.L)
                self.robustprune(node, visited, self.alpha, self.R)
                protected = None
                if iteration == 1 and self.fla == 2:
                    protected = self.refine_node_use_shortcuts(node, visited, B=int(os.getenv('LLM_INDEX_SHORTCUT_B', '2')), cand_cap=int(os.getenv('LLM_INDEX_SHORTCUT_CAND_CAP', '24')))
                self.interinsert(node, protected_neighbors=protected)

    def cmp(self, x, y):
        if dist(self.now, self.v[x]) < dist(self.now, self.v[y]):
            return -1
        elif dist(self.now, self.v[y]) < dist(self.now, self.v[x]):
            return 1
        elif x < y:
            return -1
        else:
            return 1

    def _txt(self, idx: int, max_chars: int) -> str:
        if self.texts is None:
            return ''
        if idx < 0 or idx >= len(self.texts):
            return ''
        t = self.texts[idx]
        if t is None:
            return ''
        t = str(t).replace('\n', ' ').strip()
        if os.getenv('LLM_INDEX_TRUNCATE_TEXT', '1').lower() in {'0', 'false', 'no'}:
            return t
        return t[:max_chars]

    def _ordered_prune_candidates(self, p, V):
        self.now = self.v[p]
        U = list(self.edge[p])
        seen = set(U)
        for i in V:
            if i not in seen:
                U.append(i)
                seen.add(i)
        anchor_dist = {i: dist(self.v[p], self.v[i]) for i in U}
        if all((math.isfinite(d) for d in anchor_dist.values())):
            U.sort(key=lambda i: (anchor_dist[i], i))
        else:
            U.sort(key=functools.cmp_to_key(self.cmp))
        if p in U:
            U.remove(p)
        return (U, anchor_dist)

    def robustprune(self, p, V, alpha, R):
        (U, anchor_dist) = self._ordered_prune_candidates(p, V)
        vis = [True] * len(U)
        self.edge[p] = []
        for i in range(len(U)):
            if vis[i] == False:
                continue
            x = U[i]
            self.edge[p].append(x)
            for j in range(i + 1, len(U)):
                y = U[j]
                if dist(self.v[x], self.v[y]) * alpha <= anchor_dist[y]:
                    vis[j] = False
            if len(self.edge[p]) == R:
                break

    def robustprune_with_protection(self, p, V, alpha, R, protected=None):
        if protected is None:
            protected = set()
        (U, anchor_dist) = self._ordered_prune_candidates(p, V)
        vis = [True] * len(U)
        self.edge[p] = []
        for i in range(len(U)):
            x = U[i]
            if x not in protected:
                continue
            if vis[i] == False:
                continue
            self.edge[p].append(x)
            for j in range(i + 1, len(U)):
                y = U[j]
                if dist(self.v[x], self.v[y]) * alpha <= anchor_dist[y]:
                    vis[j] = False
            vis[i] = False
            if len(self.edge[p]) == R:
                return
        for i in range(len(U)):
            if vis[i] == False:
                continue
            x = U[i]
            self.edge[p].append(x)
            for j in range(i + 1, len(U)):
                y = U[j]
                if dist(self.v[x], self.v[y]) * alpha <= anchor_dist[y]:
                    vis[j] = False
            if len(self.edge[p]) == R:
                break

    def greedysearch(self, s, q, k, L, edge_traffic=None, node_traffic=None):
        self._qflag += 1
        flag = self._qflag
        in_Q = self._in_Q
        self.now = q
        Q = [(dist(self.v[s], q), s)]
        vis = []
        ret_vis = []
        p = s
        in_Q[s] = flag
        while p != -1:
            if node_traffic is not None:
                node_traffic[p] += 1
            ret_vis.append(p)
            bisect.insort(vis, heapq.heappop(Q))
            for x in self.edge[p]:
                if in_Q[x] != flag:
                    if edge_traffic is not None:
                        outgoing = edge_traffic[p]
                        outgoing[x] = outgoing.get(x, 0) + 1
                    heapq.heappush(Q, (dist(self.v[x], q), x))
                    in_Q[x] = flag
            p = -1
            if len(Q) > 0:
                if bisect.bisect(vis, Q[0]) < L:
                    p = Q[0][1]
        ret_Q = []
        for i in range(k):
            ret_Q.append(vis[i][1])
        return (ret_Q, ret_vis)

    def random_init(self):
        self.edge = [[] for i in range(self.n)]
        deg = [0 for i in range(self.n)]
        rem_id = list(range(self.n))
        while len(rem_id) > 0:
            p = rem_id[0]
            rem_deg = self.R - deg[p]
            rem_id.remove(p)
            if rem_deg <= len(rem_id):
                neighbors = self._rng.sample(rem_id, rem_deg)
            else:
                neighbors = rem_id.copy()
            self.edge[p] = self.edge[p] + neighbors
            for v in neighbors:
                deg[v] += 1
                self.edge[v].append(p)
                if deg[v] == self.R:
                    rem_id.remove(v)
        self.perm = []
        for iter in range(2):
            perm = list(range(self.n))
            self._rng.shuffle(perm)
            self.perm.append(perm)

    def _alpha_covered(self, p: int, selected: list[int], cand: int) -> bool:
        for x in selected:
            if dist(self.v[x], self.v[cand]) * self.alpha <= dist(self.v[p], self.v[cand]):
                return True
        return False

    def _min_cosdist_to_selected(self, selected_ids, cand_id):
        if not selected_ids:
            return 1000000000.0
        return min((dist(self.v[sid], self.v[cand_id], metric='cosine') for sid in selected_ids))

    def _llm_select_use_shortcuts(self, anchor_id: int, selected_ids: list[int], cand_ids: list[int], keep: int=2, selected_cap: int=4, cand_cap: int=16, generic_prompt: bool=False, general_question_prompt: bool=False, refined_general_prompt: bool=False, structural_general_prompt: bool=False, query_overlap_prompt: bool=False, query_route_prompt: bool=False) -> list[int]:
        if keep <= 0 or not cand_ids:
            return []
        if self.llm is None or self.texts is None:
            return []
        selected_ids = list(dict.fromkeys([x for x in selected_ids if x != anchor_id]))
        selected_ids.sort(key=lambda sid: dist(self.v[anchor_id], self.v[sid]))
        selected_ids = selected_ids[:selected_cap]
        cand_ids = list(dict.fromkeys([x for x in cand_ids if x != anchor_id and x not in selected_ids]))
        cand_ids.sort(key=lambda cid: (-self._candidate_coverage_gain(anchor_id, selected_ids, cid), dist(self.v[anchor_id], self.v[cid])))
        cand_ids = cand_ids[:cand_cap]
        if not cand_ids:
            return []
        prompt_version = 'theorem_v1'
        key = ('query_sim_shortcut', prompt_version, anchor_id, tuple(selected_ids), tuple(cand_ids), keep)
        if self.llm_cache and key in self._llm_judge_cache:
            return self._llm_judge_cache[key]
        default_anchor_chars = '1200'
        default_keep_chars = '700'
        default_cand_chars = '900'
        anchor_chars = int(os.getenv('LLM_INDEX_ANCHOR_CHARS', default_anchor_chars))
        keep_chars = int(os.getenv('LLM_INDEX_KEEP_CHARS', default_keep_chars))
        cand_chars = int(os.getenv('LLM_INDEX_CAND_CHARS', default_cand_chars))
        anchor = self._txt(anchor_id, anchor_chars)
        keep_lines = []
        for (i, sid) in enumerate(selected_ids):
            keep_lines.append(f'(K{i + 1}) {self._txt(sid, keep_chars)}')
        keep_block = '\n'.join(keep_lines) if keep_lines else '(none)'
        cand_lines = []
        for (i, cid) in enumerate(cand_ids):
            cand_lines.append(f'[{i + 1}] {self._txt(cid, cand_chars)}')
        cand_block = '\n\n'.join(cand_lines)
        sys = f"You are editing a theorem retrieval graph at indexing time.\nThere is no real user query available, and you must not assume a particular test query.\nInstead, privately simulate plausible latent user queries / problem intents for which the ANCHOR might be retrieved as a nearby theorem.\n\nYour goal is to improve query-time routing from the ANCHOR to useful answers.\nFirst, if helpful, optionally write a short latent query line beginning with QUERY: ; then select at most the best candidates for shortcut edges.\n\nFor each CANDIDATE, ask: would a user query that lands near the ANCHOR also reasonably need this CANDIDATE as the target theorem, a prerequisite theorem, a bridge theorem, a generalization/special case, or a theorem used in the same proof/problem-solving route?\n\nSelect a CANDIDATE only if adding an edge ANCHOR -> CANDIDATE would help query-time graph search route from the ANCHOR's local neighborhood toward useful answers that are not already covered by the ALREADY-SELECTED LOCAL NEIGHBORS.\n\nDo NOT select based on wording similarity alone.\nDo NOT select broad but unrelated facts.\nIf no candidate has a clear latent-query/use-case connection, return NONE.\nReturn at most {keep} candidate numbers, best first, or NONE. Examples: [3] [1]  or  NONE"
        user = f'ANCHOR:\n{anchor}\n\nALREADY-SELECTED LOCAL NEIGHBORS:\n{keep_block}\n\nCANDIDATE SHORTCUTS:\n{cand_block}\n\nPrivately imagine the latent queries/use-cases, then choose which candidates should be connected to the ANCHOR for query-time routing. Return only candidate numbers or NONE.\nIf you include a latent query, start it with QUERY: on its own line.'
        try:
            out = self.llm.run_one_message(sys=sys, input=user)
        except Exception as exc:
            if os.getenv('LLM_INDEX_FAIL_ON_LLM_ERROR', '0').lower() in {'1', 'true', 'yes'}:
                raise RuntimeError('LLM shortcut selection request failed') from exc
            return []
        out = str(out or '')
        if '</think>' in out:
            out = out.split('</think>')[-1]
        if '</reasoning>' in out:
            out = out.split('</reasoning>')[-1]
        query_match = re.search('(?mi)^QUERY:\\s*(.+)$', out)
        nums = []
        matches = re.findall('\\[(\\d+)\\]', out)
        if not matches and 'NONE' not in out.upper():
            matches = re.findall('(?<!\\d)(\\d+)(?!\\d)', out)
        for m in matches:
            j = int(m) - 1
            if 0 <= j < len(cand_ids) and cand_ids[j] not in nums:
                nums.append(cand_ids[j])
            if len(nums) >= keep:
                break
        if self.llm_cache:
            self._llm_judge_cache[key] = nums
        return nums

    def refine_node_use_shortcuts(self, p: int, vis: list[int], B: int=2, cand_cap: int=24) -> set[int]:
        S = list(dict.fromkeys([x for x in self.edge[p] if x != p]))
        if not S:
            return set()
        if self.llm is None or self.texts is None:
            return set()
        if self.debug_stats['num_use_shortcut_calls'] >= self.llm_max_calls_per_prune:
            return set()
        min_2hop_gain = int(os.getenv('LLM_INDEX_MIN_2HOP_GAIN', '12'))
        require_uncovered = os.getenv('LLM_INDEX_REQUIRE_UNCOVERED', '0').lower() not in {'0', 'false', 'no'}
        protected_local = set()
        cand_pool = set((x for x in vis if x != p and x not in S))
        for x in S:
            for y in self.edge[x]:
                if y != p and y not in S:
                    cand_pool.add(y)
        cand_pool = list(cand_pool)
        if not cand_pool:
            return set()
        (metrics, alpha_covered) = self._shortlist_metrics(p, S, cand_pool)
        near_rank = [c for c in sorted(cand_pool, key=lambda x: metrics[x]['cand_anchor_dist']) if not alpha_covered[c]]
        gain_rank = sorted(cand_pool, key=lambda x: (-metrics[x]['cand_unique_2hop_gain'], metrics[x]['cand_anchor_dist']))
        diverse_rank = sorted(cand_pool, key=lambda x: (-metrics[x]['cand_separation'], -metrics[x]['cand_unique_2hop_gain'], metrics[x]['cand_anchor_dist']))
        candidate_source = normalize_candidate_source(os.getenv('LLM_INDEX_CANDIDATE_SOURCE', 'balanced'))
        geo_cands = compose_candidate_shortlist(near_rank, gain_rank, diverse_rank, cand_cap, candidate_source)
        if not geo_cands:
            return set()
        prompt_cands = geo_cands
        self.debug_stats['num_use_shortcut_calls'] += 1
        chosen = self._llm_select_use_shortcuts(anchor_id=p, selected_ids=S, cand_ids=prompt_cands, keep=B, selected_cap=int(os.getenv('LLM_INDEX_SELECTED_CONTEXT_CAP', '4')), cand_cap=cand_cap, generic_prompt=False, general_question_prompt=False, refined_general_prompt=False, structural_general_prompt=False, query_overlap_prompt=False, query_route_prompt=False)
        if not chosen:
            return set()
        old = set(self.edge[p])
        cur = list(S)
        removed = []
        accepted = []
        rejected = []
        for c in chosen:
            gain = self._candidate_coverage_gain(p, S, c)
            covered = self._alpha_covered(p, S, c)
            ok_uncovered = not require_uncovered or not covered
            ok_gain = gain >= min_2hop_gain
            if not ok_uncovered:
                self.debug_stats['num_gate_fail_covered'] += 1
            if not ok_gain:
                self.debug_stats['num_gate_fail_2hop_gain'] += 1
            if ok_uncovered and ok_gain:
                accepted.append(c)
            else:
                rejected.append({'id': c, 'covered': bool(covered),
                                 'unique_2hop_gain': gain,
                                 'required_2hop_gain': min_2hop_gain})
        self.debug_stats['num_shortcut_proposed'] += len(chosen)
        self.debug_stats['num_shortcut_accepted'] += len(accepted)
        self.debug_stats['num_shortcut_rejected'] += len(chosen) - len(accepted)
        if not accepted:
            if rejected and len(self.edit_logs) < 20:
                self.edit_logs.append({'p': p, 'mode': 'use_shortcut_rejected', 'num_geo_cands': len(geo_cands), 'rejected': rejected[:3]})
            return set()
        committed_accepted = []
        for c in accepted:
            if c in cur:
                committed_accepted.append(c)
                continue
            drop = None
            if len(cur) >= self.R:
                drop_protected = protected_local | set(accepted)
                drop = self._pick_low_value_edge_to_replace(p, cur, protected=drop_protected)
                if drop is not None:
                    cur = [x for x in cur if x != drop]
                    removed.append(drop)
                elif getattr(self, '_require_shortcut_replacement', False):
                    self.debug_stats['num_gate_fail_no_replaceable_edge'] += 1
                    rejected.append({'id': c, 'reason': 'no_replaceable_edge'})
                    continue
            cur.append(c)
            committed_accepted.append(c)
        if getattr(self, '_require_shortcut_replacement', False):
            rejected_count = len(accepted) - len(committed_accepted)
            self.debug_stats['num_shortcut_accepted'] -= rejected_count
            self.debug_stats['num_shortcut_rejected'] += rejected_count
            accepted = committed_accepted
        self.edge[p] = list(dict.fromkeys([x for x in cur if x != p]))
        added = set(self.edge[p]) - old
        removed_set = old - set(self.edge[p])
        if added:
            self.debug_stats['num_nodes_with_edit'] += 1
            self.debug_stats['num_total_edits'] += len(added) + len(removed_set)
            self.debug_stats['num_edges_removed'] += len(removed_set)
            if removed_set:
                self.debug_stats['num_replacement_edits'] += 1
            self.edit_logs.append({'p': p, 'mode': 'query_sim_replace', 'added': sorted(added), 'removed': sorted(removed_set), 'rejected': rejected[:3], 'num_geo_cands': len(geo_cands)})
        return added

    def _score_low_value_edges(self, p: int, S: list[int], protected: set[int] | None=None) -> list[tuple]:
        protected = protected or set()
        candidates = [x for x in S if x != p and x not in protected]
        if len(candidates) <= 1:
            return []
        scored = []
        for x in candidates:
            S_wo = [y for y in S if y != x and y != p]
            twohop_wo = self._twohop_set_from_neighbor_set(p, S_wo)
            unique_gain = len(set(self.edge[x]) - twohop_wo)
            redundancy = self._min_cosdist_to_selected(S_wo, x)
            anchor_dist = dist(self.v[p], self.v[x])
            scored.append((unique_gain, redundancy, anchor_dist, x))
        victim_policy = os.getenv('LLM_INDEX_VICTIM_POLICY', 'lexicographic').strip().lower()
        if victim_policy == 'lexicographic':
            scored.sort()
        elif victim_policy == 'redundancy':
            scored.sort(key=lambda item: (item[1], item[3]))
        elif victim_policy == 'coverage':
            scored.sort(key=lambda item: (item[0], item[3]))
        else:
            raise ValueError(f'LLM_INDEX_VICTIM_POLICY must be lexicographic, redundancy, or coverage; got {victim_policy!r}')
        return scored

    def _pick_low_value_edge_to_replace(self, p: int, S: list[int], protected: set[int] | None=None):
        scored = self._score_low_value_edges(p, S, protected=protected)
        if not scored:
            return None
        return scored[0][-1]

    def interinsert(self, i: int, protected_neighbors: set[int] | None=None, semantic_prune_overflow: bool | None=None):
        if protected_neighbors is None:
            protected_neighbors = set()
        nbrs = list(self.edge[i])
        for j in nbrs:
            if j == i:
                continue
            if i not in self.edge[j]:
                self.edge[j].append(i)
            if len(self.edge[j]) > self.R:
                cand_pool = list(self.edge[j])
                if j in protected_neighbors:
                    self.robustprune_with_protection(j, cand_pool, self.alpha, self.R, protected={i})
                else:
                    self.robustprune(j, cand_pool, self.alpha, self.R)

    def _twohop_set_from_neighbor_set(self, p: int, S: list[int]) -> set[int]:
        out = set()
        for x in S:
            for y in self.edge[x]:
                if y != p:
                    out.add(y)
        return out

    def _shortlist_metrics(self, p, selected, candidates):
        cur_twohop = self._twohop_set_from_neighbor_set(p, selected)
        metrics = {}
        covered = {}
        for c in candidates:
            anchor_dist = dist(self.v[p], self.v[c])
            selected_dist = [dist(self.v[x], self.v[c]) for x in selected]
            metrics[c] = {'cand_anchor_dist': float(anchor_dist), 'cand_separation': float(min(selected_dist) if selected_dist else 1000000000.0), 'cand_unique_2hop_gain': int(len(set(self.edge[c]) - cur_twohop))}
            covered[c] = any((d * self.alpha <= anchor_dist for d in selected_dist))
        return (metrics, covered)

    def _candidate_coverage_gain(self, p: int, selected: list[int], c: int) -> int:
        current = self._twohop_set_from_neighbor_set(p, selected)
        return len(set(self.edge[c]) - current)
