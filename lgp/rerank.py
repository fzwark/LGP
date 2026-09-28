from __future__ import annotations
import os
from .prompts import LLM_SEARCH_SYS, LLM_SEARCH_USER

_RERANK_MAX_RAW_CHARS = int(os.getenv('LLM_RERANK_MAX_RAW_CHARS', '80000'))

_RERANK_DOC_MAX_TOKENS = int(os.getenv('LLM_RERANK_DOC_MAX_TOKENS', '2800'))

_RERANK_QUERY_MAX_TOKENS = int(os.getenv('LLM_RERANK_QUERY_MAX_TOKENS', '1800'))

def _truncate_rerank_text(llm, text: str, max_tokens: int) -> str:
    text = str(text or '')
    token_ids = llm.tok.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    marker = '\n...[middle truncated for context limit]...\n'
    marker_ids = llm.tok.encode(marker, add_special_tokens=False)
    content_budget = max(2, max_tokens - len(marker_ids))
    head_budget = max(1, int(content_budget * 0.8))
    tail_budget = max(1, content_budget - head_budget)
    return llm.tok.decode(token_ids[:head_budget], skip_special_tokens=True) + marker + llm.tok.decode(token_ids[-tail_budget:], skip_special_tokens=True)

def llm_rank(llm, query_text: str, cand_pids: list[int], p_texts: list[str]) -> list[int]:
    if not cand_pids:
        return []
    query_for_prompt = str(query_text or '')
    passage_texts = [str(p_texts[pid] or '') for pid in cand_pids]
    raw_chars = len(query_for_prompt) * 2 + sum((len(text) for text in passage_texts))
    if raw_chars > _RERANK_MAX_RAW_CHARS:
        query_for_prompt = _truncate_rerank_text(llm, query_for_prompt, _RERANK_QUERY_MAX_TOKENS)
        passage_texts = [_truncate_rerank_text(llm, text, _RERANK_DOC_MAX_TOKENS) for text in passage_texts]
    blocks = []
    for (i, text) in enumerate(passage_texts):
        blocks.append(f'[{i + 1}] {text}')
    passages_block = '\n\n'.join(blocks)
    user_prompt = passages_block + '\n\n' + LLM_SEARCH_USER.format(QUERY=query_for_prompt, N=len(cand_pids))
    raw_output = llm.run_one_message(sys=LLM_SEARCH_SYS.format(QUERY=query_for_prompt, N=len(cand_pids)), input=user_prompt)
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)
    positions = []
    for i in range(len(cand_pids)):
        token = f'[{i + 1}]'
        pos = raw_output.find(token)
        if pos == -1:
            pos = 10 ** 9 + i
        positions.append((pos, i))
    positions.sort()
    ordered_pids = [cand_pids[i] for (_, i) in positions]
    return ordered_pids

def sliding_rerank(candidates, rank_window, window=10, stride=5):
    if not 0 < stride <= window:
        raise ValueError("Require 0 < stride <= window")
    ordered = list(candidates)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Candidates must be unique")
    end = len(ordered)
    calls = 0
    while end > 0:
        start = max(0, end-window)
        batch = ordered[start:end]
        if len(batch) > 1:
            ranked = list(rank_window(batch))
            if sorted(ranked) != sorted(batch):
                raise ValueError("Reranker changed candidate membership")
            ordered[start:end] = ranked
            calls += 1
        end = max(0, end-stride)
    return ordered, calls
