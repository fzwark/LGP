
from __future__ import annotations


VALID_CANDIDATE_SOURCES = ("balanced", "near", "gain", "diverse")


def normalize_candidate_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in VALID_CANDIDATE_SOURCES:
        choices = ", ".join(VALID_CANDIDATE_SOURCES)
        raise ValueError(
            f"LLM_INDEX_CANDIDATE_SOURCE must be one of: {choices}; "
            f"got {value!r}"
        )
    return source


def compose_candidate_shortlist(
    near_rank,
    gain_rank,
    diverse_rank,
    cand_cap: int,
    source: str = "balanced",
):
    source = normalize_candidate_source(source)
    cand_cap = int(cand_cap)
    if cand_cap <= 0:
        return []

    near_rank = list(near_rank)
    gain_rank = list(gain_rank)
    diverse_rank = list(diverse_rank)
    if source == "balanced":
        per_source_cap = max(4, cand_cap // 3)
        ranked = (
            near_rank[:per_source_cap]
            + gain_rank[:per_source_cap]
            + diverse_rank
        )
    elif source == "near":
        ranked = near_rank
    elif source == "gain":
        ranked = gain_rank
    else:
        ranked = diverse_rank

    result = []
    for candidate in ranked:
        if candidate not in result:
            result.append(candidate)
        if len(result) >= cand_cap:
            break
    return result
