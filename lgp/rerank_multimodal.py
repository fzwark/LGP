from __future__ import annotations
import base64
import functools
import io
import os
from PIL import Image, ImageOps

@functools.lru_cache(maxsize=2048)
def _image_to_data_url(path: str) -> str:
    max_side = max(64, int(os.getenv('LLM_RERANK_MM_IMAGE_MAX_SIDE', '512')))
    quality = min(95, max(30, int(os.getenv('LLM_RERANK_MM_JPEG_QUALITY', '80'))))
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'

def _safe_item_text(item: dict, max_chars: int=400) -> str:
    txt = item.get('text', '') or item.get('src_content', '') or ''
    txt = str(txt).replace('\n', ' ').strip()
    return txt[:max_chars]

def _item_to_mm_parts(item: dict, prefix: str, text_chars: int=400) -> list:
    parts = []
    txt = _safe_item_text(item, text_chars)
    img = str(item.get('image', '') or '').strip()
    if txt:
        parts.append({'type': 'text', 'text': f'{prefix} text:\n{txt}'})
    if img and os.path.exists(img):
        parts.append({'type': 'image_url', 'image_url': {'url': _image_to_data_url(img)}})
    elif img:
        raise FileNotFoundError(f'Declared rerank image does not exist: {img}')
    if not parts:
        parts.append({'type': 'text', 'text': f'{prefix} text:\n'})
    return parts

def llm_rank_mm(llm, query_item: dict, cand_pids: list[int], p_items: list[dict], dataset_name: str='', query_prompt_map: dict | None=None, cand_text_chars: int=160, query_text_chars: int=320) -> list[int]:
    if not cand_pids:
        return []
    query_prompt = ''
    if query_prompt_map is not None:
        query_prompt = query_prompt_map.get(dataset_name, '') or ''
    q_text = _safe_item_text(query_item, query_text_chars)
    q_text_full = ' '.join([x for x in [query_prompt, q_text] if x]).strip()
    cand_lines = []
    for (i, pid) in enumerate(cand_pids):
        cand_lines.append(f'[{i + 1}] {_safe_item_text(p_items[pid], cand_text_chars)}')
    cand_block = '\n\n'.join(cand_lines)
    sys = 'You rank retrieval candidates for multimodal retrieval.\nGiven one QUERY and several CANDIDATES, rank the candidates from best to worst match.\nUse all available evidence, including image and text when present.\nReturn only an ordering of candidate indices like [2] [1] [3].'
    user = f'QUERY:\n{q_text_full}\n\nCANDIDATES:\n{cand_block}\n\nRank the CANDIDATES from best to worst match for the QUERY.\nReturn only candidate indices like [2] [1] [3].'
    parts = []
    parts.append({'type': 'text', 'text': 'QUERY:'})
    if query_prompt:
        parts.append({'type': 'text', 'text': f'Task instruction:\n{query_prompt}'})
    parts.extend(_item_to_mm_parts(query_item, prefix='Query', text_chars=query_text_chars))
    parts.append({'type': 'text', 'text': 'CANDIDATES:'})
    for (i, pid) in enumerate(cand_pids):
        parts.append({'type': 'text', 'text': f'[{i + 1}]'})
        parts.extend(_item_to_mm_parts(p_items[pid], prefix=f'Candidate {i + 1}', text_chars=cand_text_chars))
    parts.append({'type': 'text', 'text': 'Rank the CANDIDATES from best to worst match for the QUERY.\nUse image and text jointly when available.\nReturn only candidate indices like [2] [1] [3].'})
    if not hasattr(llm, 'run_one_multimodal_message'):
        raise TypeError('Multimodal reranking requires a multimodal client')
    raw_output = llm.run_one_multimodal_message(parts=parts, sys=sys)
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
