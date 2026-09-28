from __future__ import annotations
import base64
import functools
import io
import os
import re
from pathlib import Path
from PIL import Image, ImageOps
from .graph import dist, point
from .graph import diskann_graph as _TextDiskANNGraph

class diskann_graph(_TextDiskANNGraph):

    def __init__(self, A, L, R, alpha, llm=None, texts=None, items=None, fla=0, llm_anchor_chars=600, llm_keep_chars=260, llm_cand_chars=260, llm_cache=True, llm_max_calls_per_prune=10000000000, seed: int=42):
        self.items = list(items) if items is not None else None
        if texts is None and self.items is not None:
            texts = [self._raw_item_text(item) for item in self.items]
        super().__init__(A=A, L=L, R=R, alpha=alpha, llm=llm, texts=texts, fla=fla, llm_anchor_chars=llm_anchor_chars, llm_keep_chars=llm_keep_chars, llm_cand_chars=llm_cand_chars, llm_cache=llm_cache, llm_max_calls_per_prune=llm_max_calls_per_prune, seed=seed)
        self._mm_request_failures = 0

    @staticmethod
    def _raw_item_text(item: dict | None) -> str:
        item = item or {}
        text = item.get('text', '') or item.get('src_content', '') or ''
        return str(text).replace('\n', ' ').strip()

    def _get_item(self, idx: int) -> dict:
        if self.items is not None and 0 <= idx < len(self.items):
            return self.items[idx] or {}
        if self.texts is not None and 0 <= idx < len(self.texts):
            return {'text': self.texts[idx], 'image': '', 'modality': 'text', 'src_content': ''}
        return {}

    def _txt(self, idx: int, max_chars: int) -> str:
        text = self._raw_item_text(self._get_item(idx))
        if os.getenv('LLM_INDEX_TRUNCATE_TEXT', '1').lower() in {'0', 'false', 'no'}:
            return text
        return text[:max(0, int(max_chars))]

    def _img(self, idx: int) -> str:
        return str(self._get_item(idx).get('image', '') or '').strip()

    @functools.lru_cache(maxsize=2048)
    def _image_to_data_url(self, image_path: str) -> str:
        max_side = max(64, int(os.getenv('LLM_INDEX_MM_IMAGE_MAX_SIDE', '512')))
        quality = min(95, max(30, int(os.getenv('LLM_INDEX_MM_JPEG_QUALITY', '80'))))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f'data:image/jpeg;base64,{encoded}'

    def _item_to_mm_parts(self, idx: int, prefix: str, text_chars: int) -> list[dict]:
        parts: list[dict] = []
        text = self._txt(idx, text_chars)
        image_path = self._img(idx)
        if text:
            parts.append({'type': 'text', 'text': f'{prefix} text:\n{text}'})
        if image_path:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f'Declared image for item {idx} does not exist: {image_path}')
            parts.append({'type': 'image_url', 'image_url': {'url': self._image_to_data_url(image_path)}})
        if not parts:
            parts.append({'type': 'text', 'text': f'{prefix}: (no text or image content)'})
        return parts

    def _run_vlm(self, *, sys: str, parts: list[dict]) -> str:
        if self.llm is None:
            return ''
        if not hasattr(self.llm, 'run_one_multimodal_message'):
            raise TypeError('The multimodal graph requires an LLM client with run_one_multimodal_message().')
        return self.llm.run_one_multimodal_message(parts=parts, sys=sys)

    def _warn_request_failure(self, exc: Exception) -> None:
        if os.getenv('LLM_INDEX_FAIL_ON_LLM_ERROR', '0').lower() in {'1', 'true', 'yes'}:
            raise RuntimeError('Multimodal shortcut selection request failed') from exc
        self._mm_request_failures += 1
        if self._mm_request_failures <= 5:
            print(f'WARNING: multimodal pruning request failed; keeping the graph unchanged for this decision: {exc}')
        elif self._mm_request_failures == 6:
            print('WARNING: suppressing further multimodal request errors')

    def _llm_select_use_shortcuts(self, anchor_id: int, selected_ids: list[int], cand_ids: list[int], keep: int=2, selected_cap: int=4, cand_cap: int=16, generic_prompt: bool=False, general_question_prompt: bool=False, refined_general_prompt: bool=False, structural_general_prompt: bool=False, query_overlap_prompt: bool=False, query_route_prompt: bool=False) -> list[int]:
        if keep <= 0 or not cand_ids or self.llm is None or (self.items is None):
            return []
        selected_ids = list(dict.fromkeys((node for node in selected_ids if node != anchor_id)))
        selected_ids.sort(key=lambda node: dist(self.v[anchor_id], self.v[node]))
        selected_ids = selected_ids[:selected_cap]
        cand_ids = list(dict.fromkeys((node for node in cand_ids if node != anchor_id and node not in selected_ids)))
        cand_ids.sort(key=lambda node: (-self._candidate_coverage_gain(anchor_id, selected_ids, node), dist(self.v[anchor_id], self.v[node]), node))
        cand_ids = cand_ids[:cand_cap]
        if not cand_ids:
            return []
        prompt_version = 'mm_control_v1'
        key = ('query_sim_shortcut', prompt_version, anchor_id, tuple(selected_ids), tuple(cand_ids), keep)
        if self.llm_cache and key in self._llm_judge_cache:
            return self._llm_judge_cache[key]
        anchor_chars = int(os.getenv('LLM_INDEX_ANCHOR_CHARS', '1200'))
        keep_chars = int(os.getenv('LLM_INDEX_KEEP_CHARS', '700'))
        cand_chars = int(os.getenv('LLM_INDEX_CAND_CHARS', '900'))
        sys = f'You are editing a multimodal retrieval graph at indexing time. There is no real evaluation query. Privately infer plausible user queries or intents for which the ANCHOR ITEM could be retrieved or visited. Use all supplied text and visual evidence.\n\nSelect a CANDIDATE SHORTCUT only when an edge from the anchor to that candidate would help query-time graph search reach a useful target, prerequisite, continuation, contrasting case, or bridge that is not already covered by the RETAINED LOCAL NEIGHBORS.\n\nVisual or textual similarity alone is insufficient. A candidate may be useful despite looking different when it supports the same retrieval task. Do not invent relationships unsupported by the items. If no candidate adds a clear route, return NONE. Return at most {keep} bracketed candidate numbers, best first, and nothing else; for example [3] [1] or NONE.'
        parts: list[dict] = [{'type': 'text', 'text': 'ANCHOR ITEM:'}]
        parts.extend(self._item_to_mm_parts(anchor_id, 'Anchor', anchor_chars))
        parts.append({'type': 'text', 'text': 'RETAINED LOCAL NEIGHBORS (routes already available):'})
        if selected_ids:
            for (rank, node) in enumerate(selected_ids, start=1):
                parts.append({'type': 'text', 'text': f'(K{rank})'})
                parts.extend(self._item_to_mm_parts(node, f'Retained neighbor K{rank}', keep_chars))
        else:
            parts.append({'type': 'text', 'text': '(none)'})
        parts.append({'type': 'text', 'text': 'CANDIDATE SHORTCUTS:'})
        for (rank, node) in enumerate(cand_ids, start=1):
            parts.append({'type': 'text', 'text': f'[{rank}]'})
            parts.extend(self._item_to_mm_parts(node, f'Candidate {rank}', cand_chars))
        parts.append({'type': 'text', 'text': 'Choose candidates that add the greatest distinct query-time routing value beyond the retained neighbors. Return only bracketed candidate numbers or NONE.'})
        try:
            output = self._run_vlm(sys=sys, parts=parts)
        except Exception as exc:
            self._warn_request_failure(exc)
            return []
        output = str(output or '')
        if '</think>' in output:
            output = output.split('</think>')[-1]
        if '</reasoning>' in output:
            output = output.split('</reasoning>')[-1]
        chosen: list[int] = []
        matches = re.findall('\\[(\\d+)\\]', output)
        if not matches and 'NONE' not in output.upper():
            matches = re.findall('(?<!\\d)(\\d+)(?!\\d)', output)
        for match in matches:
            position = int(match) - 1
            if 0 <= position < len(cand_ids):
                node = cand_ids[position]
                if node not in chosen:
                    chosen.append(node)
            if len(chosen) >= keep:
                break
        if self.llm_cache:
            self._llm_judge_cache[key] = chosen
        return chosen
__all__ = ['point', 'dist', 'diskann_graph']
