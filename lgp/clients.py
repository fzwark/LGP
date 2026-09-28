import os
import re

import requests


class VLLMClient:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", model="lgp",
                 tokenizer="Qwen/Qwen3-32B", max_tokens=256, timeout=600,
                 multimodal=False):
        self.base_url = base_url.rstrip("/")
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.session = requests.Session()
        if os.getenv("LGP_API_KEY"):
            self.session.headers["Authorization"] = "Bearer " + os.environ["LGP_API_KEY"]
        self.tok = None
        if not multimodal:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def _post(self, route, payload):
        response = self.session.post(self.base_url + route, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _sampling(self):
        return {"model": self.model, "max_tokens": self.max_tokens,
                "temperature": 0.0, "top_p": 1.0}

    def run_one_message(self, sys="", input=""):
        if self.tok is None:
            raise ValueError("Text completion requires a tokenizer")
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": input}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        result = self._post("/completions", {**self._sampling(), "prompt": prompt})
        return (result["choices"][0].get("text", "") or "").strip()

    def run_one_multimodal_message(self, *, parts, sys=""):
        messages = ([{"role": "system", "content": sys}] if sys else [])
        messages.append({"role": "user", "content": parts})
        result = self._post("/chat/completions", {**self._sampling(), "messages": messages})
        raw = (result["choices"][0]["message"].get("content", "") or "").strip()
        for tag in ("</think>", "</reasoning>"):
            if tag in raw:
                return raw.split(tag)[-1].strip()
        for pattern in (r"(?:^|\n)\s*final answer\s*:\s*(.*)$", r"(?:^|\n)\s*final\s*:\s*(.*)$"):
            match = re.search(pattern, raw, flags=re.I | re.S)
            if match:
                return match.group(1).strip()
        return raw
