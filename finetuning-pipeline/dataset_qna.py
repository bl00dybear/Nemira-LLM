from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:

    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return []

    # upload JSON
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except JSONDecodeError:
        pass

    # Fallback JSONL
    out: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_qa_dataset(paths: Sequence[str] | str,
                    max_samples: Optional[int] = None,
                    filter_empty: bool = True) -> List[Dict[str, Any]]:
    if isinstance(paths, str):
        paths = [paths]

    all_items: List[Dict[str, Any]] = []
    for p in paths:
        items = _load_json_or_jsonl(p)
        all_items.extend(items)

    # Norm + filter
    filtered: List[Dict[str, Any]] = []
    for obj in all_items:
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            continue
        norm_msgs = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue
            norm_msgs.append({"role": role, "content": content})
        if not norm_msgs:
            continue
        if filter_empty:
            has_user = any(m["role"] == "user" for m in norm_msgs)
            has_assistant = any(m["role"] == "assistant" for m in norm_msgs)
            if not (has_user and has_assistant):
                continue
        filtered.append({"messages": norm_msgs})

    if max_samples:
        filtered = filtered[:max_samples]

    print(f"[STATUS] Loaded {len(filtered)} Q&A examples (after filtering)")
    return filtered


def _format_mistral_prompt_for_target(messages: List[Dict[str, str]], target_assistant_idx: int) -> Optional[Dict[str, str]]:
    if target_assistant_idx <= 0 or target_assistant_idx >= len(messages):
        return None
    if messages[target_assistant_idx]["role"] != "assistant":
        return None
    
    prompt_parts: List[str] = []
    i = 0
    while i < target_assistant_idx:
        m = messages[i]
        if m["role"] == "user":
            user_content = m["content"].strip()
            assistant_follow = ""
           
            if i + 1 < target_assistant_idx and messages[i + 1]["role"] == "assistant":
                assistant_follow = " " + messages[i + 1]["content"].strip()
                i += 1
            prompt_parts.append(f"[INST] {user_content} [/INST]{assistant_follow}")
        elif m["role"] == "assistant":
           
            prompt_parts.append(messages[i]["content"].strip())
        i += 1

    prompt_text = "".join(prompt_parts) if prompt_parts else "[INST] [/INST]"
    response_text = " " + messages[target_assistant_idx]["content"].strip()

    return {"prompt": prompt_text, "response": response_text}


def _conversation_to_examples(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

    examples: List[Dict[str, str]] = []
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        pair = _format_mistral_prompt_for_target(messages, idx)
        if pair is not None:
            examples.append(pair)
    return examples


class InstructionDataset(Dataset):

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_len: int = 2048,
        min_resp_tokens: int = 16,
    ) -> None:
        assert hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None, "Set tokenizer.pad_token_id înainte de a crea datasetul."
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.min_resp_tokens = int(min_resp_tokens)

        # Aplatizează conversațiile în exemple (prompt, response)
        flat: List[Dict[str, str]] = []
        for obj in data:
            msgs = obj.get("messages", [])
            if not isinstance(msgs, list) or not msgs:
                continue
            exs = _conversation_to_examples(msgs)
            flat.extend(exs)

        self.examples: List[Dict[str, str]] = flat

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_text = ex["prompt"]
        response_text = ex["response"] if ex["response"].strip() else " [OK]"

         # 1) Limitează promptul la (max_len - min_resp_tokens)
        max_prompt_len = max(1, self.max_len - self.min_resp_tokens)
        p = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids: List[int] = p["input_ids"]

        # 2) Limitează răspunsul la spațiul rămas
        remain = max(1, self.max_len - len(prompt_ids))
        r = self.tokenizer(
            response_text,
            add_special_tokens=False,
            truncation=True,
            max_length=remain,
        )
        resp_ids: List[int] = r["input_ids"]

        input_ids: List[int] = prompt_ids + resp_ids
        attention_mask: List[int] = [1] * len(input_ids)

        # Padding la max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        # Labels: maschează prompt + padding
        labels = input_ids_tensor.clone()
        prompt_len = min(len(prompt_ids), self.max_len)
        labels[:prompt_len] = -100
        labels[attention_mask_tensor == 0] = -100

        # Dacă nu a rămas nimic antrenabil (răspuns trunchiat complet), injectează un fallback compact
        if (labels != -100).sum().item() == 0:
            fb = self.tokenizer(" [OK]", add_special_tokens=False)["input_ids"]
            pos = prompt_len
            for t in fb:
                if pos >= self.max_len:
                    break
                input_ids_tensor[pos] = t
                attention_mask_tensor[pos] = 1
                labels[pos] = t
                pos += 1

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
        }