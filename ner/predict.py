"""Word-level NER prediction for pre-split or raw sentences (requirement 6).

Sub-tokens are merged back into words by taking the prediction of each word's FIRST sub-token
(the same convention used for the training labels).
"""
from __future__ import annotations

import re
from typing import List, Sequence

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner.data import entity_spans

_WORD_RE = re.compile(r"\w+(?:[-'’.&]\w+)*|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def words_from_subtokens(word_ids: Sequence, sub_preds: Sequence[str], n_words: int) -> List[str]:
    """Merge sub-token predictions to words using the first sub-token of each word ('O' if truncated)."""
    out = ["O"] * n_words
    prev = None
    for w, p in zip(word_ids, sub_preds):
        if w is not None and w != prev:
            out[w] = p
        prev = w
    return out


def spans_to_text(words: List[str], tags: List[str]) -> str:
    sp = entity_spans(tags)
    return "; ".join(f"{' '.join(words[s:e])} [{t}]" for t, s, e in sp) or "(none)"


class NERPredictor:
    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(device).eval()
        self.id2label = self.model.config.id2label
        self.max_length = max_length

    @torch.no_grad()
    def predict_words(self, sentences: List[List[str]], batch_size: int = 64) -> List[List[str]]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = self.tokenizer(batch, is_split_into_words=True, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            ids = logits.argmax(-1).cpu().numpy()
            for j, words in enumerate(batch):
                sub = [self.id2label[int(x)] for x in ids[j]]
                results.append(words_from_subtokens(enc.word_ids(j), sub, len(words)))
        return results

    def predict_text(self, text: str):
        words = simple_tokenize(text)
        tags = self.predict_words([words])[0]
        return words, tags, entity_spans(tags)
