"""Tokenization with word_ids() label alignment (requirement 2).

Choice: only the FIRST sub-token of each word carries the word's label; the remaining sub-tokens and the
special tokens ([CLS], [SEP], padding) get -100 so the loss and metrics ignore them. The alternative is to
label every sub-token (B-X -> I-X for continuation pieces); we use first-sub-token labelling because it makes
evaluation exactly word-level and is the convention used by the original BERT NER setup.
"""
from __future__ import annotations

import collections
from typing import Dict, List

IGNORE = -100


def align_labels(word_ids: List, word_labels: List[int]) -> List[int]:
    labels, prev = [], None
    for w in word_ids:
        if w is None:
            labels.append(IGNORE)
        elif w != prev:
            labels.append(word_labels[w])
        else:
            labels.append(IGNORE)
        prev = w
    return labels


def tokenize_and_align(batch: Dict, tokenizer, max_length: int = 128, tokens_key: str = "tokens") -> Dict:
    enc = tokenizer(batch[tokens_key], is_split_into_words=True, truncation=True, max_length=max_length)
    enc["labels"] = [align_labels(enc.word_ids(i), tags) for i, tags in enumerate(batch["ner_tags"])]
    return enc


def tokenize_dataset(ds, tokenizer, max_length: int = 128):
    cols = ds["train"].column_names
    return ds.map(lambda b: tokenize_and_align(b, tokenizer, max_length), batched=True, remove_columns=cols)


def show_tokenization(tokenizer, words: List[str]):
    enc = tokenizer(words, is_split_into_words=True)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    return list(zip(toks, enc.word_ids()))


def subword_stats(tokenizer, ds_split, names: List[str], max_length: int = 128) -> Dict:
    """Pieces per word, share of words split, [UNK] rate overall and for entity words, truncation count."""
    unk = tokenizer.unk_token
    n_words = n_pieces = n_split = n_unk = 0
    ent = collections.Counter()
    truncated_sents = truncated_words = 0
    for toks, tags in zip(ds_split["tokens"], ds_split["ner_tags"]):
        pieces_per_word = [tokenizer.tokenize(w) for w in toks]
        total = sum(len(p) for p in pieces_per_word) + 2
        if total > max_length:
            truncated_sents += 1
            running, kept = 2, 0
            for p in pieces_per_word:
                if running + len(p) > max_length:
                    break
                running += len(p)
                kept += 1
            truncated_words += len(toks) - kept
        for p, g in zip(pieces_per_word, tags):
            n_words += 1
            n_pieces += len(p)
            split = len(p) > 1
            is_unk = unk in p
            n_split += split
            n_unk += is_unk
            if g:
                ent["words"] += 1
                ent["split"] += split
                ent["unk"] += is_unk
                ent["pieces"] += len(p)
    return {
        "avg_pieces_per_word": n_pieces / n_words,
        "share_words_split": n_split / n_words,
        "unk_rate_words": n_unk / n_words,
        "entity_avg_pieces": ent["pieces"] / ent["words"],
        "entity_share_split": ent["split"] / ent["words"],
        "entity_unk_rate": ent["unk"] / ent["words"],
        "n_unk_words": n_unk,
        "n_entity_unk_words": ent["unk"],
        "truncated_sentences": truncated_sents,
        "truncated_words": truncated_words,
    }
