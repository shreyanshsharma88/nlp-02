"""Dataset loading, label maps, case-variant transforms and capitalization profiling (requirement 1)."""
from __future__ import annotations

import collections
from typing import Dict, List

from datasets import DatasetDict, load_dataset

# The canonical "conll2003" / "eriktks/conll2003" repos ship a loading script, which datasets>=4 refuses to run.
# The Hub's automatic parquet conversion of the same repo contains identical data and ClassLabel features.
DATASET_SOURCES = [
    ("conll2003", {}),
    ("eriktks/conll2003", {"revision": "refs/convert/parquet"}),
    ("eriktks/conll2003", {}),
]
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
CASE_VARIANTS = ["original", "lower", "upper", "title"]


def load_conll(verbose: bool = True) -> DatasetDict:
    """Load CoNLL-2003, trying the sources in order. Records the working source on the returned object."""
    errors = []
    for name, kw in DATASET_SOURCES:
        try:
            ds = load_dataset(name, **kw)
            ds.source = f"{name} {kw}" if kw else name
            if verbose:
                print(f"Loaded CoNLL-2003 from: {ds.source}")
            return ds
        except Exception as e:  # noqa: BLE001 - we want to try every source
            errors.append(f"{name} {kw}: {type(e).__name__}: {str(e)[:120]}")
            if verbose:
                print("  failed:", errors[-1])
    raise RuntimeError("Could not load CoNLL-2003:\n" + "\n".join(errors))


def label_maps(ds: DatasetDict):
    names: List[str] = ds["train"].features["ner_tags"].feature.names
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}
    return names, id2label, label2id


# ---------------------------------------------------------------- case transforms
def title_word(w: str) -> str:
    """Title-case one word: first character upper, rest lower (str.title() would give "Mcdonald'S")."""
    return w[:1].upper() + w[1:].lower()


def case_transform(tokens: List[str], variant: str) -> List[str]:
    """Return a case-perturbed copy of a sentence. Word count is preserved, so gold labels stay aligned."""
    if variant == "original":
        return list(tokens)
    if variant == "lower":
        return [t.lower() for t in tokens]
    if variant == "upper":
        return [t.upper() for t in tokens]
    if variant == "title":
        return [title_word(t) for t in tokens]
    raise ValueError(variant)


# ---------------------------------------------------------------- capitalization profile
def word_shape(w: str) -> str:
    if not any(c.isalpha() for c in w):
        return "nonalpha"
    letters = [c for c in w if c.isalpha()]
    if all(c.isupper() for c in letters):
        return "ALLCAPS" if len(letters) > 1 else "Title"
    if all(c.islower() for c in letters):
        return "lower"
    if w[0].isupper() and all(c.islower() for c in letters[1:]):
        return "Title"
    return "mixed"


def is_allcaps_sentence(tokens: List[str]) -> bool:
    alpha = [t for t in tokens if any(c.isalpha() for c in t)]
    return len(alpha) >= 2 and all(word_shape(t) == "ALLCAPS" or (len(t) == 1 and t.isupper()) for t in alpha)


def capitalization_profile(ds: DatasetDict, names: List[str]) -> Dict:
    out = {}
    for split in ds:
        d = ds[split]
        shapes = collections.defaultdict(collections.Counter)
        n_caps = n_doc = 0
        first_cap = collections.Counter()
        for toks, tags in zip(d["tokens"], d["ner_tags"]):
            if toks == ["-DOCSTART-"]:
                n_doc += 1
                continue
            n_caps += is_allcaps_sentence(toks)
            for i, (t, g) in enumerate(zip(toks, tags)):
                ty = names[g][2:] if g else "O"
                shapes[ty][word_shape(t)] += 1
                if i == 0 and t[:1].isupper():
                    first_cap["entity" if g else "O"] += 1
        out[split] = {
            "n_sentences": len(d),
            "n_docstart": n_doc,
            "n_allcaps_sentences": n_caps,
            "sentence_initial_capitalised": dict(first_cap),
            "shape_counts": {k: dict(v) for k, v in shapes.items()},
            "shape_share": {k: {s: c / sum(v.values()) for s, c in v.items()} for k, v in shapes.items()},
        }
    return out


def entity_spans(tags: List[str]):
    """Return (type, start, end_exclusive) spans from a BIO tag sequence (I- without B- starts a span)."""
    spans, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                spans.append((typ, start, i))
                typ = None
            if t != "O":
                typ, start = t[2:], i
    return spans


def case_augment(ds_train, p_lower: float = 0.25, p_upper: float = 0.10, seed: int = 42):
    """Extra experiment: randomly lower-case (p_lower) or upper-case (p_upper) whole training sentences.
    Same number of sentences and labels; only the surface casing changes."""
    import random
    rng = random.Random(seed)

    def f(ex):
        r = rng.random()
        if r < p_lower:
            ex["tokens"] = case_transform(ex["tokens"], "lower")
        elif r < p_lower + p_upper:
            ex["tokens"] = case_transform(ex["tokens"], "upper")
        return ex
    return ds_train.map(f, load_from_cache_file=False)
