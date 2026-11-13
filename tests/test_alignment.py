import pytest
from transformers import AutoTokenizer

from ner.tokenize import IGNORE, align_labels, tokenize_and_align

WORDS = ["Apple", "CEO", "Tim", "Cook", "visited", "Washington", "and", "Gyllenhaal", "."]
TAGS = [3, 0, 1, 2, 0, 5, 0, 1, 0]


@pytest.fixture(scope="module", params=["bert-base-cased", "bert-base-uncased"])
def tok(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_first_subtoken_gets_label_rest_ignored(tok):
    enc = tokenize_and_align({"tokens": [WORDS], "ner_tags": [TAGS]}, tok)
    labels, wids = enc["labels"][0], enc.word_ids(0)
    prev = None
    for lab, w in zip(labels, wids):
        if w is None:
            assert lab == IGNORE  # [CLS]/[SEP]
        elif w != prev:
            assert lab == TAGS[w]
        else:
            assert lab == IGNORE
        prev = w
    assert sum(l != IGNORE for l in labels) == len(WORDS)
    assert any(wids.count(i) > 1 for i in range(len(WORDS))), "test sentence should contain a split word"


def test_special_tokens_ignored(tok):
    enc = tokenize_and_align({"tokens": [WORDS], "ner_tags": [TAGS]}, tok)
    assert enc["labels"][0][0] == IGNORE and enc["labels"][0][-1] == IGNORE


def test_align_labels_pure():
    assert align_labels([None, 0, 0, 1, None], [5, 6]) == [IGNORE, 5, IGNORE, 6, IGNORE]


def test_uncased_tokenizer_ignores_case():
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    a = t(WORDS, is_split_into_words=True)["input_ids"]
    b = t([w.lower() for w in WORDS], is_split_into_words=True)["input_ids"]
    c = t([w.upper() for w in WORDS], is_split_into_words=True)["input_ids"]
    assert a == b == c


def test_cased_tokenizer_sees_case():
    t = AutoTokenizer.from_pretrained("bert-base-cased")
    a = t(WORDS, is_split_into_words=True)["input_ids"]
    b = t([w.lower() for w in WORDS], is_split_into_words=True)["input_ids"]
    assert a != b
