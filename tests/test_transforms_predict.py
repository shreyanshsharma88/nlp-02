from ner.data import CASE_VARIANTS, case_transform, entity_spans, title_word, word_shape
from ner.predict import simple_tokenize, spans_to_text, words_from_subtokens

SENT = ["McDonald's", "CEO", "met", "NATO", "in", "São", "Paulo", "."]


def test_case_transforms_keep_length_and_labels():
    labels = [3, 0, 0, 3, 0, 5, 6, 0]
    for v in CASE_VARIANTS:
        out = case_transform(SENT, v)
        assert len(out) == len(SENT) == len(labels)
    assert case_transform(SENT, "lower")[0] == "mcdonald's"
    assert case_transform(SENT, "upper")[3] == "NATO"
    assert case_transform(SENT, "title")[1] == "Ceo"
    assert title_word("McDonald's") == "Mcdonald's"  # not "Mcdonald'S"
    assert case_transform(SENT, "original") == SENT


def test_word_shape():
    assert word_shape("Paris") == "Title" and word_shape("NATO") == "ALLCAPS"
    assert word_shape("eBay") == "mixed" and word_shape("apple") == "lower" and word_shape("1996") == "nonalpha"


def test_words_from_subtokens():
    # [CLS] Gy ##llen ##haal visited [SEP]
    wids = [None, 0, 0, 0, 1, None]
    sub = ["O", "B-PER", "I-PER", "I-PER", "O", "O"]
    assert words_from_subtokens(wids, sub, 2) == ["B-PER", "O"]


def test_bio_decoding_with_orphan_I():
    tags = ["I-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-ORG", "B-ORG"]
    assert entity_spans(tags) == [("PER", 0, 2), ("LOC", 3, 5), ("ORG", 5, 6), ("ORG", 6, 7)]
    words = ["Tim", "Cook", "in", "New", "York", "X", "Y"]
    assert spans_to_text(words, tags).startswith("Tim Cook [PER]; New York [LOC]")


def test_simple_tokenize():
    assert simple_tokenize("Apple's CEO, Tim Cook.") == ["Apple's", "CEO", ",", "Tim", "Cook", "."]
