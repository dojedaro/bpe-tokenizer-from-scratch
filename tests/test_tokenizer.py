from collections import Counter
from itertools import pairwise

import pytest

from bpe import BPETokenizer, chunks
from bpe.tokenizer import _merge

CORPUS = ["the cat sat on the mat", "the cat ate the rat", "a cat, a hat, a bat."] * 20


def naive_train(texts: list[str], vocab_size: int) -> list[tuple[int, int]]:
    """Recount every pair after every merge. Slow, obviously correct, same tie-break."""
    words = [list(c) for t in texts for c in chunks(t)]
    merges = []
    while 256 + len(merges) < vocab_size:
        counts = Counter(p for w in words for p in pairwise(w))
        if not counts:
            break
        best_n = max(counts.values())
        if best_n < 2:
            break
        pair = min(p for p, n in counts.items() if n == best_n)
        new_id = 256 + len(merges)
        merges.append(pair)
        words = [_merge(w, pair, new_id) for w in words]
    return merges


def test_incremental_training_matches_naive_recount():
    assert BPETokenizer.train(CORPUS, 300).merges == naive_train(CORPUS, 300)


def test_round_trip_on_arbitrary_unicode():
    tok = BPETokenizer.train(CORPUS, 300)
    for text in ["", "the cat", "unseen words entirely", "日本語 and émojis 🐈", "  spaces\n\ttabs  ", "x_y__z 123"]:
        assert tok.decode(tok.encode(text)) == text


def test_merges_compress_the_training_text():
    tok = BPETokenizer.train(CORPUS, 300)
    text = " ".join(CORPUS)
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_first_merge_is_the_most_frequent_pair():
    tok = BPETokenizer.train(["aaab aaab aaab"], 257)
    assert tok.token_bytes(256) == b"aa"


def test_merges_stay_inside_chunks():
    tok = BPETokenizer.train(["the cat"] * 50, 400)
    assert {tok.token_bytes(i) for i in tok.encode("the cat")} == {b"the", b" cat"}


def test_stops_when_nothing_repeats():
    tok = BPETokenizer.train(["abc"], 1000)  # every pair occurs once, so no merge is worth learning
    assert tok.merges == [] and tok.vocab_size == 256


def test_rejects_vocab_smaller_than_bytes():
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, 100)


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(CORPUS, 300)
    tok.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merges == tok.merges
    assert loaded.encode("the hat") == tok.encode("the hat")
