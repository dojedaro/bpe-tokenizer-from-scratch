"""Byte-level byte-pair encoding, written out in plain Python.

The vocabulary starts as the 256 possible bytes, so any string can be encoded and nothing is ever
unknown. Training repeatedly finds the most frequent adjacent pair of tokens and merges it into a new
token, until the vocabulary reaches the requested size. Encoding replays those merges in the order
they were learned.

Text is first split into chunks (a word with its leading space, a run of digits, a run of
punctuation, whitespace), and merges never cross a chunk boundary. That is the GPT-2 approach: it
keeps tokens like "ing" and " the" reusable instead of learning "the." and "the," separately.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

# A word with an optional leading space, a run of non-word symbols, or whitespace. Trailing
# whitespace before a word is left for the word, so " the" is one chunk.
CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|_+|\s+(?!\S)|\s+")

Pair = tuple[int, int]


def chunks(text: str) -> list[bytes]:
    return [m.encode("utf-8") for m in CHUNK.findall(text)]


@dataclass
class BPETokenizer:
    merges: list[Pair] = field(default_factory=list)  # in the order they were learned

    def __post_init__(self) -> None:
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        self._vocab = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def token_bytes(self, token_id: int) -> bytes:
        return self._vocab[token_id]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> BPETokenizer:
        """Learn ``vocab_size - 256`` merges from ``texts``.

        Works on unique chunks weighted by frequency, and keeps an index from each pair to the chunks
        that contain it, so a merge only revisits the chunks it changes. Ties between equally
        frequent pairs go to the smallest pair, which makes training deterministic.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        freq = Counter(c for t in texts for c in chunks(t))
        words = [list(c) for c in freq]
        counts = list(freq.values())

        pair_count: Counter[Pair] = Counter()
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            for p in pairwise(w):
                pair_count[p] += counts[i]
                where[p].add(i)

        # Max-heap of (-count, pair). Entries go stale as counts change; they are checked on pop.
        heap = [(-n, p) for p, n in pair_count.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        while 256 + len(merges) < vocab_size and heap:
            neg, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg:
                continue  # stale
            if -neg < 2:
                break  # nothing left that repeats
            new_id = 256 + len(merges)
            merges.append(pair)
            touched: Counter[Pair] = Counter()
            for i in where.pop(pair):
                w, n = words[i], counts[i]
                for p in pairwise(w):
                    pair_count[p] -= n
                    touched[p] -= n
                w = _merge(w, pair, new_id)
                words[i] = w
                for p in pairwise(w):
                    pair_count[p] += n
                    touched[p] += n
                    where[p].add(i)
            pair_count.pop(pair, None)
            # Re-queue every pair whose count changed; its old heap entry is now stale.
            for p, delta in touched.items():
                if delta and p != pair and pair_count[p] > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
        return cls(merges)

    # ------------------------------------------------------------------ encoding
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for c in chunks(text):
            if c not in self._cache:
                self._cache[c] = self._encode_chunk(c)
            ids.extend(self._cache[c])
        return ids

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        w = list(chunk)
        while len(w) > 1:
            # Apply the earliest-learned merge present, exactly as training would have.
            pair = min(pairwise(w), key=lambda p: self._rank.get(p, len(self._rank)))
            if pair not in self._rank:
                break
            w = _merge(w, pair, 256 + self._rank[pair])
        return w

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte-level-bpe", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])


def _merge(w: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``w``, left to right."""
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and w[i] == pair[0] and w[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out
