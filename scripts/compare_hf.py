"""Train this BPE and Hugging Face's byte-level BPE on the same cleaned WikiText-2, then compare them.

    pip install -e ".[compare]"
    python scripts/compare_hf.py --vocab 8192

Both tokenizers are byte-level, so neither ever needs an unknown token. They pre-tokenize slightly
differently (this one with a stdlib regex, Hugging Face with GPT-2's), so their tokens are not
expected to be identical; the comparison is compression on held-out text, agreement on token
boundaries, and training time. Writes results/compare-<vocab>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bpe import BPETokenizer


def clean(texts: list[str]) -> list[str]:
    """The notebook's cleaning: drop empty lines, remove <unk>, normalize whitespace, deduplicate."""
    out, seen = [], set()
    for t in texts:
        t = re.sub(r" +", " ", t.replace("<unk>", "").replace("\t", " ")).strip()
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if t and h not in seen:
            seen.add(h)
            out.append(t)
    return out


def train_hf(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def byte_ends(lengths: list[int]) -> set[int]:
    """Byte offsets where tokens end, given each token's length in bytes."""
    ends, pos = set(), 0
    for n in lengths:
        pos += n
        ends.add(pos)
    return ends


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train = clean(ds["train"]["text"])
    held_out = clean(ds["validation"]["text"]) + clean(ds["test"]["text"])
    held_chars = sum(len(t) for t in held_out)

    t0 = time.perf_counter()
    ours = BPETokenizer.train(train, args.vocab)
    ours_train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hf = train_hf(train, args.vocab)
    hf_train_s = time.perf_counter() - t0

    ours_tokens = hf_tokens = agree = total = 0
    round_trip_ok = True
    for t in held_out:
        ids = ours.encode(t)
        round_trip_ok &= ours.decode(ids) == t
        enc = hf.encode(t)
        ours_tokens += len(ids)
        hf_tokens += len(enc.ids)
        # Compare in byte offsets: a byte-level token can end in the middle of a multi-byte character.
        a = byte_ends([len(ours.token_bytes(i)) for i in ids])
        char_to_byte = [0]
        for ch in t:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        b = {char_to_byte[end] for _, end in enc.offsets}
        agree += len(a & b)
        total += len(a | b)

    result = {
        "vocab_size": args.vocab,
        "train_documents": len(train),
        "held_out_documents": len(held_out),
        "held_out_chars": held_chars,
        "ours": {
            "merges": len(ours.merges),
            "train_seconds": round(ours_train_s, 1),
            "chars_per_token": round(held_chars / ours_tokens, 3),
            "round_trip_exact": round_trip_ok,
        },
        "huggingface": {"train_seconds": round(hf_train_s, 1), "chars_per_token": round(held_chars / hf_tokens, 3)},
        "token_boundary_agreement": round(agree / total, 3),
    }
    out = Path("results") / f"compare-{args.vocab}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
