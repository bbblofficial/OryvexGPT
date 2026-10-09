"""
Byte-level BPE tokenizer, written from scratch (no external tokenizer library).

Token id layout:
    0..5      special tokens
    6..261    the 256 raw bytes (so ANY text, in any language, can be encoded)
    262..     learned merges (BPE)
"""
from __future__ import annotations

import json
import re
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>"]
PAD, BOS, EOS, SYSTEM, USER, ASSISTANT = range(6)
BYTE_OFFSET = len(SPECIALS)          # id of byte 0x00
FIRST_MERGE_ID = BYTE_OFFSET + 256   # id of the first learned merge

_WORD = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+", re.UNICODE)
_SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._rebuild()

    # ------------------------------------------------------------------ setup
    def _rebuild(self):
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [b""] * BYTE_OFFSET + [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    # --------------------------------------------------------------- training
    def train(self, texts, vocab_size: int, verbose: bool = True):
        """Learn merges from an iterable of strings."""
        target = max(vocab_size, FIRST_MERGE_ID) - FIRST_MERGE_ID
        counts: Counter = Counter()
        for text in texts:
            counts.update(_WORD.findall(text))
        words = {tuple(BYTE_OFFSET + b for b in w.encode("utf-8")): f for w, f in counts.items()}

        merges: list[tuple[int, int]] = []
        while len(merges) < target:
            pairs: Counter = Counter()
            for word, freq in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += freq
            if not pairs:
                break
            best, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            new_id = FIRST_MERGE_ID + len(merges)
            merges.append(best)
            merged = {}
            for word, f in words.items():
                key = self._merge_word(word, best, new_id)
                merged[key] = merged.get(key, 0) + f
            words = merged
            if verbose and len(merges) % 250 == 0:
                print(f"  tokenizer: {len(merges)}/{target} merges")
        self.merges = merges
        self._rebuild()
        return self

    @staticmethod
    def _merge_word(word: tuple, pair: tuple, new_id: int) -> tuple:
        if len(word) < 2:
            return word
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ---------------------------------------------------------------- encode
    def _encode_word(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = [BYTE_OFFSET + b for b in word.encode("utf-8")]
        while len(ids) > 1:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = list(self._merge_word(tuple(ids), best_pair, FIRST_MERGE_ID + best_rank))
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, special: bool = True) -> list[int]:
        """Text -> ids. If `special`, literal '<eos>' etc. become special ids."""
        ids: list[int] = []
        for part in _SPECIAL_SPLIT.split(text) if special else [text]:
            if special and part in SPECIALS:
                ids.append(SPECIALS.index(part))
            elif part:
                for word in _WORD.findall(part):
                    ids.extend(self._encode_word(word))
        return ids

    # ---------------------------------------------------------------- decode
    def decode(self, ids, keep_special: bool = False) -> str:
        out = b""
        for i in ids:
            if i < BYTE_OFFSET:
                if keep_special:
                    out += SPECIALS[i].encode()
            else:
                out += self.token_bytes[i]
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save/load
    def to_dict(self) -> dict:
        return {"merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        return cls([tuple(m) for m in d["merges"]])

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ------------------------------------------------------------------- chat format
def encode_chat(tok: BPETokenizer, messages: list[dict], system: str | None = None,
                add_generation_prompt: bool = False):
    """
    Turn a conversation into (ids, loss_mask).

        <bos><|system|>..<|user|>..<|assistant|>..<eos><|user|>..<|assistant|>..<eos>

    loss_mask is 1 only on assistant text and its closing <eos>, so the model is
    trained to write replies rather than to imitate the user.
    """
    ids, mask = [BOS], [0]
    if system:
        seg = [SYSTEM] + tok.encode(system, special=False)
        ids += seg
        mask += [0] * len(seg)
    for m in messages:
        if m["role"] == "user":
            seg = [USER] + tok.encode(m["content"], special=False)
            ids += seg
            mask += [0] * len(seg)
        elif m["role"] == "assistant":
            body = tok.encode(m["content"], special=False)
            ids += [ASSISTANT] + body + [EOS]
            mask += [0] + [1] * len(body) + [1]
    if add_generation_prompt:
        ids.append(ASSISTANT)
        mask.append(0)
    return ids, mask
