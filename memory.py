"""Learned memory: exact recall of what the panel learned from Wikipedia.

A 4M-parameter model trained for a few thousand steps picks up the *style* of new text but cannot
memorise it word for word (you can see it in the garbled answers). So the panel keeps every learned
question/answer pair in data/learned.jsonl and, when a chat question matches one closely, answers with
the stored text. Anything that does not match goes to the neural model as before.
"""
from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path

STOP = {"the", "a", "an", "of", "is", "are", "was", "were", "what", "who", "or", "do", "does", "you", "me",
        "to", "and", "please", "tell", "about", "give", "i", "should", "know", "can", "could"}
THRESHOLD = 0.74   # weighted Dice overlap needed to count as the same question


def tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9").replace("\u200c", " ")
    t = re.sub(r"[\u064b-\u065f\u0670]", "", t)          # Arabic diacritics
    return [w for w in re.findall(r"\w+", t) if w not in STOP]


class LearnedMemory:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.items: list[tuple[str, str, Counter]] = []   # (question, answer, tokens)
        self.idf: dict[str, float] = {}
        self.load()

    def __len__(self):
        return len(self.items)

    def load(self):
        items, seen = [], set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        msgs = json.loads(line)["messages"]
                        q = next(m["content"] for m in msgs if m["role"] == "user")
                        a = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
                    except (ValueError, KeyError, StopIteration, TypeError):
                        continue
                    toks = tokens(q)
                    if not toks or not a.strip() or (q, a) in seen:
                        continue
                    seen.add((q, a))
                    items.append((q, a, Counter(toks)))
        df = Counter(t for _, _, c in items for t in c)
        n = max(1, len(items))
        idf = {t: math.log(1 + n / d) for t, d in df.items()}
        with self._lock:
            self.items, self.idf = items, idf

    def lookup(self, question: str):
        """Return (answer, score) for the best matching learned question, or None."""
        qt = Counter(tokens(question))
        if not qt:
            return None
        with self._lock:
            items, idf = self.items, self.idf
        default = math.log(1 + len(items))   # words never seen while learning count as rare
        w = lambda t: idf.get(t, default)
        qsum = sum(w(t) for t in qt)
        best, best_score = None, 0.0
        for q, a, ct in items:
            shared = sum(w(t) for t in qt.keys() & ct.keys())
            if not shared:
                continue
            score = 2 * shared / (qsum + sum(w(t) for t in ct))
            if score > best_score:
                best, best_score = a, score
                if score >= 0.999:
                    break
        return (best, best_score) if best is not None and best_score >= THRESHOLD else None
