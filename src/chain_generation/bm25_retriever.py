"""Lightweight BM25 retriever for per-sub-question context focus.

Segments a flattened HotpotQA-style context ("[Title] sents | [Title] sents ...")
into paragraphs by [Title] markers, builds a BM25 index, and retrieves the top-k
paragraphs for a given sub-question. Pure Python (no external dependencies).

Rationale (Direction C / step B): the full HotpotQA context has ~10 paragraphs
(2 gold + distractors). Feeding all of them to every sub-question lets fluent
models latch onto a distracting paragraph and hallucinate an early entity. The
DAG decomposition tells us WHAT each sub-question needs, so we retrieve only the
top-k relevant paragraphs per sub-question, reducing distraction at the source.
"""
import re
import math
from collections import Counter
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


def segment_context(context_str: str) -> List[Tuple[str, str]]:
    """Split a flattened context into (title, text) paragraphs by [Title] markers.

    The preprocessing join is ``" | ".join(f"[{title}] " + " ".join(sentences))``,
    so a paragraph separator is `` | `` followed by ``[``. Splitting only on that
    pattern avoids breaking on pipes that may occur inside a sentence.
    """
    if not context_str:
        return []
    parts = re.split(r"\s*\|\s*(?=\[)", context_str)
    paras = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"\[([^\]]+)\]\s*(.*)", part, re.DOTALL)
        if m:
            title, text = m.group(1).strip(), m.group(2).strip()
        else:
            title, text = "", part
        if text:
            paras.append((title, text))
    return paras


class BM25Retriever:
    def __init__(self, context_str: str, k1: float = 1.5, b: float = 0.75):
        self.paras = segment_context(context_str)
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(t) for _, t in self.paras]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: max(0.0, math.log(1 + (self.N - n + 0.5) / (n + 0.5)))
            for w, n in df.items()
        }

    def retrieve_context(self, query: str, k: int = 2, max_chars: int = 1500) -> str:
        """Return a flattened context string of the top-k paragraphs for the query."""
        if not self.paras:
            return ""
        q = _tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            if not doc:
                scores.append(0.0)
                continue
            tf = Counter(doc)
            dl = len(doc)
            s = 0.0
            denom_base = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1.0))
            for w in q:
                f = tf.get(w, 0)
                if f == 0 or w not in self.idf:
                    continue
                s += self.idf[w] * (f * (self.k1 + 1)) / (f + denom_base)
            scores.append(s)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        chosen = [i for i in order if scores[i] > 0][:k]
        if not chosen:  # no token overlap; fall back to first k paragraphs
            chosen = order[:k]
        out = " | ".join(f"[{self.paras[i][0]}] {self.paras[i][1]}" for i in chosen)
        return out[:max_chars]
