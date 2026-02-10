from collections import Counter
from dataclasses import dataclass

import math

from .base import BaseTextMetric


def _calculate_entropy(probabilities):
    return float(-sum(p * math.log2(p) for p in probabilities if p > 0.0))


def _generate_ngrams(text, n):
    words = text.split()
    if len(words) < n:
        return []
    ngrams_iter = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams_iter]


def calculate_ngram_entropy(text, n):
    """
    Normalized entropy of n grams.

    normalized_entropy = H(p) / log2(len(all_ngrams))
    """
    ngrams = _generate_ngrams(text, n)
    if len(ngrams) == 0:
        return 0.0

    ngram_counts = Counter(ngrams)
    total = sum(ngram_counts.values())
    if total <= 0:
        return 0.0

    probabilities = [count / total for count in ngram_counts.values()]
    entropy = _calculate_entropy(probabilities)

    max_entropy = math.log2(len(ngrams)) if len(ngrams) > 0 else 1.0
    if max_entropy <= 0.0:
        return 0.0

    return float(entropy / max_entropy)


@dataclass
class IEMetric(BaseTextMetric):
    """
    IE: normalized n gram entropy.

    Parameters
    - n: n gram size
    """

    n: int = 1

    def forward(self, text):
        return float(calculate_ngram_entropy(text, n=self.n))
