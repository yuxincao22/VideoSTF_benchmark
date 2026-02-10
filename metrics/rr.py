from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np

from .base import BaseTextMetric, MetricResult


NgramWord = Tuple[str, ...]
NgramChar = str
NgramType = Union[NgramWord, NgramChar]

def max_ngram_repeats(text, n, by_word=True):
    """
    Compute the maximum repetition count of n-grams in a given text.

    Args:
        text: The input text.
        n: The n in the n-gram.
        by_word: If True, compute n-grams by splitting on whitespace (word-level).
                 If False, compute n-grams over characters (character-level).

    Returns:
        max_count: The maximum occurrence count of any n-gram.
        max_ngrams: A list of all n-grams whose occurrence count equals max_count.
                    For word-level, each n-gram is a tuple of strings.
                    For character-level, each n-gram is a string.
    """
    if by_word:
        tokens = text.split()
    else:
        tokens = list(text)

    if len(tokens) < n:
        return 0, []

    ngrams = []
    for i in range(len(tokens) - n + 1):
        gram = tokens[i : i + n]
        if by_word:
            gram = tuple(gram)
        else:
            gram = "".join(gram)
        ngrams.append(gram)

    counter = Counter(ngrams)
    max_count = max(counter.values())
    max_ngrams = [g for g, c in counter.items() if c == max_count]
    return max_count, max_ngrams


@dataclass
class RRResult(MetricResult):
    """
    value: mean RR over a batch, equals mean(1[max_count > threshold])
    per_sample: per sample RR indicators (length = batch size)
    max_counts: per sample max_count values
    max_ngrams: per sample max_ngrams values
    """
    value: float = 0.0
    per_sample: Optional[List[int]] = None
    max_counts: Optional[List[int]] = None
    max_ngrams: Optional[List[List[NgramType]]] = None


@dataclass
class RRMetric(BaseTextMetric):
    """
    RR: proportion of texts with max_count > threshold.

    Notes
    - For a single text, RR is an indicator in {0.0, 1.0}
    - For a batch, RR is the mean of indicators, matching the dataset average in the paper

    Parameters
    - n: n-gram size
    - by_word: True uses whitespace tokenization, False uses character level
    - threshold: default 1, checks max_count > threshold
    """

    n: int = 5
    by_word: bool = True
    threshold: int = 1

    def forward(self, text):
        max_count, _ = max_ngram_repeats(text, n=self.n, by_word=self.by_word)
        rr_value = 1.0 if max_count > self.threshold else 0.0
        return rr_value

    def forward_detail(self, text, return_per_sample=False):
        """
        Return (rr_value, max_count, max_ngrams) for a single text.

        max_ngrams is [] if max_count <= threshold.
        """
        max_count, max_ngrams = max_ngram_repeats(text, n=self.n, by_word=self.by_word)
        rr_value = 1.0 if max_count > self.threshold else 0.0
        max_ngrams = list(max_ngrams) if max_count > self.threshold else []
        return RRResult(
            value=rr_value,
            per_sample=[rr_value] if return_per_sample else None,
            max_counts=[max_count],
            max_ngrams=[max_ngrams],
        )

    def forward_batch(self, texts, return_per_sample=False):
        rr_list = []
        max_counts = []
        max_ngrams_list = []

        for t in texts:
            max_count, max_ngrams = max_ngram_repeats(t, n=self.n, by_word=self.by_word)
            max_counts.append(int(max_count))

            rr_value = 1.0 if max_count > self.threshold else 0.0
            rr_list.append(rr_value)

            if max_count > self.threshold:
                max_ngrams_list.append(list(max_ngrams))
            else:
                max_ngrams_list.append([])

        mean_value = float(np.mean(rr_list)) if len(rr_list) > 0 else 0.0

        return RRResult(
            value=mean_value,
            per_sample=rr_list if return_per_sample else None,
            max_counts=max_counts,
            max_ngrams=max_ngrams_list,
        )
