from dataclasses import dataclass

from .base import BaseTextMetric


def calculate_rep_n(text, n):
    """
    rep n = 1 - unique_ngrams / (L - n + 1)
    where L is token length using whitespace split.
    """
    tokens = text.split()
    L = len(tokens)
    if L < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(L - n + 1)]
    unique_ngrams = len(set(ngrams))
    denom = (L - n + 1)
    rep_n = 1.0 - (unique_ngrams / denom)
    return float(rep_n)


@dataclass
class RIMetric(BaseTextMetric):
    """
    RI: repetition intensity using rep n.

    Parameters
    - n: n gram size
    """

    n: int = 1

    def forward(self, text):
        return float(calculate_rep_n(text, n=self.n))
