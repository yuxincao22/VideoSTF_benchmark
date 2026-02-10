from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class MetricResult:
    value: float
    per_sample: Optional[List[float]] = None


class BaseTextMetric:
    """
    Base class for text metrics.

    Conventions
    - forward(text) computes a single text score
    - forward_batch(texts) computes dataset mean and optionally per sample scores
    - apply is an alias of forward or forward_batch depending on input type
    - __call__ delegates to apply
    """

    def forward(self, text):
        raise NotImplementedError

    def forward_batch(self, texts, return_per_sample=False):
        scores: List[float] = [float(self.forward(t)) for t in texts]
        mean_value = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return MetricResult(value=mean_value, per_sample=scores if return_per_sample else None)

    def apply(self, x, return_per_sample=False):
        if isinstance(x, str):
            return float(self.forward(x))
        return self.forward_batch(x, return_per_sample=return_per_sample)

    def __call__(self, x, return_per_sample= False):
        return self.apply(x, return_per_sample=return_per_sample)
