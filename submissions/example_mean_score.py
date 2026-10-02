"""Example submission: a deliberately simple estimator.

Copy this file, rename the class and `name`, implement `estimate`, then run:

    uv run python scripts/evaluate_submission.py submissions/my_method.py:MyMethod \
        --author "Your Name" --description "One sentence" --url https://...

See CONTRIBUTING.md for the full process.
"""

import numpy as np

from lfmm.estimators import Estimator, History


class MeanScore(Estimator):
    """Estimate the positive rate as the average model score (no labels used)."""

    name = "example_mean_score"
    supports = ("prevalence",)  # metrics this method can estimate

    def estimate(self, metric: str, proba: np.ndarray, history: History) -> float:
        # proba: the frozen model's scores on the new batch.
        # history: earlier rows, with only the labels that have arrived by now.
        return float(proba.mean())
