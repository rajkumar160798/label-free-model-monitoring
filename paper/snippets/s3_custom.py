import numpy as np
from s1_stream import stream
from lfmm.estimators import Estimator, History
from lfmm.harness import run_estimation

class RecentPositiveRate(Estimator):
    """Toy method: positive rate among labels that arrived in the last 90 days."""
    name = "recent_rate"
    supports = ("prevalence",)

    def estimate(self, metric: str, proba: np.ndarray, history: History) -> float:
        arrived = ~np.isnat(history.label_time)
        recent = arrived & (history.label_time > history.now - np.timedelta64(90, "D"))
        return float(np.nanmean(history.y_known[recent])) if recent.any() else np.nan

res = run_estimation(stream, "2023-01-01", freq="M", metrics=("prevalence",),
                     estimators=[RecentPositiveRate()])
print(res.summary())
