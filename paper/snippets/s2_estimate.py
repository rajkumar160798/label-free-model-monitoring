from s1_stream import stream
from lfmm.estimators import CBPE, DelayAdjustedCBPE, LatestCompleteCohort
from lfmm.harness import run_estimation

# Train on fully labeled cohorts before 2023, freeze, walk forward monthly.
result = run_estimation(
    stream, train_end="2023-01-01", freq="M", metrics=("roc_auc", "prevalence"),
    estimators=[CBPE(), DelayAdjustedCBPE(), LatestCompleteCohort(freq="M")],
)
print(result.summary())                     # MAE and bias per estimator
print(result.bootstrap_mae("prevalence"))   # 95% block-bootstrap intervals
print(result.compare("prevalence", "da_cbpe", "cbpe"))
