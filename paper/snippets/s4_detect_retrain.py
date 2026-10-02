from s1_stream import stream
from lfmm.detectors import PSI, EstimatorAlarm
from lfmm.estimators import DelayAdjustedCBPE
from lfmm.harness import run_detection_events
from lfmm.retrain import Always, Every, Never, OnAlarm, build_bank, evaluate_policies

# Detection: score each detector against "AUC dropped by more than delta".
events = {"roc_auc": ((0.01, 0.02), 0.02)}
detectors = [PSI(), EstimatorAlarm(DelayAdjustedCBPE(), "roc_auc", 0.02)]
results = run_detection_events(stream, "2023-01-01", detectors, events, freq="M")
print(results["roc_auc"].summary()[["spearman", "precision", "recall", "alarm_rate"]])

# Retrain or not: one candidate model per month, then any policy by lookup.
bank = build_bank(stream, "2023-01-01", "M", verbose=False)
regret = evaluate_policies(bank, [Never(), Always(), Every(3), OnAlarm(PSI)])
print(regret[regret.rho == 1.0][["policy", "retrains", "regret"]])
