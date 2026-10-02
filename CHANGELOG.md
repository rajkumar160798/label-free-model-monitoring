# Changelog

## 0.1.0 (2026-10)

First release.

- Streams with label arrival times; loaders for Freddie Mac (natural label delay), ACS income, BRFSS, NYC taxi and three TabReD datasets; checksummed downloaders (`lfmm-download`).
- Tasks: next-batch estimation, in-flight book estimation, degradation detection, retrain-or-not with a hindsight-optimal oracle; block-bootstrap intervals.
- Estimators: reference, recent arrivals, latest complete cohort, labels to date, CBPE, ATC, DoC, delay-adjusted CBPE and its hazard variant.
- Detectors: univariate KS/chi-squared, PSI, KS on scores, domain classifier, MMD, ADWIN (scores and arrived errors), estimator alarms.
- Retraining policies: never, always, every k, on alarm.
