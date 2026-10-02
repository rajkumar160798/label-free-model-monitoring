# Progress

Where the project stands and what comes next. Read this together with [DATA.md](DATA.md) at the start of each session.

## Environment

- Python env lives **outside OneDrive** at `C:\venvs\lfmm` (OneDrive locked files in an in-project `.venv`). Set as user env vars: `UV_PROJECT_ENVIRONMENT=C:\venvs\lfmm`, `UV_LINK_MODE=copy`, `LFMM_DATA_DIR=C:\data\lfmm`.
- Run anything with `uv run ...`; tests with `uv run pytest -q`.

## Code layout

| Path | What it does |
|---|---|
| `src/lfmm/download/` | Downloaders and manual-file registration; writes `manifest.csv` |
| `src/lfmm/streams/base.py` | `Stream`: features, label, `event_time`, `label_time` (when the label arrives); `with_label_delay` for simulated delay |
| `src/lfmm/streams/freddie.py` | Freddie Mac SFLLD stream: 90+ days delinquent within 24 months; **natural** label arrival; cached at `processed/freddie_sflld_h24.parquet` |
| `src/lfmm/streams/acs.py` | ACSIncome stream, 2014–2024; RELP/RELSHIPP and ST/STATE harmonized; simulated 365-day delay; cached at `processed/acs_income.parquet` |
| `src/lfmm/estimators.py` | Estimators: reference, recent_arrivals, latest_complete_cohort, CBPE, ATC, DoC |
| `src/lfmm/harness.py` | `run_estimation`: train on fully labeled cohorts, freeze model, walk forward, score estimates against hidden truth |
| `scripts/run_estimation.py` | The three experiments below; writes `results/*_records.csv` and `*_summary.csv` |
| `src/lfmm/detectors.py` | Detectors: univariate tests, PSI, score KS, domain classifier, estimator-based alarms |
| `scripts/run_detection.py` | Detection experiments; writes `results/detection_*` |
| `tests/test_core.py` | Synthetic checks: expected-AUC formula, no label leakage, CBPE accurate under pure covariate shift |

## Design decisions

- **Training uses only cohorts whose labels have ≥95% arrived** by the training date. Partly labeled cohorts over-represent early-arriving labels (defaults), which biases the model.
- **Freddie Mac label timing:** a positive is known in the month of the first credit event; a negative is known at loan age 24 or at earlier prepayment. Loans with neither (data ends, rare terminations) never resolve (`y = NaN`). 4 modified loans with reset first-payment dates were dropped.
- **Truth for a period** is computed only when ≥90% of its labels ever resolve, so 2024Q2 onward has estimates but no truth: the real "labels haven't arrived" frontier.
- `accuracy` is not used for Freddie Mac: with ~1% defaults it just tracks the default rate.

## First results (2026-10-01)

Mean absolute error of each estimate against the true metric, per quarter (Freddie Mac) or year (ACS). Lower is better.

| Experiment | Metric | Best | Notes |
|---|---|---|---|
| freddie_2007 (69 quarters) | AUC | CBPE 0.038 | latest complete cohort 0.048, reference 0.056 |
| freddie_2007 | default rate | reference 0.012 | CBPE 0.015, misses the crisis entirely |
| freddie_2016 (33 quarters) | AUC | latest complete cohort 0.044 | CBPE 0.068, reference 0.102 |
| acs_income_2016 (8 years) | AUC | latest complete cohort 0.002 | CBPE 0.006 |
| acs_income_2016 | accuracy | latest complete cohort 0.009 | ATC 0.025, CBPE/DoC 0.029 |

**The headline pattern so far: the two kinds of estimator fail in opposite ways.**

- *Label-free* (CBPE) misses shocks that change outcomes but not features. 2008Q1 loans: true default rate 7.7%, CBPE 1.0%. COVID, 2019Q1 loans: true 4.4%, CBPE 0.4%.
- *Label-based* (latest complete cohort) sees the shock about 2 years late, then keeps reporting it after it ends. It reported 4.7% in 2010Q1 when the true rate was 0.8%, and 4.6% in 2021Q1 against a true 0.8%.
- Neither family handles both. That gap is the benchmark's core motivation.

## Degradation detection task (2026-10-01)

Code: `src/lfmm/detectors.py`, `run_detection` in `src/lfmm/harness.py`, `scripts/run_detection.py` (about 4.5 min, mostly the domain classifier). Tests: `tests/test_detectors.py`.

**Detectors:**
- Input and output drift tests:
  - `univariate_tests`: KS / chi-squared with Bonferroni correction; scored by the largest effect size (KS statistic or Cramér's V).
  - `psi_max`: largest PSI over features; alarms above 0.25.
  - `score_ks`: KS test on the model's scores.
  - `domain_classifier`: cross-validated AUC of telling reference rows from batch rows; alarms above 0.6.
- Estimator-based alarms (`<estimator>:<metric>`): alarm when the estimated metric is worse than the reference by more than delta.

**Event:** a period whose true metric is worse than the reference by more than delta. For default rate, a change in either direction counts. The baseline is the reference-window metric, which is in-time and therefore optimistic. In freddie_2016, 85% of quarters lose more than 0.05 AUC, so the event rate is reported next to precision and recall.

**Scores:**
- Spearman correlation between the detector's score and the true degradation.
- AUROC at several delta values.
- Alarm precision, recall and false-alarm rate at one delta.

| Experiment / event | Best at ranking | Alarms |
|---|---|---|
| freddie_2007, AUC drop > 0.05 | CBPE (ρ 0.56, AUROC 0.82) | CBPE almost never fires (recall 3%). Latest complete cohort: precision 0.70, recall 0.68. All drift tests fire every quarter. |
| freddie_2007, default rate change > 0.01 | nothing beats chance (best AUROC 0.54) | CBPE is *anti*-correlated (ρ −0.56): the crisis is invisible in the inputs |
| freddie_2016, AUC drop > 0.05 | PSI (ρ 0.50); domain classifier AUROC 0.91 | Latest complete cohort is the only usable alarm (precision 0.96, recall 0.86); drift tests fire every quarter |
| freddie_2016, default rate change > 0.01 | drift tests (ρ ≈ 0.3, AUROC ≈ 0.8) | still fire every quarter |
| acs_income_2016 (accuracy, prevalence) | every detector ranks perfectly | uninformative: only 8 periods, and drift grows steadily with time |

**Findings:**
- **Input-drift tests fire constantly on real data.** Alarm rate is 100% on Freddie Mac. The largest PSI comes from changes that aren't necessarily harmful: lender churn (`seller_name`), interest-rate levels (`orig_rate`), the "TPO" channel code disappearing after 2007, and `valuation_method` only existing from 2017. Their scores rank severity somewhat, but their alarm thresholds are useless.
- **CBPE ranks AUC drops well but its alarms almost never fire**, because its estimates move too little. Its threshold would need calibrating separately from its estimate.
- **Shifts in outcome rate with no change in inputs (2007–08, COVID) are invisible to every label-free method.** That is the opening for methods that use early-arriving labels.
- **ACS needs a harder setup** (states as shifts, monthly or state-level batches) before its detection results mean anything.

## Next steps

1. ~~**Drift detection task**~~ (done, see above). Still to do: thresholds calibrated on held-out reference periods rather than rule-of-thumb values, and MMD and ADWIN detectors.
2. **More estimators:** importance weighting (domain classifier), a prevalence/label-shift estimator (BBSE or EM), and methods that combine early partial labels with label-free estimates. Early-arriving defaults are informative if their bias is corrected.
3. **Retrain-or-not task:** cost = compute + accuracy lost, using the same walk-forward loop.
4. **More streams:** BRFSS (monthly, interview date), NYC TLC (tip prediction, COVID), TabReD (8 datasets, simulated delay).
5. Repeated seeds and confidence intervals before reporting any numbers in a paper.
6. Figures: estimate-vs-truth timelines per stream.
