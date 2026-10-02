"""Benchmark experiment registry shared by the task scripts.

Each experiment fixes a stream, the training date, the batch frequency, the
metrics to estimate, and the degradation events to detect:
``events = {metric: (deltas scored by AUROC, delta used for alarms)}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from .streams.acs import load_acs_income
from .streams.base import Stream
from .streams.brfss import load_brfss_diabetes
from .streams.freddie import load_freddie
from .streams.nyc_tlc import load_tlc_tips
from .streams.tabred import load_tabred

DEFAULT_EVENTS = {"roc_auc": ((0.01, 0.02, 0.05), 0.02),
                  "prevalence": ((0.01, 0.02, 0.05), 0.02)}


@dataclass
class Experiment:
    name: str
    load: Callable[[], Stream]
    train_end: str
    freq: str
    metrics: tuple[str, ...] = ("roc_auc", "prevalence")
    events: dict = field(default_factory=lambda: dict(DEFAULT_EVENTS))
    train_start: str | None = None
    note: str = ""

    def deploy_kwargs(self) -> dict:
        kw = {"train_end": self.train_end, "freq": self.freq}
        if self.train_start:
            kw["train_start"] = self.train_start
        return kw


EXPERIMENTS = {e.name: e for e in [
    Experiment(
        "freddie_2007", load_freddie, "2007-01-01", "Q",
        events={"roc_auc": ((0.02, 0.05, 0.08), 0.05), "prevalence": ((0.005, 0.01, 0.02), 0.01)},
        note="built from 1999-2004 loans; monitored through the crisis, COVID and the 2020s"),
    Experiment(
        "freddie_2016", load_freddie, "2016-01-01", "Q", train_start="2010-01-01",
        events={"roc_auc": ((0.02, 0.05, 0.08), 0.05), "prevalence": ((0.005, 0.01, 0.02), 0.01)},
        note="built from 2010-2013 loans; COVID is the main shock"),
    Experiment(
        "acs_income_2016", load_acs_income, "2016-01-01", "Y",
        metrics=("roc_auc", "accuracy", "prevalence"),
        events={"accuracy": ((0.01, 0.03, 0.05), 0.03), "prevalence": ((0.02, 0.05, 0.1), 0.05)},
        note="trained on 2014 survey; yearly to 2024"),
    Experiment(
        "brfss_2014", load_brfss_diabetes, "2014-01-01", "Q",
        note="trained on 2011-mid 2013 interviews; quarterly to 2025"),
    Experiment(
        "tlc_2019", load_tlc_tips, "2019-07-01", "M", train_start="2019-02-01",
        metrics=("roc_auc", "accuracy", "prevalence"),
        note="trained Feb-May 2019 (after the Feb 2019 tip-rate jump); monthly through COVID"),
    Experiment(
        "tabred_homecredit", partial(load_tabred, "homecredit-default"), "2019-09-01", "M",
        events={"roc_auc": ((0.01, 0.02, 0.05), 0.02), "prevalence": ((0.005, 0.01, 0.02), 0.01)}),
    Experiment("tabred_homesite", partial(load_tabred, "homesite-insurance"), "2013-10-01", "M"),
    Experiment("tabred_ecom", partial(load_tabred, "ecom-offers"), "2013-04-01", "W",
               note="only ~10 weeks of data: few deployment periods"),
]}
