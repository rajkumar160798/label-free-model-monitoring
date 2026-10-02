"""Time-ordered prediction streams with label arrival times.

A stream is a table of prediction events. Each row has:

- features ``X`` available when the prediction is made,
- ``event_time``: when the model scores the row,
- ``y``: the eventual label (NaN if it never resolves within the data),
- ``label_time``: when that label becomes known (NaT if never).

Monitoring methods may only use labels with ``label_time <= now``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass
class Stream:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    event_time: np.ndarray  # datetime64[ns]
    label_time: np.ndarray  # datetime64[ns], NaT = never arrives
    task: str = "binary"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.X)
        self.X = self.X.reset_index(drop=True)
        self.y = np.asarray(self.y, dtype=float)
        self.event_time = np.asarray(self.event_time, dtype="datetime64[ns]")
        self.label_time = np.asarray(self.label_time, dtype="datetime64[ns]")
        if not (len(self.y) == len(self.event_time) == len(self.label_time) == n):
            raise ValueError("X, y, event_time and label_time must have the same length")
        resolved = ~np.isnan(self.y)
        if np.any(resolved != ~np.isnat(self.label_time)):
            raise ValueError("label_time must be set exactly where y is known")
        if np.any(self.label_time[resolved] < self.event_time[resolved]):
            raise ValueError("a label cannot arrive before its event")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def cat_features(self) -> list[str]:
        return [c for c in self.X.columns if isinstance(self.X[c].dtype, pd.CategoricalDtype)]

    def periods(self, freq: str = "M") -> pd.PeriodIndex:
        return pd.PeriodIndex(pd.DatetimeIndex(self.event_time).to_period(freq))

    def subset(self, mask: np.ndarray) -> "Stream":
        mask = np.asarray(mask)
        return replace(
            self,
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            event_time=self.event_time[mask],
            label_time=self.label_time[mask],
        )

    def known_by(self, now: np.datetime64) -> np.ndarray:
        """Mask of rows whose label has arrived by ``now``."""
        return ~np.isnat(self.label_time) & (self.label_time <= np.datetime64(now, "ns"))

    def summary(self) -> pd.DataFrame:
        """Rows, label resolution and positive rate per year."""
        df = pd.DataFrame({
            "year": pd.DatetimeIndex(self.event_time).year,
            "y": self.y,
            "delay_days": (self.label_time - self.event_time) / np.timedelta64(1, "D"),
        })
        return df.groupby("year").agg(
            n=("y", "size"),
            resolved=("y", lambda s: s.notna().mean()),
            positive_rate=("y", "mean"),
            median_delay_days=("delay_days", "median"),
        )


def with_label_delay(stream: Stream, delay: np.timedelta64 | np.ndarray) -> Stream:
    """Replace label arrival times with event_time + delay.

    ``delay`` is a single timedelta or one per row. Use this for datasets with no
    natural label delay (ACS, TabReD, ...).
    """
    delay = np.asarray(delay, dtype="timedelta64[ns]")
    label_time = stream.event_time + delay
    label_time = np.where(np.isnan(stream.y), np.datetime64("NaT", "ns"), label_time)
    return replace(stream, label_time=label_time)
