"""TabReD (Rubachev et al.) binary-classification datasets as streams.

Uses the authors' preprocessed arrays (``raw/tabred/preprocessed/<name>.tabred``,
a gzipped tar). Rows are ordered by the timestamp in ``x_meta[:, 0]``, whose unit
differs per dataset. Labels have no natural delay in these files, so they
arrive ``label_delay_days`` after the event (defaults below are plausible for
each domain, not measured).

Only the three binary tasks are loaded; the five regression tasks need
regression support in the harness first.
"""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pandas as pd

from ..config import raw_dir
from .base import Stream, with_label_delay

DATASETS = {
    # name: (timestamp unit in x_meta[:, 0], default label delay in days, suggested batch freq)
    "homecredit-default": ("D", 90, "M"),   # loan default
    "homesite-insurance": ("us", 30, "M"),  # quote converts to a policy
    "ecom-offers": ("D", 14, "W"),          # shopper repeats after an offer
}


def load_tabred(name: str, data_dir=None, label_delay_days: int | None = None) -> Stream:
    if name not in DATASETS:
        raise ValueError(f"{name} is not a supported binary TabReD dataset: {sorted(DATASETS)}")
    unit, default_delay, freq = DATASETS[name]
    path = raw_dir(data_dir) / "tabred" / "preprocessed" / f"{name}.tabred"
    arrays = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            fname = m.name.split("/", 1)[-1]
            if m.isfile() and fname.endswith(".npy") and "/" not in fname:
                arrays[fname[:-4]] = np.load(io.BytesIO(t.extractfile(m).read()))
            elif fname == "info.json":
                info = json.load(t.extractfile(m))
    if info["task"]["type"] != "binclass":
        raise ValueError(f"{name} is {info['task']['type']}, not binclass")

    parts = []
    if "x_num" in arrays:
        parts.append(pd.DataFrame(arrays["x_num"], columns=[f"num_{i}" for i in range(arrays["x_num"].shape[1])]))
    if "x_bin" in arrays:
        parts.append(pd.DataFrame(arrays["x_bin"], columns=[f"bin_{i}" for i in range(arrays["x_bin"].shape[1])]))
    if "x_cat" in arrays:
        cat = pd.DataFrame(arrays["x_cat"], columns=[f"cat_{i}" for i in range(arrays["x_cat"].shape[1])])
        parts.append(cat.astype("category"))
    X = pd.concat(parts, axis=1)

    event_time = arrays["x_meta"][:, 0].astype(f"datetime64[{unit}]").astype("datetime64[ns]")
    order = np.argsort(event_time, kind="stable")
    stream = Stream(
        name=f"tabred_{name}",
        X=X.iloc[order],
        y=arrays["y"][order].astype(float),
        event_time=event_time[order],
        label_time=event_time[order],
        meta={"suggested_freq": freq, "source": "TabReD preprocessed (irubachev/tabred on Kaggle)"},
    )
    delay = default_delay if label_delay_days is None else label_delay_days
    stream.meta["label_delay"] = f"simulated {delay} days"
    return with_label_delay(stream, np.timedelta64(delay, "D"))
