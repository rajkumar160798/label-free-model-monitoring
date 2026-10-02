"""NYC yellow taxi trips (2019-2021) as a monthly stream: predict a generous tip.

Task: for credit-card trips (cash tips are not recorded), does the tip reach
``tip_share`` of the fare? Features are known when the trip ends: zones,
distance, duration, time of day, fare and surcharges. ``total_amount`` is left
out because it includes the tip.

The main natural shift is COVID: ridership fell about 90% in April 2020 and the
mix of trips (zones, airports, times) changed for months. ``airport_fee`` only
exists from 2021, so earlier rows have it missing. Labels have no natural delay
here; they arrive ``label_delay_days`` after the trip.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import processed_dir, raw_dir
from .base import Stream, with_label_delay

NUMERIC = ["passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax", "tolls_amount",
           "improvement_surcharge", "congestion_surcharge", "airport_fee", "duration_min",
           "hour", "PULocationID", "DOLocationID"]
CATEGORICAL = ["VendorID", "RatecodeID", "store_and_fwd_flag", "weekday"]


def _read_month(path, per_month: int, seed: int) -> pl.DataFrame:
    year, month = map(int, path.stem.rsplit("_", 1)[1].split("-"))
    df = pl.read_parquet(path)
    if "airport_fee" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("airport_fee"))
    df = df.with_columns(
        pl.col("airport_fee").cast(pl.Float64),
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60)
        .alias("duration_min"),
    ).filter(
        (pl.col("payment_type") == 1)
        & (pl.col("tpep_pickup_datetime").dt.year() == year)
        & (pl.col("tpep_pickup_datetime").dt.month() == month)
        & (pl.col("tpep_dropoff_datetime").dt.month() == month)
        & (pl.col("fare_amount") > 0) & (pl.col("trip_distance") > 0)
        & pl.col("duration_min").is_between(1, 180)
    )
    if df.height > per_month:
        df = df.sample(per_month, seed=seed)
    return df.with_columns(
        pl.col("tpep_pickup_datetime").dt.hour().cast(pl.Float64).alias("hour"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("weekday"),
        (pl.col("tip_amount") / pl.col("fare_amount")).alias("tip_share"),
        pl.col("tpep_dropoff_datetime").alias("event_time"),
    ).select(["event_time", "tip_share", *NUMERIC, *CATEGORICAL])


def build_tlc_table(data_dir=None, per_month: int = 50_000, seed: int = 0) -> pl.DataFrame:
    files = sorted((raw_dir(data_dir) / "nyc_tlc" / "yellow").glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError("no yellow taxi parquet files; run `python -m lfmm.download tlc`")
    frames = []
    for f in files:
        print(f"  sampling {f.name}...")
        frames.append(_read_month(f, per_month, seed))
    return pl.concat(frames, how="vertical_relaxed").sort("event_time")


def load_tlc_tips(data_dir=None, tip_share: float = 0.25, label_delay_days: int = 30,
                  rebuild: bool = False) -> Stream:
    """Yellow-taxi generous-tip stream, 50k card trips per month."""
    cache = processed_dir(data_dir) / "nyc_tlc_yellow_sample.parquet"
    if rebuild or not cache.exists():
        build_tlc_table(data_dir).write_parquet(cache)
    df = pl.read_parquet(cache)

    X = df.select(NUMERIC + CATEGORICAL).to_pandas()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = (df["tip_share"] >= tip_share).cast(pl.Float64).to_numpy()
    event_time = df["event_time"].to_numpy().astype("datetime64[ns]")
    stream = Stream("nyc_tlc_tips", X, y, event_time, event_time,
                    meta={"label_delay": f"simulated {label_delay_days} days",
                          "target": f"tip >= {tip_share:.0%} of fare (card trips)"})
    return with_label_delay(stream, np.timedelta64(label_delay_days, "D"))
