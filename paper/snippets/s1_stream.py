import numpy as np
import pandas as pd
from lfmm.streams.base import Stream, with_label_delay

# Any table of scored events becomes a stream: features, label,
# when each row is scored, and when its label becomes known.
rng = np.random.default_rng(0)
n = 24_000
month = np.repeat(np.arange(24), n // 24)
X = pd.DataFrame({"x1": rng.normal(month / 12, 1), "x2": rng.normal(0, 1, n)})
y = (rng.random(n) < 1 / (1 + np.exp(-(X.x1 - X.x2 - 1)))).astype(float)
t = (np.datetime64("2022-01", "M") + month).astype("datetime64[ns]")

stream = Stream("demo", X, y, event_time=t, label_time=t)
stream = with_label_delay(stream, np.timedelta64(60, "D"))  # labels 60 days late
print(stream.summary())
