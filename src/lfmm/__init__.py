"""lfmm: monitoring ML models before their labels arrive.

A benchmark and toolkit for performance estimation, drift detection and
retraining decisions on time-ordered streams with label delay.

Core pieces:

- :class:`lfmm.streams.base.Stream`: rows with an event time and a label arrival time;
- :mod:`lfmm.harness`: the walk-forward tasks (``run_estimation``, ``run_inflight``,
  ``run_detection_events``);
- :mod:`lfmm.estimators`, :mod:`lfmm.detectors`, :mod:`lfmm.retrain`: methods and policies.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lfmm")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
