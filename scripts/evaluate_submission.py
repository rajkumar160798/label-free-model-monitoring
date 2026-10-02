"""Evaluate a submitted method on the benchmark and save results for the leaderboard.

    uv run python scripts/evaluate_submission.py submissions/my_method.py:MyEstimator \
        --author "Jane Doe" --url https://github.com/jane/my-method --description "One line"

The class must subclass ``lfmm.estimators.Estimator`` or ``lfmm.detectors.Detector``
and take no constructor arguments (set defaults in ``__init__``). Estimators are
scored on next-batch estimation for every experiment (and on the in-flight book
for the natural-delay Freddie Mac experiments); detectors on every degradation
event. Each experiment uses the same training date, model and seed as the
published baselines, so results are directly comparable.

Results go to results/submissions/<method name>/ together with meta.json.
Experiments whose data are not available locally are skipped and listed; a
submission is merged once the maintainers have re-run it on all experiments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lfmm
from lfmm.detectors import Detector
from lfmm.estimators import Estimator
from lfmm.experiments import EXPERIMENTS
from lfmm.harness import deploy, run_detection_events, run_estimation, run_inflight

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DELAY = {"freddie_2007", "freddie_2016"}


def load_class(spec: str) -> type:
    path, _, cls_name = spec.partition(":")
    if not cls_name:
        raise SystemExit("method must be given as path/to/file.py:ClassName")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    cls = getattr(module, cls_name)
    if not (issubclass(cls, Estimator) or issubclass(cls, Detector)):
        raise SystemExit(f"{cls_name} must subclass lfmm.estimators.Estimator or lfmm.detectors.Detector")
    return cls


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", help="path/to/file.py:ClassName")
    p.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS), help="default: all")
    p.add_argument("--author", required=True)
    p.add_argument("--description", required=True, help="one sentence")
    p.add_argument("--url", default="", help="paper or code link")
    args = p.parse_args(argv)

    cls = load_class(args.method)
    probe = cls()
    kind = "estimator" if isinstance(probe, Estimator) else "detector"
    name = probe.name
    if not name or name in ("base",):
        raise SystemExit("set a unique class attribute `name` on your method")
    out = ROOT / "results" / "submissions" / name
    out.mkdir(parents=True, exist_ok=True)

    done, skipped = [], {}
    for exp_name in args.experiments:
        exp = EXPERIMENTS[exp_name]
        t0 = time.time()
        try:
            stream = exp.load()
        except (FileNotFoundError, RuntimeError, ImportError) as exc:  # data not available locally
            skipped[exp_name] = str(exc).splitlines()[0]
            print(f"skip {exp_name}: {skipped[exp_name]}")
            continue
        dep = deploy(stream, **exp.deploy_kwargs())
        if kind == "estimator":
            res = run_estimation(stream, exp.train_end, exp.freq, metrics=exp.metrics,
                                 estimators=[cls()], deployment=dep)
            res.summary().to_csv(out / f"estimation_{exp_name}_summary.csv")
            if exp_name in NATURAL_DELAY:
                book = run_inflight(stream, exp.train_end, exp.freq, metrics=("prevalence", "roc_auc"),
                                    estimators=[cls()], deployment=dep)
                book.summary().to_csv(out / f"inflight_{exp_name}_summary.csv")
        else:
            results = run_detection_events(stream, exp.train_end, [cls()], exp.events,
                                           freq=exp.freq, deployment=dep)
            for metric, res in results.items():
                res.summary().to_csv(out / f"detection_{exp_name}_{metric}_summary.csv")
        done.append(exp_name)
        print(f"{exp_name}: done in {time.time() - t0:.0f}s")

    meta = {
        "name": name, "kind": kind, "class": args.method, "author": args.author,
        "description": args.description, "url": args.url,
        "experiments": done, "skipped": skipped, "complete": not skipped,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lfmm_version": lfmm.__version__, "git_commit": git_commit(),
        "python": platform.python_version(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({len(done)} experiments, {len(skipped)} skipped)")
    if skipped:
        print("Missing data for some experiments: the maintainers will run those before merging.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
