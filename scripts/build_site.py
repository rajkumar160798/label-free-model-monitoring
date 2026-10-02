"""Build the static leaderboard website from leaderboard/ and figures/.

    uv run python scripts/build_leaderboard.py   # refresh leaderboard/ first
    uv run python scripts/build_site.py          # writes _site/ (open _site/index.html)

Deployed to GitHub Pages by .github/workflows/pages.yml on every push to master.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LB, FIG, OUT = ROOT / "leaderboard", ROOT / "figures", ROOT / "_site"
REPO = "https://github.com/rajkumar160798/label-free-model-monitoring"
sys.path.insert(0, str(Path(__file__).parent))
from build_leaderboard import _family  # noqa: E402  (same method grouping as the leaderboard)

NAMES = {
    "reference": "Reference (assume no change)", "recent_arrivals": "Recent arrivals",
    "latest_complete_cohort": "Latest complete cohort", "labels_to_date": "Labels to date",
    "cbpe": "CBPE", "atc": "ATC", "doc": "DoC",
    "da_cbpe": "Delay-adjusted CBPE", "da_cbpe_hz": "Delay-adjusted CBPE (hazard)",
    "univariate_tests": "Univariate KS / χ²", "psi_max": "PSI (max over features)",
    "score_ks": "KS on model scores", "domain_classifier": "Domain classifier", "mmd": "MMD",
    "adwin_scores": "ADWIN on scores", "adwin_errors": "ADWIN on arrived errors",
    "never": "Never retrain", "always": "Always retrain", "every_k (best k)": "Every k periods (best k)",
}
OURS = {"da_cbpe", "da_cbpe_hz"}
EXPERIMENT_NAMES = {
    "freddie_2007": "Freddie Mac 2007", "freddie_2016": "Freddie Mac 2016", "acs_income_2016": "ACS income",
    "brfss_2014": "BRFSS diabetes", "tlc_2019": "NYC taxi", "tabred_homecredit": "TabReD homecredit",
    "tabred_homesite": "TabReD homesite", "tabred_ecom": "TabReD ecom",
}
METRIC_NAMES = {"roc_auc": "AUC", "prevalence": "positive rate", "accuracy": "accuracy"}

TASKS = [
    {
        "key": "inflight", "title": "In-flight book estimation", "metric": "MAE", "lower": True, "digits": 4,
        "question": "What is the model's metric on everything it scored in the last 24 months, part of which is already labeled?",
        "note": "Freddie Mac only (natural label delay). This is where early-arriving labels pay off.",
    },
    {
        "key": "estimation", "title": "Next-batch estimation", "metric": "MAE", "lower": True, "digits": 4,
        "question": "What will the metric of the batch scored this period turn out to be, before its labels arrive?",
        "note": "8 experiments × their metrics. ATC and DoC estimate accuracy only, so they cover fewer cases.",
    },
    {
        "key": "detection", "title": "Degradation detection", "metric": "Spearman", "lower": False, "digits": 2,
        "question": "Has the model got worse than at deployment? Scored by the rank correlation between each detector's drift score and the true degradation.",
        "note": "16 cases: each experiment with a performance-drop event and a positive-rate-change event. A detector whose score never changes in a case scores 0 there.",
    },
    {
        "key": "retrain", "title": "Retrain or not", "metric": "regret", "lower": True, "digits": 3,
        "question": "Should the model be retrained now, given the labels that have arrived? Scored as regret against the hindsight-optimal schedule (retrain cost ρ = 1).",
        "note": "The best k for fixed schedules is chosen in hindsight, which flatters them.",
    },
]


def esc(s) -> str:
    return html.escape(str(s))


def base_name(method: str) -> str:
    """Leaderboard family name -> key in NAMES ('cbpe alarm' -> 'cbpe', 'retrain on psi_max' -> 'psi_max')."""
    m = method.removeprefix("retrain on ").removesuffix(" alarm")
    return m


def display(method: str, subs: dict) -> str:
    key = base_name(method)
    label = NAMES.get(key, key)
    if method.endswith(" alarm") and not method.startswith("retrain on"):
        label = f"{label} alarm"
    if method.startswith("retrain on"):
        label = f"Retrain on {label}" + (" alarm" if method.endswith(" alarm") else "")
    badges = ""
    if key in OURS:
        badges += '<span class="badge ours">ours</span>'
    if key in subs:
        s = subs[key]
        author = esc(s.get("author", ""))
        link = f'<a href="{esc(s["url"])}">{author}</a>' if s.get("url") else author
        badges += f'<span class="badge sub">submitted · {link}</span>'
    return f"{esc(label)}{badges}"


def case_label(case: str) -> str:
    exp, _, metric = case.partition("/")
    out = EXPERIMENT_NAMES.get(exp, exp)
    return f"{out} · {METRIC_NAMES.get(metric, metric)}" if metric else out


def leaderboard_table(task: dict, subs: dict) -> str:
    df = pd.read_csv(LB / f"{task['key']}.csv")
    total = int(df["cases"].max())
    d = task["digits"]
    rows = []
    for rank, r in enumerate(df.itertuples(), start=1):
        partial = r.cases < total
        cls = ' class="partial"' if partial else ""
        rows.append(
            f"<tr{cls}><td class='num'>{rank if not partial else '–'}</td><td>{display(r.method, subs)}</td>"
            f"<td class='num' data-v='{r.mean_rank:.4f}'>{r.mean_rank:.2f}</td><td class='num'>{int(r.wins)}</td>"
            f"<td class='num' data-v='{r.cases}'>{int(r.cases)}/{total}</td>"
            f"<td class='num' data-v='{r.mean_value:.6f}'>{r.mean_value:.{d}f}</td></tr>")
    better = "lower" if task["lower"] else "higher"
    return f"""
<div class="tablebox"><table class="sortable lb">
  <thead><tr><th class="num">#</th><th>Method</th><th class="num" title="Average rank across cases (1 = best)">Mean rank</th>
  <th class="num" title="Cases where the method ranked first">Wins</th><th class="num">Cases</th>
  <th class="num">Mean {esc(task['metric'])} ({better} is better)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>
<p class="caption">Methods evaluated on every case are ranked first; partial ones (–) follow.</p>"""


def detail_table(task: dict, subs: dict) -> str:
    cases = pd.read_csv(LB / f"cases_{task['key']}.csv").dropna(subset=["value"])
    cases["family"] = cases["method"].map(_family)
    agg = "min" if task["lower"] else "max"
    wide = cases.groupby(["family", "case"])["value"].agg(agg).unstack("case")
    order = pd.read_csv(LB / f"{task['key']}.csv")["method"].tolist()
    wide = wide.reindex([m for m in order if m in wide.index])
    best = wide.min() if task["lower"] else wide.max()
    d = task["digits"]
    head = "".join(f"<th class='num'>{esc(case_label(c))}</th>" for c in wide.columns)
    body = []
    for m, row in wide.iterrows():
        cells = []
        for c in wide.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("<td class='num muted'>–</td>")
            else:
                strong = abs(v - best[c]) < 1e-12
                cells.append(f"<td class='num{' best' if strong else ''}'>{v:.{d}f}</td>")
        body.append(f"<tr><td class='sticky'>{display(m, subs)}</td>{''.join(cells)}</tr>")
    return f"""
<details><summary>Scores per experiment</summary>
<div class="scroll"><table class="detail"><thead><tr><th class="sticky">Method</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<p class="caption">Best per column in bold.</p></details>"""


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--line:#e1e0d9;
--accent:#1f5fae;--accent-bg:#e8f0fb;--best-bg:#eef6ee;--ours:#0b6b3a;--ours-bg:#e5f3ea;--sub:#7a4b00;--sub-bg:#fbf1dc;
--code:#f1f0ec;--shadow:0 1px 2px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f5f5f3;--ink2:#c3c2b7;
--muted:#9a988f;--line:#2c2c2a;--accent:#7fb2f0;--accent-bg:#16243a;--best-bg:#17281b;--ours:#7fd6a4;--ours-bg:#14281c;
--sub:#f0c46a;--sub-bg:#2c2412;--code:#232321;--shadow:none}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f5f5f3;--ink2:#c3c2b7;--muted:#9a988f;--line:#2c2c2a;
--accent:#7fb2f0;--accent-bg:#16243a;--best-bg:#17281b;--ours:#7fd6a4;--ours-bg:#14281c;--sub:#f0c46a;--sub-bg:#2c2412;--code:#232321;--shadow:none}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;overflow-wrap:anywhere;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}a:hover{text-decoration-thickness:2px}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
header.top{border-bottom:1px solid var(--line);background:var(--surface)}
header.top .wrap{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding-top:12px;padding-bottom:12px}
.brand{font-weight:700;font-size:18px;color:var(--ink);text-decoration:none}
nav a{margin-left:16px;color:var(--ink2);text-decoration:none;font-size:15px}nav a:hover{color:var(--accent)}
.hero{padding-top:40px;padding-bottom:16px}.hero h1{font-size:clamp(26px,4vw,38px);line-height:1.2;margin:0 0 12px;letter-spacing:-.01em}
.hero p.lede{font-size:18px;color:var(--ink2);max-width:760px;margin:0 0 20px}
.cta{display:flex;gap:10px;flex-wrap:wrap}.btn{display:inline-block;padding:9px 16px;border-radius:8px;border:1px solid var(--line);
background:var(--surface);color:var(--ink);text-decoration:none;font-weight:600;font-size:15px;box-shadow:var(--shadow)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
:root[data-theme="dark"] .btn.primary{color:#0d0d0d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .btn.primary{color:#0d0d0d}}
code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;line-height:1.5}
p code,li code{background:var(--code);padding:1px 5px;border-radius:4px}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:24px 0 8px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:var(--shadow)}
.card .k{font-size:28px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}.card .t{color:var(--ink2);font-size:14.5px;margin-top:6px}
section{padding:28px 0;border-top:1px solid var(--line)}section h2{font-size:24px;margin:0 0 6px}
.q{color:var(--ink2);margin:0 0 4px;max-width:820px}.note{color:var(--muted);font-size:14px;margin:0 0 14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.tablebox{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface);box-shadow:var(--shadow)}table.lb{min-width:560px}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:13px;font-weight:600;color:var(--ink2);white-space:nowrap;background:var(--surface)}
table.sortable th{cursor:pointer;user-select:none}table.sortable th:hover{color:var(--accent)}
th.sorted-asc::after{content:" ▲";font-size:10px}th.sorted-desc::after{content:" ▼";font-size:10px}
td.num,th.num{text-align:right;white-space:nowrap;overflow-wrap:normal}tr:last-child td{border-bottom:none}
tr.partial td{color:var(--muted)}
.badge{display:inline-block;margin-left:8px;padding:0 7px;border-radius:999px;font-size:12px;font-weight:600;vertical-align:1px}
.badge.ours{background:var(--ours-bg);color:var(--ours)}.badge.sub{background:var(--sub-bg);color:var(--sub)}.badge.sub a{color:inherit}
.caption{font-size:13.5px;color:var(--muted);margin:8px 0 0}
details{margin-top:14px}summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:15px}
.scroll{overflow-x:auto;margin-top:10px;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table.detail{font-size:13.5px}table.detail th{white-space:normal;min-width:96px;vertical-align:bottom}
td.best{font-weight:700;background:var(--best-bg)}td.muted{color:var(--muted)}
.sticky{position:sticky;left:0;background:var(--surface);min-width:220px;z-index:1}
figure{margin:18px 0 0}figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fcfcfb}
figcaption{font-size:14px;color:var(--ink2);margin-top:8px}
.steps{counter-reset:s;list-style:none;padding:0;margin:0}.steps>li{counter-increment:s;position:relative;padding:0 0 18px 44px}
.steps>li::before{content:counter(s);position:absolute;left:0;top:0;width:30px;height:30px;border-radius:50%;
background:var(--accent-bg);color:var(--accent);font-weight:700;display:flex;align-items:center;justify-content:center}
footer{border-top:1px solid var(--line);padding:24px 0 40px;color:var(--muted);font-size:14px}
.theme{background:none;border:1px solid var(--line);border-radius:8px;color:var(--ink2);padding:4px 10px;cursor:pointer;margin-left:16px;font:inherit;font-size:14px}
@media (max-width:640px){nav a{margin-left:10px;font-size:14px}th,td{padding:8px}.hero{padding-top:28px}.sticky{min-width:150px}.hero p.lede{font-size:16.5px}.card .k{font-size:24px}}
"""

JS = """
(function(){
  const root=document.documentElement, key='lfmm-theme';
  try{const t=localStorage.getItem(key); if(t) root.dataset.theme=t;}catch(e){}
  const b=document.querySelector('.theme');
  if(b) b.addEventListener('click',()=>{
    const dark=root.dataset.theme? root.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme=dark?'light':'dark'; try{localStorage.setItem(key,root.dataset.theme);}catch(e){}
  });
  document.querySelectorAll('table.sortable').forEach(t=>{
    t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
      const body=t.tBodies[0], rows=[...body.rows], asc=!th.classList.contains('sorted-asc');
      t.querySelectorAll('th').forEach(h=>h.classList.remove('sorted-asc','sorted-desc'));
      th.classList.add(asc?'sorted-asc':'sorted-desc');
      const val=r=>{const c=r.cells[i]; const v=c.dataset.v ?? c.textContent.trim(); const n=parseFloat(v); return isNaN(n)?v.toLowerCase():n;};
      rows.sort((a,b)=>{const x=val(a),y=val(b); return (x>y?1:x<y?-1:0)*(asc?1:-1);});
      rows.forEach(r=>body.appendChild(r));
    }));
  });
})();
"""


def page(title: str, body: str, description: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="{esc(description)}">
<link rel="stylesheet" href="assets/style.css"></head>
<body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">lfmm leaderboard</a>
<nav><a href="index.html#leaderboards">Leaderboards</a><a href="index.html#findings">Findings</a><a href="submit.html">Submit</a>
<a href="{REPO}">GitHub</a><button class="theme" type="button" aria-label="Toggle dark mode">◐</button></nav></div></header>
{body}
<footer><div class="wrap">Built {date.today().isoformat()} from the results in the
<a href="{REPO}">repository</a>. MIT license. Install with <code>pip install lfmm</code>.</div></footer>
<script src="assets/app.js"></script></body></html>"""


def index(subs: dict) -> str:
    sections = []
    for task in TASKS:
        sections.append(f"""
<section id="{task['key']}"><div class="wrap">
<h2>{esc(task['title'])}</h2><p class="q">{esc(task['question'])}</p><p class="note">{esc(task['note'])}</p>
{leaderboard_table(task, subs)}
{detail_table(task, subs)}
</div></section>""")
    n_subs = len(subs)
    body = f"""
<div class="wrap hero">
<h1>Monitoring ML models before the labels arrive</h1>
<p class="lede">An open benchmark for performance estimation, drift detection and retraining decisions on
<strong>real, naturally drifting data</strong>, where each label becomes visible only when it would have arrived in
practice. A loan default, for example, is known up to two years after the loan is made.</p>
<div class="cta"><a class="btn primary" href="submit.html">Submit a method</a>
<a class="btn" href="{REPO}/blob/master/paper/main.pdf">Read the paper</a><a class="btn" href="https://pypi.org/project/lfmm/">pip install lfmm</a>
<a class="btn" href="{REPO}">GitHub</a></div>
<div class="findings" id="findings">
<div class="card"><div class="k">100%</div><div class="t">of periods flagged by every input-drift test (PSI, KS, MMD, domain classifier) in the median experiment: on real data, the inputs always change.</div></div>
<div class="card"><div class="k">2 years</div><div class="t">how late label-based monitoring saw the 2007–08 mortgage crisis; label-free estimators missed it entirely.</div></div>
<div class="card"><div class="k">31% lower</div><div class="t">default-rate error on the in-flight mortgage book with delay-adjusted CBPE (hazard) than the best label-based baseline, over 5 seeds.</div></div>
<div class="card"><div class="k">5 of 8</div><div class="t">experiments where a fixed retraining calendar beats retraining on monitor alarms.</div></div>
</div>
<p class="note">{len(TASKS)} tasks · 8 experiments on 5 public data sources · {n_subs} community submission{'s' if n_subs != 1 else ''}</p>
</div>
<section id="leaderboards" style="padding-bottom:0;border-top:none"><div class="wrap"><h2>Leaderboards</h2>
<p class="q">Within each case (an experiment and metric), methods are ranked; the tables give each method's mean rank, number of
first places and mean score. Click a column to sort.</p></div></section>
{''.join(sections)}
<section id="figures"><div class="wrap"><h2>What the methods see</h2>
<figure><img src="figures/freddie_2007_default_rate.png" alt="Freddie Mac default rate over 2007 to 2024: truth and four estimators, for new loans and the in-flight book" loading="lazy">
<figcaption>Freddie Mac, model built January 2007. Label-free CBPE misses the crisis; waiting for complete labels sees it two years late;
the delay-adjusted estimators react within quarters, and the hazard variant also handles COVID forbearance.</figcaption></figure>
<figure><img src="figures/detection_overview.png" alt="Detectors' median alarm rate against median correlation with true degradation" loading="lazy">
<figcaption>Drift tests alarm in every period; alarms from performance estimates are both more selective and better correlated with real degradation.</figcaption></figure>
</div></section>"""
    return page("lfmm leaderboard: monitoring models before the labels arrive", body,
                "Leaderboard of methods for estimating performance, detecting degradation and deciding when to retrain under label delay.")


def submit_page() -> str:
    body = f"""
<div class="wrap hero"><h1>Submit a method</h1>
<p class="lede">Any performance estimator or degradation detector can join the leaderboard. Your method is scored on the same
experiments, model and seed as the published baselines, so the comparison is like for like.</p></div>
<section style="border-top:none;padding-top:0"><div class="wrap">
<ol class="steps">
<li><strong>Set up.</strong> Clone the repository and install everything:
<pre>git clone {REPO}.git
cd label-free-model-monitoring
uv sync --all-extras      # or: pip install -e ".[all]"</pre></li>
<li><strong>Write your method</strong> as a class in <code>submissions/&lt;your_method&gt;.py</code>, starting from
<a href="{REPO}/blob/master/submissions/example_mean_score.py"><code>example_mean_score.py</code></a>. An estimator subclasses
<code>lfmm.estimators.Estimator</code> and implements <code>estimate(metric, proba, history)</code>; a detector subclasses
<code>lfmm.detectors.Detector</code> and implements <code>score(X, proba, history)</code>, returning a drift score and an alarm.
<pre>from lfmm.estimators import Estimator, History

class MyMethod(Estimator):
    name = "my_method"                       # unique, lowercase
    supports = ("prevalence", "roc_auc")     # metrics you can estimate

    def estimate(self, metric, proba, history: History) -> float:
        ...  # proba: the model's scores on the new batch
             # history: earlier rows, only labels that have already arrived</pre>
The harness never shows your method a label before it would have arrived; methods must not read data from anywhere else.</li>
<li><strong>Get the data</strong> you can (see <a href="{REPO}/blob/master/DATA.md">DATA.md</a>): <code>lfmm-download all</code>
fetches ACS, BRFSS and NYC taxi; Freddie Mac and TabReD need a free account and are registered with
<code>lfmm-download sflld</code> / <code>lfmm-download tabred</code>. Missing datasets are skipped.</li>
<li><strong>Evaluate:</strong>
<pre>uv run python scripts/evaluate_submission.py submissions/my_method.py:MyMethod \\
    --author "Your Name" --description "One sentence" --url https://link-to-paper-or-code</pre>
This writes <code>results/submissions/my_method/</code>. Running <code>scripts/build_leaderboard.py</code> shows where you rank.</li>
<li><strong>Open a pull request</strong> with your method file and its results folder. The maintainers re-run it on every experiment
(including any you could not download), and once merged it appears on this site automatically.
No pull request? <a href="{REPO}/issues/new?template=method_submission.yml">Open a submission issue</a> with a link to your code instead.</li>
</ol>
<p class="note">Full rules: <a href="{REPO}/blob/master/CONTRIBUTING.md">CONTRIBUTING.md</a>.</p>
</div></section>"""
    return page("Submit a method · lfmm leaderboard", body, "How to add a method to the lfmm leaderboard.")


def main() -> None:
    # Clear old files without removing the folders (sync tools such as OneDrive may hold them open).
    if OUT.exists():
        for f in sorted(OUT.rglob("*"), reverse=True):
            if f.is_file():
                f.unlink()
    (OUT / "assets").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    subs = {}
    sub_file = LB / "submissions.json"
    if sub_file.exists():
        subs = {s["name"]: s for s in json.loads(sub_file.read_text(encoding="utf-8"))}
    (OUT / "assets" / "style.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (OUT / "assets" / "app.js").write_text(JS.strip() + "\n", encoding="utf-8")
    for f in ("freddie_2007_default_rate.png", "detection_overview.png"):
        shutil.copy(FIG / f, OUT / "figures" / f)
    (OUT / "index.html").write_text(index(subs), encoding="utf-8")
    (OUT / "submit.html").write_text(submit_page(), encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print("wrote", OUT, sorted(p.name for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
