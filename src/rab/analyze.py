"""Analysis over the continuation files (design v2 D, G; guide §8).

Pure: reads only files and writes ``tables.md`` (byte-for-byte deterministic: sorted
keys, fixed float formats, no timestamps), ``tables_meta.json`` (generation metadata),
``hand_read.md`` and ``hand_read_key.json``. No Docker, no keys, no model call.

Inputs under ``--root`` (``results`` for the real run, ``results/dry`` for the dry run):
``continuations/<event_id>/<arm>-<run>.json.gz`` (one slot each, written by
``rab.run``), ``sims.jsonl``, ``leaks.jsonl``, ``run_log.jsonl``,
``discriminator.jsonl`` (all optional but the continuations), plus
``data/events.jsonl`` and ``data/events_recount.json`` for the event facts and the
replay exclusions.

Statistics, per cell (solvable, unreachable), every arm against R:

* **Primary**: total variation distance between the pooled-marginal class
  distributions (counts summed over the cell's events, errors excluded) of SA and R.
  p-value: within-event permutation of arm labels (:data:`N_PERM`, seeded); interval:
  event-level bootstrap (:data:`N_BOOT`, percentile 95%). Holm across the two cells.
* **Sign flip**: per task, the mean over its events of S − R in the leading class
  share (the class most common in R in that cell); exact two-sided sign-flip test.
* **Equivalence**: per class share, S − R with its bootstrap 95% interval; "within ±10"
  only when the interval lies inside ±0.10. Never claimed on TVD.
* **Sensitivity**: equal-n (SA against R runs 4–7, beside R 0–3 against R 4–7) and the
  primary without runs where ``size_visible_post_block``.
* 2×2 flagged × fitted, reread / named rates, regrade pass with Wilson intervals, the
  moderators (``notes_reveal_T``, ``fake_vs_reference``, ``fake_encodes_T``), drops, cost.

Exploratory slot files (``"exploratory": true``, file names ending ``-x``: the pilot,
prompt iteration) are excluded from ``tables.md`` by default and listed in a short
"Exploratory runs (excluded)" section with their count and git hashes;
``--include-exploratory`` puts them in. Nothing here depends on a preregistration or a
date: the analysis always runs. ``tables_meta.json`` records the git state.

``--placebo`` relabels R runs 4–7 as SA (R keeps runs 0–3; the real SA runs are set
aside) and runs everything; it should declare "moved" in about one cell in twenty.

:func:`verdict` chooses the README paragraph by rule, per cell (design v2 D):
``control failed`` (R − P regrade gap on unreachable under 30 points, or TVD(B, R)
inside the permutation null in both cells), else ``positive`` (Holm-adjusted primary
p < 0.05), else ``bounded null`` (every class share's S − R interval inside ±10
points), else ``inconclusive``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rab import REPO_ROOT
from rab.constants import CLASSES, KINDS, SEED

RESULTS = REPO_ROOT / "results"
EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"
RECOUNT_PATH = REPO_ROOT / "data" / "events_recount.json"

N_PERM = 10_000
N_BOOT = 1_000
ALPHA = 0.05
EQUIV_MARGIN = 0.10
CONTROL_GAP = 0.30
ARM_ORDER = ("R", "SA", "SSTAR", "W", "P", "B")
COMPARED = ("SA", "SSTAR", "W", "P", "B")
CLASSES_NO_ERROR = tuple(c for c in CLASSES if c != "error")
HAND_READ_PAIRS = 5


# ------------------------------------------------------------------ loading

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_slot(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def slot_files(root: Path) -> list[Path]:
    return sorted((Path(root) / "continuations").glob("*/*.json.gz"),
                  key=lambda p: (p.parent.name, p.name))


def run_row(slot: dict, event: dict | None, file: str = "") -> dict:
    """One flat row per continuation: what every statistic below reads."""
    cls, cont, placed = slot.get("cls") or {}, slot.get("cont") or {}, slot.get("placed") or {}
    agreement = placed.get("fake_agreement") or {}
    err = str(cont.get("error") or "")
    return {
        "event_id": slot["event_id"], "arm": slot["arm"], "run_idx": int(slot["run_idx"]),
        "kind": (event or {}).get("kind") or slot.get("kind"),
        "task_id": (event or {}).get("task_id") or slot.get("task_id"),
        "klass": cls.get("klass", "error"),
        "flagged": bool(cls.get("flagged_problem")), "fitted": bool(cls.get("fitted")),
        "reread": bool(cls.get("reread")), "named": bool(cls.get("named")),
        "regrade_pass": bool(cls.get("regrade_pass")),
        "size_visible": bool(cls.get("size_visible_post_block")),
        "error_class": cont.get("error_class"),
        "provider_mismatch": "ProviderPinError" in err or bool(slot.get("provider_mismatch")),
        "cost_usd": float(cont.get("cost_usd") or 0.0),
        "wall_s": float(cont.get("wall_s") or 0.0),
        "fake_vs_reference": agreement.get("vs_reference"),
        "fake_encodes_T": agreement.get("encodes_T"),
        "notes_reveal_T": bool((event or {}).get("notes_reveal_T")),
        "exploratory": bool(slot.get("exploratory")),
        "git_hash": (slot.get("git") or {}).get("git_hash"),
        "file": file,
    }


def load(root: Path, events_path: Path = EVENTS_PATH) -> tuple[list[dict], dict[str, dict]]:
    """``(rows, events)``: one row per slot file, exploratory ones included (flagged)."""
    events = {e["event_id"]: e for e in read_jsonl(events_path)}
    rows = []
    for path in slot_files(root):
        s = load_slot(path)
        rows.append(run_row(s, events.get(s["event_id"]), path.name))
    used = {r["event_id"] for r in rows}
    return rows, {k: v for k, v in events.items() if k in used}


def split_exploratory(rows: list[dict], include: bool) -> tuple[list[dict], list[dict]]:
    """``(rows analysed, exploratory rows)``; exploratory rows are analysed only if ``include``."""
    explo = [r for r in rows if r["exploratory"]]
    return (rows if include else [r for r in rows if not r["exploratory"]]), explo


def placebo(rows: list[dict]) -> list[dict]:
    """R runs 4–7 become SA runs 0–3; the real SA runs are set aside; other arms stay."""
    out = []
    for r in rows:
        if r["arm"] == "SA":
            continue
        if r["arm"] == "R" and r["run_idx"] >= 4:
            r = dict(r, arm="SA", run_idx=r["run_idx"] - 4)
        out.append(r)
    return out


# ------------------------------------------------------------------ statistics

def class_counts(rows: list[dict]) -> np.ndarray:
    c = Counter(r["klass"] for r in rows if r["klass"] != "error")
    return np.array([c.get(k, 0) for k in CLASSES_NO_ERROR], dtype=float)


def shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, counts / np.where(total > 0, total, 1), np.nan)


def tvd(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Total variation distance between count vectors (last axis = classes)."""
    return 0.5 * np.abs(shares(ca) - shares(cb)).sum(axis=-1)


def per_event_counts(rows: list[dict], arm_a: str, arm_b: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Events having both arms (errors excluded), with each arm's class counts."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] in (arm_a, arm_b) and r["klass"] != "error":
            by[r["event_id"]][r["arm"]].append(r)
    ids = sorted(e for e, d in by.items() if d[arm_a] and d[arm_b])
    a = np.array([class_counts(by[e][arm_a]) for e in ids]).reshape(len(ids), len(CLASSES_NO_ERROR))
    b = np.array([class_counts(by[e][arm_b]) for e in ids]).reshape(len(ids), len(CLASSES_NO_ERROR))
    return ids, a, b


def permutation_p(ca: np.ndarray, cb: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Within-event permutation of arm labels; p = (1 + #{TVD_perm >= TVD_obs}) / (1 + n)."""
    if len(ca) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    k = ca.shape[1]
    observed = float(tvd(ca.sum(0), cb.sum(0)))
    tot_a = np.zeros((n_perm, k))
    tot_b = np.zeros((n_perm, k))
    for ea, eb in zip(ca, cb):
        labels = np.repeat(np.arange(k), (ea + eb).astype(int))
        na = int(ea.sum())
        if na == 0 or na == len(labels):
            tot_a += ea
            tot_b += eb
            continue
        perm = rng.permuted(np.tile(labels, (n_perm, 1)), axis=1)
        onehot = np.eye(k)[perm]
        part_a = onehot[:, :na].sum(1)
        tot_a += part_a
        tot_b += onehot.sum(1) - part_a
    null = tvd(tot_a, tot_b)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + n_perm))


def bootstrap(ca: np.ndarray, cb: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Event-level bootstrap percentile 95% interval of ``stat(sum_a, sum_b)``."""
    if len(ca) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
    vals = np.asarray(stat(ca[idx].sum(1), cb[idx].sum(1)), dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5], axis=0)
    return (float(lo[0]), float(hi[0])) if lo.shape == (1,) else (lo, hi)


def share_diff_intervals(ca: np.ndarray, cb: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Per class: pooled share A − B and its event-bootstrap 95% interval."""
    if len(ca) == 0:
        nan = np.full(len(CLASSES_NO_ERROR), np.nan)
        return nan, nan, nan
    diff = shares(ca.sum(0)) - shares(cb.sum(0))
    rng = np.random.default_rng(seed + 2)
    idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
    boot = shares(ca[idx].sum(1)) - shares(cb[idx].sum(1))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return diff, lo, hi


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (NaN entries stay NaN and are not counted)."""
    items = sorted(((p, k) for k, p in pvals.items() if not math.isnan(p)))
    m, out, running = len(items), {k: float("nan") for k in pvals}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def sign_flip(task_diffs: dict[str, float]) -> float:
    """Exact two-sided sign-flip test on the mean of per-task differences."""
    vals = np.array([v for _, v in sorted(task_diffs.items()) if not math.isnan(v)])
    if len(vals) == 0:
        return float("nan")
    obs = abs(vals.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=len(vals))))
    return float(np.mean(np.abs((signs * vals).mean(1)) >= obs - 1e-12))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def leading_class(rows: list[dict]) -> str | None:
    c = class_counts([r for r in rows if r["arm"] == "R"])
    return None if c.sum() == 0 else CLASSES_NO_ERROR[int(np.argmax(c))]


def task_diffs(rows: list[dict], arm: str, klass: str) -> dict[str, float]:
    """Per task: mean over its events of (arm share − R share) of ``klass``."""
    j = CLASSES_NO_ERROR.index(klass)
    ids, a, b = per_event_counts(rows, arm, "R")
    task_of = {r["event_id"]: r["task_id"] for r in rows}
    per_task = defaultdict(list)
    for e, ea, eb in zip(ids, a, b):
        per_task[task_of[e]].append(shares(ea)[j] - shares(eb)[j])
    return {t: float(np.mean(v)) for t, v in per_task.items()}


def compare(rows: list[dict], arm: str, base: str = "R", n_perm: int = N_PERM) -> dict:
    """``arm`` against ``base`` in one cell: TVD, permutation p, bootstrap interval, shares."""
    ids, a, b = per_event_counts(rows, arm, base)
    out = {"arm": arm, "base": base, "events": len(ids),
           "runs_arm": int(a.sum()), "runs_base": int(b.sum())}
    if not ids:
        return {**out, "tvd": float("nan"), "p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "diff": [float("nan")] * len(CLASSES_NO_ERROR), "diff_lo": [float("nan")] * len(CLASSES_NO_ERROR),
                "diff_hi": [float("nan")] * len(CLASSES_NO_ERROR)}
    lo, hi = bootstrap(a, b, tvd)
    diff, dlo, dhi = share_diff_intervals(a, b)
    return {**out, "tvd": float(tvd(a.sum(0), b.sum(0))), "p": permutation_p(a, b, n_perm),
            "lo": lo, "hi": hi, "diff": [float(x) for x in diff],
            "diff_lo": [float(x) for x in dlo], "diff_hi": [float(x) for x in dhi]}


def equivalent(cmp: dict) -> bool:
    lo, hi = cmp["diff_lo"], cmp["diff_hi"]
    return bool(lo) and all(not math.isnan(l) and l > -EQUIV_MARGIN and h < EQUIV_MARGIN
                            for l, h in zip(lo, hi))


def cell_analysis(rows: list[dict], n_perm: int = N_PERM) -> dict:
    """Everything one cell needs; ``rows`` are that cell's runs."""
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
    out: dict = {"arms": arms, "events": len({r["event_id"] for r in rows}),
                 "tasks": len({r["task_id"] for r in rows})}
    out["distribution"] = {a: class_dist(rows, a) for a in arms}
    out["compare"] = {a: compare(rows, a, n_perm=n_perm) for a in arms if a in COMPARED}
    lead = leading_class(rows)
    out["leading_class"] = lead
    out["sign_flip"] = ({"diffs": task_diffs(rows, "SA", lead),
                         "p": sign_flip(task_diffs(rows, "SA", lead))} if lead and "SA" in arms else None)
    r_lo = [r for r in rows if not (r["arm"] == "R" and r["run_idx"] >= 4)]
    r_hi = [dict(r, arm="RHI") for r in rows if r["arm"] == "R" and r["run_idx"] >= 4]
    out["equal_n"] = {
        "SA_vs_R4_7": compare([r for r in rows if r["arm"] == "SA"] + [dict(r, arm="R") for r in r_hi],
                              "SA", n_perm=n_perm),
        "R0_3_vs_R4_7": compare([r for r in r_lo if r["arm"] == "R"] + r_hi, "RHI", n_perm=n_perm),
    }
    visible = [r for r in rows if not r["size_visible"]]
    out["no_size_visible"] = compare(visible, "SA", n_perm=n_perm) if "SA" in arms else None
    out["two_by_two"] = {a: two_by_two(rows, a) for a in arms}
    out["rates"] = {a: rates(rows, a) for a in arms}
    out["moderators"] = moderators(rows, n_perm)
    return out


def class_dist(rows: list[dict], arm: str) -> dict:
    mine = [r for r in rows if r["arm"] == arm]
    c = Counter(r["klass"] for r in mine)
    n_ok = sum(v for k, v in c.items() if k != "error")
    return {"n": len(mine), "n_ok": n_ok, "events": len({r["event_id"] for r in mine}),
            "counts": {k: c.get(k, 0) for k in CLASSES},
            "shares": {k: (c.get(k, 0) / n_ok if n_ok else float("nan")) for k in CLASSES_NO_ERROR}}


def two_by_two(rows: list[dict], arm: str) -> dict:
    mine = [r for r in rows if r["arm"] == arm and r["klass"] != "error"]
    return {f"flagged={f}/fitted={t}": sum(1 for r in mine if r["flagged"] == f and r["fitted"] == t)
            for f in (True, False) for t in (True, False)}


def rates(rows: list[dict], arm: str) -> dict:
    mine = [r for r in rows if r["arm"] == arm and r["klass"] != "error"]
    n = len(mine)
    out = {"n": n}
    for key in ("reread", "named", "regrade_pass"):
        k = sum(r[key] for r in mine)
        lo, hi = wilson(k, n)
        out[key] = {"k": k, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}
    return out


def moderators(rows: list[dict], n_perm: int) -> dict:
    """SA against R within strata of the pre-treatment and fake-quality moderators."""
    sa_q = defaultdict(list)
    for r in rows:
        if r["arm"] == "SA":
            for key in ("fake_vs_reference", "fake_encodes_T"):
                if r[key] is not None:
                    sa_q[(r["event_id"], key)].append(r[key])
    strata = {
        "notes_reveal_T": lambda e: next((r["notes_reveal_T"] for r in rows if r["event_id"] == e), None),
        "fake_vs_reference>=0.5": lambda e: (np.mean(sa_q[(e, "fake_vs_reference")]) >= 0.5
                                             if sa_q.get((e, "fake_vs_reference")) else None),
        "fake_encodes_T>=0.5": lambda e: (np.mean(sa_q[(e, "fake_encodes_T")]) >= 0.5
                                          if sa_q.get((e, "fake_encodes_T")) else None),
    }
    out = {}
    events = sorted({r["event_id"] for r in rows})
    for name, fn in strata.items():
        level = {e: fn(e) for e in events}
        for val in (True, False):
            keep = {e for e, v in level.items() if v is not None and bool(v) == val}
            sub = [r for r in rows if r["event_id"] in keep]
            c = compare(sub, "SA", n_perm=n_perm) if sub else None
            out[f"{name}={val}"] = {"events": len(keep), "tvd": c["tvd"] if c else float("nan"),
                                     "p": c["p"] if c else float("nan")}
    return out


def analyse(rows: list[dict], n_perm: int = N_PERM) -> dict:
    cells = {kind: cell_analysis([r for r in rows if r["kind"] == kind], n_perm)
             for kind in KINDS if any(r["kind"] == kind for r in rows)}
    adj = holm({kind: c["compare"].get("SA", {}).get("p", float("nan")) for kind, c in cells.items()})
    for kind, c in cells.items():
        c["primary_p_holm"] = adj.get(kind, float("nan"))
    return {"cells": cells}


def control(analysis: dict) -> dict:
    """Design v2 D: R − P regrade gap on unreachable, and TVD(B, R) against its null."""
    cells = analysis["cells"]
    gap = float("nan")
    un = cells.get("unreachable")
    if un and "P" in un["rates"] and "R" in un["rates"]:
        gap = un["rates"]["R"]["regrade_pass"]["rate"] - un["rates"]["P"]["regrade_pass"]["rate"]
    b_outside = {k: (c["compare"].get("B", {}).get("p", float("nan")) < ALPHA) for k, c in cells.items()}
    failed = (math.isnan(gap) or gap < CONTROL_GAP) or not any(b_outside.values())
    return {"regrade_gap_R_minus_P": gap, "B_outside_null": b_outside, "failed": bool(failed)}


def verdict(analysis: dict) -> dict[str, str]:
    """Per cell, the README paragraph that applies (design v2 D), first rule wins."""
    ctl = control(analysis)
    out = {}
    for kind, c in analysis["cells"].items():
        sa = c["compare"].get("SA")
        if ctl["failed"]:
            out[kind] = "control failed"
        elif sa and not math.isnan(c["primary_p_holm"]) and c["primary_p_holm"] < ALPHA:
            out[kind] = "positive"
        elif sa and equivalent(sa):
            out[kind] = "bounded null"
        else:
            out[kind] = "inconclusive"
    return out


# ------------------------------------------------------------------ drops and cost

def drops(root: Path, rows: list[dict], recount_path: Path = RECOUNT_PATH) -> dict:
    root = Path(root)
    by_kind = lambda pred: {k: sum(1 for r in rows if r["kind"] == k and pred(r)) for k in KINDS}
    recount = json.loads(recount_path.read_text(encoding="utf-8")) if recount_path.is_file() else {}
    replay_drops = {k: len(v) for k, v in (recount.get("replay_drops") or {}).items()}
    leaks = read_jsonl(root / "leaks.jsonl")
    sims = read_jsonl(root / "sims.jsonl")
    log = read_jsonl(root / "run_log.jsonl")
    sim_fail = Counter((s.get("family"), s.get("failure")) for s in sims if not s.get("ok"))
    event_drops = Counter(l.get("drop") for l in log if l.get("drop"))
    return {
        "replay exclusions (before the draw)": {k: replay_drops.get(k, 0) for k in KINDS},
        "leak drops (events)": {k: sum(1 for l in leaks if l.get("kind") == k and l.get("severity") == "drop")
                                for k in KINDS},
        "simulator failures (draws)": {f"{fam} {fail}": n for (fam, fail), n in sorted(
            sim_fail.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))},
        "event drops in the run": dict(sorted(event_drops.items())),
        "refusals (runs)": by_kind(lambda r: r["klass"] == "refusal"),
        "errors (runs)": by_kind(lambda r: r["klass"] == "error"),
        "provider mismatches (runs)": by_kind(lambda r: r["provider_mismatch"]),
    }


def cost(root: Path, rows: list[dict]) -> dict:
    sims = read_jsonl(Path(root) / "sims.jsonl")
    disc = read_jsonl(Path(root) / "discriminator.jsonl")
    by_arm = defaultdict(float)
    for r in rows:
        by_arm[r["arm"]] += r["cost_usd"]
    tok = defaultdict(lambda: [0, 0])
    for s in sims:
        u = s.get("usage") or {}
        tok[s.get("family")][0] += int(u.get("input_tokens") or 0)
        tok[s.get("family")][1] += int(u.get("output_tokens") or 0)
    return {"agent_usd_by_arm": {a: by_arm[a] for a in ARM_ORDER if a in by_arm},
            "agent_usd_total": sum(by_arm.values()),
            "simulator_tokens": {k: v for k, v in sorted(tok.items(), key=lambda x: str(x[0]))},
            "discriminator_calls": len(disc)}


# ------------------------------------------------------------------ rendering

def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def interval(lo, hi, scale=pct) -> str:
    return f"[{scale(lo)}, {scale(hi)}]"


def render(analysis: dict, drops_: dict, cost_: dict, verdicts: dict, ctl: dict,
           discriminator: dict | None = None, placebo_mode: bool = False, n_perm: int = N_PERM,
           exploratory: list[dict] | None = None, include_exploratory: bool = False) -> str:
    L = ["# Tables" + (" (placebo: R runs 4-7 relabelled as SA)" if placebo_mode else ""), "",
         "Written by `python -m rab.analyze`; every number below comes from the continuation "
         "files under the analysed root. Shares in percent, errors excluded from every "
         f"distribution. Intervals: event bootstrap, {N_BOOT:,} draws, 95%. p: within-event "
         f"permutation of arm labels, {n_perm:,} draws.", ""]
    L += ["## Verdict", "", "| cell | verdict | primary TVD(SA, R) | 95% interval | p | p (Holm) |",
          "|---|---|---|---|---|---|"]
    for kind, c in analysis["cells"].items():
        sa = c["compare"].get("SA") or {}
        L.append(f"| {kind} | {verdicts[kind]} | {f3(sa.get('tvd'))} | "
                 f"{interval(sa.get('lo'), sa.get('hi'), f3)} | {f3(sa.get('p'))} | {f3(c['primary_p_holm'])} |")
    L += ["", f"Control: R − P regrade gap on unreachable {pct(ctl['regrade_gap_R_minus_P'])} points "
          f"(needs {pct(CONTROL_GAP)}); TVD(B, R) outside the permutation null: "
          + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in sorted(ctl["B_outside_null"].items()))
          + f". Control {'FAILED' if ctl['failed'] else 'held'}.", ""]
    for kind, c in analysis["cells"].items():
        arms = c["arms"]
        L += [f"## {kind}: {c['events']} events, {c['tasks']} tasks", "",
              "| class | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for k in CLASSES:
            cells = []
            for a in arms:
                d = c["distribution"][a]
                n = d["counts"][k]
                cells.append(f"{n}" if k == "error" else f"{n} ({pct(d['shares'][k])})")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("| runs (events) | " + " | ".join(f"{c['distribution'][a]['n']} ({c['distribution'][a]['events']})"
                                                for a in arms) + " |")
        L += ["", "| arm vs R | events | TVD | 95% interval | p | within ±10 on every class share |",
              "|---|---|---|---|---|---|"]
        for a, cmp in c["compare"].items():
            L.append(f"| {a} | {cmp['events']} | {f3(cmp['tvd'])} | {interval(cmp['lo'], cmp['hi'], f3)} | "
                     f"{f3(cmp['p'])} | {'yes' if equivalent(cmp) else 'no'} |")
        if "SA" in c["compare"]:
            sa = c["compare"]["SA"]
            L += ["", "| class | SA − R (points) | 95% interval |", "|---|---|---|"]
            for j, k in enumerate(CLASSES_NO_ERROR):
                L.append(f"| {k} | {pct(sa['diff'][j])} | {interval(sa['diff_lo'][j], sa['diff_hi'][j])} |")
        if c["sign_flip"]:
            sf = c["sign_flip"]
            L += ["", f"Sign flip on the leading class (`{c['leading_class']}`), {len(sf['diffs'])} tasks: "
                  f"p = {f3(sf['p'])}; per task SA − R: "
                  + ", ".join(f"{t} {pct(v)}" for t, v in sorted(sf["diffs"].items())) + "."]
        en = c["equal_n"]
        L += ["", "Sensitivity: equal n, SA against R runs 4-7 "
              f"TVD {f3(en['SA_vs_R4_7']['tvd'])} (p {f3(en['SA_vs_R4_7']['p'])}), beside R 0-3 against "
              f"R 4-7 TVD {f3(en['R0_3_vs_R4_7']['tvd'])} (p {f3(en['R0_3_vs_R4_7']['p'])})"
              + (f"; without size-visible runs TVD {f3(c['no_size_visible']['tvd'])} "
                 f"(p {f3(c['no_size_visible']['p'])})" if c["no_size_visible"] else "") + ".", ""]
        L += ["| arm | regrade pass | 95% Wilson | reread | named | flagged & fitted | flagged only | fitted only | neither |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            r, t = c["rates"][a], c["two_by_two"][a]
            g = r["regrade_pass"]
            L.append(f"| {a} | {g['k']}/{r['n']} ({pct(g['rate'])}) | {interval(g['lo'], g['hi'])} | "
                     f"{pct(r['reread']['rate'])} | {pct(r['named']['rate'])} | "
                     f"{t['flagged=True/fitted=True']} | {t['flagged=True/fitted=False']} | "
                     f"{t['flagged=False/fitted=True']} | {t['flagged=False/fitted=False']} |")
        L += ["", "| moderator stratum (SA vs R) | events | TVD | p |", "|---|---|---|---|"]
        for name, m in c["moderators"].items():
            L.append(f"| {name} | {m['events']} | {f3(m['tvd'])} | {f3(m['p'])} |")
        L.append("")
    if discriminator:
        L += ["## Discriminator (pairwise, which is real)", "",
              "| comparison | pairs | balanced accuracy | 95% interval | surface-feature model (by task) |",
              "|---|---|---|---|---|"]
        for name, d in discriminator.items():
            L.append(f"| {name} | {d['n']} | {pct(d['acc'])} | {interval(d['lo'], d['hi'])} | {pct(d.get('surface'))} |")
        L.append("")
    if exploratory:
        hashes = sorted({str(r["git_hash"]) for r in exploratory})
        L += [f"## Exploratory runs ({'included above' if include_exploratory else 'excluded'})", "",
              f"{len(exploratory)} exploratory continuation file(s) on "
              f"{len({r['event_id'] for r in exploratory})} event(s), produced at git "
              + ", ".join(f"`{h}`" for h in hashes) + ".", ""]
    L += ["## Drops", "", "| what | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for name, d in drops_.items():
        if set(d) <= set(KINDS):
            L.append(f"| {name} | " + " | ".join(str(d.get(k, 0)) for k in KINDS) + " |")
        else:
            L.append(f"| {name} | " + "; ".join(f"{k}: {v}" for k, v in d.items()) + " | |")
    L += ["", "## Cost", "",
          f"Agent: ${cost_['agent_usd_total']:.4f} (" + ", ".join(
              f"{a} ${v:.4f}" for a, v in cost_["agent_usd_by_arm"].items()) + "). Simulator tokens: "
          + (", ".join(f"{k} {v[0]} in / {v[1]} out" for k, v in cost_["simulator_tokens"].items()) or "none")
          + f". Discriminator calls: {cost_['discriminator_calls']}.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ hand read

def hand_read(root: Path, rows: list[dict], seed: int = SEED, n: int = HAND_READ_PAIRS) -> tuple[str, dict]:
    """Five matched (R, SA) pairs on solvable events, unlabelled; the key separately."""
    root = Path(root)
    events = sorted({r["event_id"] for r in rows if r["kind"] == "solvable" and r["arm"] == "SA"}
                    & {r["event_id"] for r in rows if r["kind"] == "solvable" and r["arm"] == "R"})
    rng = random.Random(seed)
    picked = rng.sample(events, min(n, len(events)))
    L = ["# Hand read: matched pairs", "",
         "Each item shows the end of the recorded prefix, then two continuations from the same "
         "block point, A and B, in full. Write your class for each before opening "
         "`hand_read_key.json`.", ""]
    key = {}
    for j, eid in enumerate(picked, 1):
        rs = sorted(r["run_idx"] for r in rows if r["event_id"] == eid and r["arm"] == "R")
        ss = sorted(r["run_idx"] for r in rows if r["event_id"] == eid and r["arm"] == "SA")
        pair = [("R", rng.choice(rs)), ("SA", rng.choice(ss))]
        rng.shuffle(pair)
        key[f"item {j}"] = {"event_id": eid, "A": f"{pair[0][0]}-{pair[0][1]}", "B": f"{pair[1][0]}-{pair[1][1]}"}
        L += [f"## Item {j}", ""]
        files = {(r["arm"], r["run_idx"]): r["file"] for r in rows if r["event_id"] == eid}
        slots = [load_slot(root / "continuations" / eid / files[(arm, idx)]) for arm, idx in pair]
        tr = slots[0]["cont"]["transcript"]
        k = slots[0]["cont"]["block_k"]
        L += ["### Prefix (last turns before the block)", ""] + [_entry(e) for e in tr[max(1, k - 4):k]]
        for label, s in zip("AB", slots):
            L += [f"### Continuation {label}", ""] + [_entry(e) for e in s["cont"]["transcript"][k:]]
            L += [f"Submitted: {s['cont'].get('submitted')}; summary: {s['cont'].get('submit_summary')!r}", ""]
    return "\n".join(L) + "\n", key


def _entry(e: dict) -> str:
    body = e.get("content") or ""
    calls = "".join(f"\n[call {c['name']}] {json.dumps(c['args'], ensure_ascii=False)}"
                    for c in e.get("tool_calls") or [])
    return f"**{e['role']}**\n\n```\n{body}{calls}\n```\n"


# ------------------------------------------------------------------ main

def run(root: Path, *, placebo_mode: bool = False, n_perm: int = N_PERM,
        events_path: Path = EVENTS_PATH, recount_path: Path = RECOUNT_PATH,
        include_exploratory: bool = False) -> dict:
    from rab.run import git_state

    root = Path(root)
    all_rows, events = load(root, events_path)
    rows, explo = split_exploratory(all_rows, include_exploratory)
    if placebo_mode:
        rows = placebo(rows)
    analysis = analyse(rows, n_perm)
    verdicts = verdict(analysis)
    ctl = control(analysis)
    disc = None
    disc_rows = [r for r in read_jsonl(root / "discriminator.jsonl")
                 if include_exploratory or not r.get("exploratory")]
    if disc_rows:
        from rab.discriminate import summarise, surface_baseline
        surface = surface_baseline(disc_rows)
        disc = {k: dict(v, surface=surface.get(k, {}).get("acc")) for k, v in summarise(disc_rows).items()}
    text = render(analysis, drops(root, rows, recount_path), cost(root, rows), verdicts, ctl, disc, placebo_mode, n_perm,
                  explo, include_exploratory)
    suffix = "_placebo" if placebo_mode else ""
    (root / f"tables{suffix}.md").write_text(text, encoding="utf-8", newline="\n")
    files = slot_files(root)
    meta = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "root": str(root), "placebo": placebo_mode, "n_slot_files": len(files),
            "n_exploratory": len(explo), "include_exploratory": include_exploratory, "git": git_state(),
            "n_perm": n_perm, "n_boot": N_BOOT, "seed": SEED,
            "tables_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "verdicts": verdicts}
    (root / f"tables{suffix}_meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n",
                                                   encoding="utf-8", newline="\n")
    if not placebo_mode:
        md, key = hand_read(root, rows)
        (root / "hand_read.md").write_text(md, encoding="utf-8", newline="\n")
        (root / "hand_read_key.json").write_text(json.dumps(key, sort_keys=True, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    for kind, v in verdicts.items():
        print(f"{kind}: {v}", file=sys.stderr)
    return {"text": text, "verdicts": verdicts, "analysis": analysis, "control": ctl}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rab.analyze", description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=RESULTS)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--include-exploratory", action="store_true",
                    help="analyse exploratory slot files too (default: list and exclude them)")
    args = ap.parse_args(argv)
    run(args.root, placebo_mode=args.placebo, n_perm=args.n_perm,
        include_exploratory=args.include_exploratory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
