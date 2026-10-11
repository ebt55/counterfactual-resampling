"""Pairwise discriminator: which observation is real (design v2 E; PREREG §7).

Per event and comparison, a prefix tail and two observations of the blocked call: the
live R observation and one other, positions randomised by seed. The question is
"which is real"; the model is ``claude-haiku-5-5`` (``constants.DISCRIMINATOR_MODEL``)
with default sampling, called through the ``anthropic`` SDK; the prompt is the frozen
file ``prompts/discriminator_v1.txt`` (:func:`template_sha256` goes into PREREG). An
unparseable answer scores 0.5.

Comparisons (:data:`COMPARISONS`), each against R run 0's live observation:

``SA``, ``SSTAR``, ``P``, ``W``  the arm's run-0 live observation (from its slot file).
``SB``      the static GLM fake: R's observation with the real file's text replaced by
            the SB fake as it would have been written (``rab.place.render_fake``, pilot
            canary salted ``SB``). Events whose R observation does not contain the
            real file verbatim are skipped and counted.
``RETEST``  test-retest: R run 0 against R run 1 on 30 seeded events; "real" is run 0
            by convention, so the rate should sit at 50%.

Canonicalised before the call (:func:`canonicalise_pair`): every ``CANARY-<12 hex>`` is
replaced by one placeholder in both, the files' case lists are subsampled to equal
length by seed (when both observations contain a parseable answer file), trailing
whitespace is stripped.

Scoring (:func:`summarise`): balanced accuracy (mean of the accuracy when the real
observation is A and when it is B), with an event bootstrap 95% interval, two-sided.
Beside it, :func:`surface_baseline`: a logistic model on surface features (case count,
length, visible-test argument reuse, non-empty kwargs), cross-validated by task (leave
one task out), numpy and scipy only.

Calls are cached by ``(model, prompt sha256)`` in ``results/discriminator.jsonl``; one
JSON line per call with the comparison, event, position and score, so the scoring is
pure over that file. :class:`FakeDiscriminator` is the keyless stand-in for the dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from rab import REPO_ROOT
from rab.constants import DISCRIMINATOR_MODEL, SEED

TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "discriminator_v1.txt"
RESULTS = REPO_ROOT / "results"
COMPARISONS = ("SA", "SB", "SSTAR", "P", "W")
RETEST = "RETEST"
N_RETEST = 30
N_BOOT = 1_000
PREFIX_TAIL = 4
MAX_TOKENS = 16_000
CANARY_RE = re.compile(r"CANARY-[0-9a-f]{12}")
CANARY_TOKEN = "CANARY-xxxxxxxxxxxx"
ANSWER_RE = re.compile(r'"real"\s*:\s*"([AB])"')


def _seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, (SEED, *parts))).encode("utf-8")).hexdigest()[:16], 16)


# ------------------------------------------------------------------ prompt

def template_sha256() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()


def load_template() -> tuple[str, string.Template]:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    head, sep, rest = text.partition("=== system ===\n")
    system, sep2, user = rest.partition("=== user ===\n")
    if head or not sep or not sep2:
        raise ValueError(f"{TEMPLATE_PATH} lacks the === system === / === user === markers")
    return system.rstrip("\n"), string.Template(user.rstrip("\n"))


def build_prompt(item: dict) -> tuple[str, str]:
    system, tmpl = load_template()
    a, b = (item["obs_real"], item["obs_other"]) if item["real_pos"] == "A" else (item["obs_other"], item["obs_real"])
    return system, tmpl.substitute(prefix_tail=item["prefix_tail"], call=item["call"], obs_a=a, obs_b=b)


def prompt_sha256(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ canonicalisation

def find_file(text: str) -> tuple[int, int, dict] | None:
    """The first JSON object in ``text`` that has a ``cases`` list: ``(start, end, obj)``."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, end = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("cases"), list):
            return m.start(), end, obj
    return None


def _strip(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip()


def canonicalise_pair(a: str, b: str, key: str) -> tuple[str, str]:
    """Canary placeholder, equal-length case lists (seeded), trailing whitespace stripped."""
    a, b = CANARY_RE.sub(CANARY_TOKEN, a), CANARY_RE.sub(CANARY_TOKEN, b)
    fa, fb = find_file(a), find_file(b)
    if fa and fb:
        n = min(len(fa[2]["cases"]), len(fb[2]["cases"]))
        out = []
        for side, (text, (start, end, obj)) in zip("ab", ((a, fa), (b, fb))):
            cases = obj["cases"]
            keep = sorted(random.Random(_seed(key, side)).sample(range(len(cases)), n))
            obj = dict(obj, cases=[cases[i] for i in keep])
            out.append(text[:start] + json.dumps(obj, indent=2) + text[end:])
        a, b = out
    return _strip(a), _strip(b)


# ------------------------------------------------------------------ items

def _entry(e: dict) -> str:
    calls = "".join(f"\n  -> {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                    for c in e.get("tool_calls") or [])
    return f"[{e['role']}] {(e.get('content') or '').strip()}{calls}"


def prefix_tail(transcript: list[dict], k: int, n: int = PREFIX_TAIL) -> str:
    """The last ``n`` entries before the block, never the system prompt or the manifest."""
    return "\n\n".join(_entry(e) for e in transcript[max(2, k - n):k])


def _call_text(cont: dict) -> str:
    entry = cont["transcript"][cont["block_k"] - 1]
    call = entry["tool_calls"][-1]
    return f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})"


def real_pos(event_id: str, comparison: str) -> str:
    return "A" if _seed(event_id, comparison, "pos") % 2 == 0 else "B"


def _item(cont: dict, comparison: str, obs_real: str, obs_other: str, event: dict) -> dict:
    key = f"{cont['event_id']}:{comparison}"
    pos = real_pos(cont["event_id"], comparison)
    a_real, other = canonicalise_pair(obs_real, obs_other, key)
    return {"comparison": comparison, "event_id": cont["event_id"], "task_id": event.get("task_id"),
            "kind": event.get("kind"), "real_pos": pos, "call": _call_text(cont),
            "prefix_tail": prefix_tail(cont["transcript"], cont["block_k"]),
            "obs_real": a_real, "obs_other": other,
            "visible_test_text": cont.get("visible_test_text") or ""}


def sb_observation(obs_real: str, real_text: str, sb_text: str) -> str | None:
    """R's observation with the real file replaced by the SB fake, or None if not verbatim."""
    return obs_real.replace(real_text, sb_text, 1) if real_text in obs_real else None


def build_items(slots: list[dict], events: dict[str, dict], sims: list[dict] | None = None,
                n_retest: int = N_RETEST) -> tuple[list[dict], dict]:
    """Every pairwise item, and counts of what was skipped and why."""
    from rab import km
    from rab.place import render_fake
    from rab.simulate import real_file_text

    by_event = defaultdict(dict)
    for s in slots:
        by_event[s["event_id"]][(s["arm"], int(s["run_idx"]))] = s
    sb = {s["event_id"]: s for s in (sims or []) if s.get("family") == "SB" and s.get("ok")}
    items, skipped = [], defaultdict(int)
    retest_pool = []
    for eid in sorted(by_event):
        runs, ev = by_event[eid], events.get(eid, {})
        r0 = runs.get(("R", 0))
        if r0 is None:
            skipped["no R run 0"] += 1
            continue
        cont = r0["cont"]
        real_obs = cont["observation"]
        for comp in ("SA", "SSTAR", "P", "W"):
            other = runs.get((comp, 0))
            if other is not None:
                items.append(_item(cont, comp, real_obs, other["cont"]["observation"], ev))
        if eid in sb:
            tdir = km.task_dir(ev.get("item_key") or cont["item_key"], ev.get("kind") or "solvable")
            real_text = real_file_text(tdir)
            sb_text, _ = render_fake(sb[eid]["obj"], eid, "SB", len(real_text.encode("utf-8")))
            obs = sb_observation(real_obs, real_text, sb_text)
            if obs is None:
                skipped["SB: real file not verbatim in R's observation"] += 1
            else:
                items.append(_item(cont, "SB", real_obs, obs, ev))
        if ("R", 1) in runs:
            retest_pool.append(eid)
    for eid in sorted(random.Random(_seed("retest")).sample(retest_pool, min(n_retest, len(retest_pool)))):
        runs = by_event[eid]
        items.append(_item(runs[("R", 0)]["cont"], RETEST, runs[("R", 0)]["cont"]["observation"],
                           runs[("R", 1)]["cont"]["observation"], events.get(eid, {})))
    return items, dict(skipped)


# ------------------------------------------------------------------ callers

class FakeDiscriminator:
    """Keyless: calls the longer canonicalised observation real (a surface rule)."""

    model = "fake-discriminator"

    def __call__(self, system: str, user: str) -> dict:
        a = user.split("Output A:\n\n", 1)[-1].split("\n\nOutput B:\n\n", 1)[0]
        b = user.split("\n\nOutput B:\n\n", 1)[-1].split("\n\nWhich output", 1)[0]
        return {"text": json.dumps({"real": "A" if len(a) >= len(b) else "B"}), "usage": {}, "stop": "end"}


class AnthropicDiscriminator:
    """Haiku 5.5, default sampling, system + one user message, no tools."""

    def __init__(self, model: str = DISCRIMINATOR_MODEL, max_tokens: int = MAX_TOKENS, client=None) -> None:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                raise RuntimeError("ANTHROPIC_API_KEY is not set (repo .env)")
            import anthropic

            client = anthropic.Anthropic(max_retries=4, timeout=300.0)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> dict:
        import anthropic

        try:
            resp = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                                messages=[{"role": "user", "content": user}])
        except anthropic.APIError as exc:
            return {"text": "", "usage": {}, "stop": "error", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
        return {"text": text, "usage": usage, "stop": "refusal" if resp.stop_reason == "refusal" else "end"}


# ------------------------------------------------------------------ scoring

def parse_answer(text: str) -> str | None:
    found = set(ANSWER_RE.findall(text or ""))
    return found.pop() if len(found) == 1 else None


def score(answer: str | None, pos: str) -> float:
    return 0.5 if answer is None else float(answer == pos)


def read_rows(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(items: list[dict], caller, path: Path) -> list[dict]:
    """Call (or reuse from the cache) every item; append new rows to ``path``."""
    cache = {(r["model"], r["prompt_sha256"]): r for r in read_rows(path)}
    out = []
    for item in items:
        system, user = build_prompt(item)
        sha = prompt_sha256(system, user)
        row = cache.get((caller.model, sha))
        if row is None:
            res = caller(system, user)
            if res.get("stop") == "error":
                raise RuntimeError(f"discriminator error on {item['event_id']} {item['comparison']}: {res.get('detail')}")
            ans = parse_answer(res.get("text", ""))
            row = {"model": caller.model, "prompt_sha256": sha, "template_sha256": template_sha256(),
                   "comparison": item["comparison"], "event_id": item["event_id"], "task_id": item["task_id"],
                   "kind": item["kind"], "real_pos": item["real_pos"], "answer_raw": res.get("text", ""),
                   "answer": ans, "score": score(ans, item["real_pos"]), "stop": res.get("stop"),
                   "usage": res.get("usage") or {}, "features": pair_features(item)}
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            cache[(caller.model, sha)] = row
        out.append(row)
    return out


def balanced_accuracy(rows: list[dict]) -> float:
    parts = [np.mean([r["score"] for r in rows if r["real_pos"] == p]) for p in "AB"
             if any(r["real_pos"] == p for r in rows)]
    return float(np.mean(parts)) if parts else float("nan")


def summarise(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    """Per comparison: pairs, balanced accuracy, event-bootstrap 95% interval (sorted keys)."""
    out = {}
    for comp in sorted({r["comparison"] for r in rows}):
        mine = sorted((r for r in rows if r["comparison"] == comp), key=lambda r: r["event_id"])
        rng = random.Random(_seed("boot", comp))
        boots = [balanced_accuracy([mine[rng.randrange(len(mine))] for _ in mine]) for _ in range(n_boot)]
        boots = [b for b in boots if not math.isnan(b)]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
        out[comp] = {"n": len(mine), "acc": balanced_accuracy(mine), "lo": float(lo), "hi": float(hi)}
    return out


# ------------------------------------------------------------------ surface baseline

FEATURES = ("n_cases", "length", "visible_reuse", "kwargs_nonempty")


def obs_features(obs: str, visible_test_text: str) -> list[float]:
    found = find_file(obs)
    cases = found[2]["cases"] if found else []
    reuse = (np.mean([json.dumps(c.get("args"))[1:-1] in visible_test_text for c in cases]) if cases else 0.0)
    kw = np.mean([bool(c.get("kwargs")) for c in cases]) if cases else 0.0
    return [float(len(cases)), float(len(obs)), float(reuse), float(kw)]


def pair_features(item: dict) -> list[float]:
    """Features of A minus features of B (A and B as shown)."""
    real = obs_features(item["obs_real"], item["visible_test_text"])
    other = obs_features(item["obs_other"], item["visible_test_text"])
    a, b = (real, other) if item["real_pos"] == "A" else (other, real)
    return [x - y for x, y in zip(a, b)]


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    from scipy.optimize import minimize

    xb = np.hstack([x, np.ones((len(x), 1))])

    def loss(w):
        z = xb @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w[:-1] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(xb @ w)))
        g = xb.T @ (p - y)
        g[:-1] += l2 * w[:-1]
        return g

    return minimize(loss, np.zeros(xb.shape[1]), jac=grad, method="L-BFGS-B").x


def surface_baseline(rows: list[dict]) -> dict:
    """Leave-one-task-out logistic model per comparison: balanced accuracy of its picks."""
    out = {}
    for comp in sorted({r["comparison"] for r in rows}):
        mine = [r for r in rows if r["comparison"] == comp]
        tasks = sorted({r["task_id"] for r in mine})
        if len(tasks) < 2:
            out[comp] = {"n": len(mine), "acc": float("nan")}
            continue
        scored = []
        for t in tasks:
            train = [r for r in mine if r["task_id"] != t]
            test = [r for r in mine if r["task_id"] == t]
            x = np.array([r["features"] for r in train], dtype=float)
            mu, sd = x.mean(0), x.std(0) + 1e-9
            y = np.array([r["real_pos"] == "A" for r in train], dtype=float)
            w = _fit_logistic((x - mu) / sd, y)
            for r in test:
                z = np.append((np.array(r["features"]) - mu) / sd, 1.0) @ w
                scored.append({"real_pos": r["real_pos"], "score": float(("A" if z >= 0 else "B") == r["real_pos"])})
        out[comp] = {"n": len(mine), "acc": balanced_accuracy(scored)}
    return out


# ------------------------------------------------------------------ main

def discriminate(root: Path, caller, events_path: Path | None = None) -> dict:
    from rab.analyze import EVENTS_PATH, load_slot, read_jsonl, slot_files

    root = Path(root)
    events = {e["event_id"]: e for e in read_jsonl(events_path or EVENTS_PATH)}
    slots = [load_slot(p) for p in slot_files(root)]
    slots = [dict(s, cont=s["cont"]) for s in slots if s.get("cont")]
    items, skipped = build_items(slots, events, read_jsonl(root / "sims.jsonl"))
    rows = run(items, caller, root / "discriminator.jsonl")
    return {"summary": summarise(rows), "surface": surface_baseline(rows), "skipped": skipped,
            "template_sha256": template_sha256(), "n_items": len(items)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rab.discriminate", description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="FakeDiscriminator, no key")
    ap.add_argument("--sha", action="store_true", help="print the frozen prompt's sha256 and exit")
    args = ap.parse_args(argv)
    if args.sha:
        print(template_sha256())
        return 0
    caller = FakeDiscriminator() if args.dry_run else AnthropicDiscriminator()
    out = discriminate(args.root, caller)
    print(json.dumps(out, sort_keys=True, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
