"""Keyed day-0 checks (guide §1 "Served" and "Credit"); checks, not gates.

    uv run python tools/day0.py

Fetches the OpenRouter endpoint lists for the agent and the SB simulator (no key),
writes them to ``results/provider.log``, picks DeepSeek's own endpoint and Z.AI's GLM
endpoint, makes exactly one pinned DeepSeek call (real ``TOOLS``, ``max_tokens`` 16) with
``rab.resume``'s logging hook and one ``claude-haiku-5-5`` call, and writes
``results/d0_keyed.md``. Costs cents. Keys come from the repo's own ``.env``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rab.constants as C
from rab import REPO_ROOT, km
from rab.resume import PROVIDER_LOG, ProviderPinError, _append, attach_hook
from harness.agent import TOOLS
from harness.providers.base import Msg

D0_KEYED = REPO_ROOT / "results" / "d0_keyed.md"

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


def fetch_endpoints(model: str) -> dict:
    import httpx

    resp = httpx.get(ENDPOINTS_URL.format(model=model), timeout=60)
    resp.raise_for_status()
    return resp.json()


def _price(ep: dict) -> dict:
    p = ep.get("pricing") or {}
    per = lambda key: round(float(p.get(key) or 0) * 1e6, 6)  # noqa: E731 - USD/token -> USD/MTok
    out = {"input": per("prompt"), "output": per("completion")}
    if p.get("input_cache_read") not in (None, ""):
        out["cached"] = per("input_cache_read")
    return out


def pick_endpoint(endpoints: list[dict], want_name: str | None = None, need: str | None = None) -> dict | None:
    """``want_name`` (display name, case-insensitive) if served, else the cheapest that supports ``need``."""
    live = [e for e in endpoints if e.get("status", 0) in (0, None)] or endpoints
    if want_name:
        hit = [e for e in live if str(e.get("provider_name", "")).lower() == want_name.lower()]
        return hit[0] if hit else None
    ok = [e for e in live if not need or need in (e.get("supported_parameters") or [])]
    return min(ok, key=lambda e: (_price(e)["input"] + _price(e)["output"]), default=None)


def day0(log_path: Path = PROVIDER_LOG, report_path: Path = D0_KEYED) -> dict:
    """Guide §1 "Served" and "Credit": endpoint lists, one pinned DeepSeek call, one Haiku call."""
    from harness.providers.openai_compat import make

    out: dict = {"date": km.utc_now()}
    eps = {}
    for model in (C.AGENT_MODEL, C.SIM_MODELS["SB"]):
        data = fetch_endpoints(model)
        eps[model] = (data.get("data") or {}).get("endpoints") or []
        _append(log_path, {"ts": km.utc_now(), "kind": "endpoints", "model": model, "endpoints": eps[model]})
    deepseek = pick_endpoint(eps[C.AGENT_MODEL], want_name="DeepSeek")
    glm = (pick_endpoint(eps[C.SIM_MODELS["SB"]], want_name="Z.AI")
           or pick_endpoint(eps[C.SIM_MODELS["SB"]], need="temperature"))
    out["agent_endpoint"] = deepseek and {"provider_name": deepseek.get("provider_name"),
                                          "tag": deepseek.get("tag"), "price": _price(deepseek)}
    out["sb_endpoint"] = glm and {"provider_name": glm.get("provider_name"), "tag": glm.get("tag"),
                                  "price": _price(glm)}
    out["endpoint_counts"] = {m: len(v) for m, v in eps.items()}
    if deepseek is None:
        out["agent_call"] = "skipped: DeepSeek's own endpoint is not served"
    else:
        provider = make("openrouter", C.AGENT_MODEL, verify_model=False, max_tokens=16,
                        extra_body={"provider": {"order": [deepseek["tag"]], "allow_fallbacks": False},
                                    "usage": {"include": True}})
        ctx = attach_hook(provider, "day0", pin_name=deepseek.get("provider_name"), log_path=log_path)
        ctx.update(event_id="day0", arm="served", run_idx=0)
        try:
            turn = provider.chat("You are a test.", [Msg(role="user", content="Reply with the word ok.")], TOOLS)
            line = ctx["lines"][-1] if ctx["lines"] else {}
            out["agent_call"] = {"stop_reason": turn.stop_reason, "text": turn.text[:40],
                                 "error": turn.refusal_details,
                                 "served_by": line.get("provider"), "status": line.get("status"),
                                 "usage": line.get("usage"), "pin_held": line.get("provider") == deepseek.get("provider_name")}
        except ProviderPinError as exc:
            out["agent_call"] = {"pin_held": False, "error": str(exc)}
        finally:
            provider.close()
    try:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=60.0)
        raw = client.messages.with_raw_response.create(
            model=C.DISCRIMINATOR_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the word ok."}])
        msg = raw.parse()
        out["credit_call"] = {"status": raw.http_response.status_code, "stop_reason": msg.stop_reason,
                              "usage": {"input_tokens": msg.usage.input_tokens,
                                        "output_tokens": msg.usage.output_tokens}}
    except Exception as exc:  # recorded, the report says what failed
        out["credit_call"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    _append(log_path, {"ts": km.utc_now(), "kind": "day0_summary", **out})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_d0_report(out), encoding="utf-8", newline="\n")
    return out


def _d0_report(o: dict) -> str:
    a, s = o.get("agent_endpoint") or {}, o.get("sb_endpoint") or {}
    ac, cc = o.get("agent_call"), o.get("credit_call")
    lines = [
        "# Day-0 keyed checks (guide §1 \"Served\" and \"Credit\")", "",
        f"Run {o['date']} by `uv run python -m rab.resume --day0`. Full endpoint lists are the "
        "`kind: endpoints` lines of `results/provider.log`; prices are USD per million tokens "
        "from the OpenRouter endpoints API.", "",
        "| Seat | Model | Endpoint (provider_name) | Routing tag (`provider.order`) | Input | Output | Cache read |",
        "|---|---|---|---|---|---|---|",
        f"| agent | `{C.AGENT_MODEL}` | {a.get('provider_name', 'NOT SERVED')} | `{a.get('tag')}` | "
        f"{(a.get('price') or {}).get('input')} | {(a.get('price') or {}).get('output')} | {(a.get('price') or {}).get('cached', '-')} |",
        f"| SB simulator | `{C.SIM_MODELS['SB']}` | {s.get('provider_name')} | `{s.get('tag')}` | "
        f"{(s.get('price') or {}).get('input')} | {(s.get('price') or {}).get('output')} | {(s.get('price') or {}).get('cached', '-')} |",
        "", f"Endpoints listed: {o.get('endpoint_counts')}.", "",
        "## Served: one pinned DeepSeek call (real `TOOLS`, `max_tokens` 16)", "",
        "```json", json.dumps(ac, indent=1, ensure_ascii=False), "```", "",
        "## Credit: one `claude-haiku-5-5` call (`max_tokens` 16)", "",
        "```json", json.dumps(cc, indent=1, ensure_ascii=False), "```", ""]
    if o.get("notes"):
        lines += ["## Notes", "", *o["notes"], ""]
    return "\n".join(lines)


def main() -> int:
    out = day0()
    print(json.dumps({k: v for k, v in out.items()}, indent=1, ensure_ascii=False, default=str), file=sys.stderr)
    ok = isinstance(out.get("agent_call"), dict) and out["agent_call"].get("pin_held") \
        and (out.get("credit_call") or {}).get("status") == 200
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
