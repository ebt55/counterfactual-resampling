# Day-0 keyed checks (guide §1 "Served" and "Credit")

Run 2026-10-11T06:05:02Z by `uv run python -m rab.resume --day0`. Full endpoint lists are the `kind: endpoints` lines of `results/provider.log`; prices are USD per million tokens from the OpenRouter endpoints API.

| Seat | Model | Endpoint (provider_name) | Routing tag (`provider.order`) | Input | Output | Cache read |
|---|---|---|---|---|---|---|
| agent | `deepseek/deepseek-v4.1-flash` | DeepSeek | `deepseek` | 0.15 | 0.6 | 0.003 |
| SB simulator | `z-ai/glm-5.3-flash` | Z.AI | `z-ai/fp8` | 0.15 | 0.5 | 0.03 |

Endpoints listed: {'deepseek/deepseek-v4.1-flash': 32, 'z-ai/glm-5.3-flash': 33}.

## Served: one pinned DeepSeek call (real `TOOLS`, `max_tokens` 16)

```json
{
 "pin_held": false,
 "served_by": null,
 "status": 404,
 "stop_reason": "error",
 "text": "",
 "usage": null,
 "error": {
  "category": "api_error",
  "explanation": "HTTP 404: No endpoints found for deepseek/deepseek-v4.1-flash. Every candidate endpoint was removed during routing: ... Filter by Guardrails removed deepseek (Paid model training violation (account settings)); Filter by Fallback removed [every other endpoint, as allow_fallbacks is false] (body abridged; read from one identical unbilled reproduction of the request without the hook, 2026-10-11)"
 }
}
```

## Credit: one `claude-haiku-5-5` call (`max_tokens` 16)

```json
{
 "status": 200,
 "stop_reason": "end_turn",
 "usage": {
  "input_tokens": 17,
  "output_tokens": 4
 }
}
```

## Notes

- **Served: FAIL, not a model outage.** OpenRouter lists DeepSeek's own endpoint (tag `deepseek`, $0.15 / $0.60 / cache read $0.003, status 0), but routing removed it under the account's guardrail: "Paid model training violation (account settings)". DeepSeek's endpoint may train on paid prompts and the new account's privacy setting forbids that. With `allow_fallbacks: false` no other endpoint may serve, so the call returned 404 (not billed).
- **Fix is an account setting, Ebin's decision:** allow endpoints that may train on paid prompts in OpenRouter's privacy settings (or decide otherwise). Nothing in this repo was changed to route around it, and the line was not switched. Re-run `uv run python -m rab.resume --day0` afterwards; it makes one pinned DeepSeek call and one Haiku call (cents).
- **Credit: PASS.** One `claude-haiku-5-5` call returned HTTP 200 (17 in, 4 out tokens; about $0.000004).
- **SB pin chosen, not yet exercised:** Z.AI's own endpoint `z-ai/fp8` ($0.15 / $0.50 / $0.03). Whether the same guardrail admits it is unknown until a call is made.
- Prices of both pinned endpoints equal KM's `analysis/prices.json` (2026-09-15), so no price override is set.
