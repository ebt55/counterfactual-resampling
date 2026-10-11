# Day-0 keyed checks (guide §1 "Served" and "Credit")

Run 2026-10-11T12:59:38Z by `uv run python tools/day0.py`. Full endpoint lists are the `kind: endpoints` lines of `results/provider.log`; prices are USD per million tokens from the OpenRouter endpoints API.

| Seat | Model | Endpoint (provider_name) | Routing tag (`provider.order`) | Input | Output | Cache read |
|---|---|---|---|---|---|---|
| agent | `deepseek/deepseek-v4.1-flash` | DeepSeek | `deepseek` | 0.15 | 0.6 | 0.003 |
| SB simulator | `z-ai/glm-5.3-flash` | Z.AI | `z-ai/fp8` | 0.15 | 0.5 | 0.03 |

Endpoints listed: {'deepseek/deepseek-v4.1-flash': 32, 'z-ai/glm-5.3-flash': 33}.

## Served: one pinned DeepSeek call (real `TOOLS`, `max_tokens` 16)

```json
{
 "stop_reason": "end_turn",
 "text": "ok",
 "error": null,
 "served_by": "DeepSeek",
 "status": 200,
 "usage": {
  "prompt_tokens": 600,
  "completion_tokens": 2,
  "total_tokens": 602,
  "cost": 9.12e-05,
  "is_byok": false,
  "prompt_tokens_details": {
   "cached_tokens": 0,
   "cache_write_tokens": 0,
   "audio_tokens": 0,
   "video_tokens": 0
  },
  "cost_details": {
   "upstream_inference_cost": 9.12e-05,
   "upstream_inference_prompt_cost": 9e-05,
   "upstream_inference_completions_cost": 1.2e-06
  },
  "completion_tokens_details": {
   "reasoning_tokens": 0,
   "image_tokens": 0,
   "audio_tokens": 0
  }
 },
 "pin_held": true
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
