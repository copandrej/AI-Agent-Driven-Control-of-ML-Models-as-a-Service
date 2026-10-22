"""
LLM Call Telemetry
==================
One JSONL line per LLM call, for every architecture that goes through litellm:
  - timing (UTC start, duration, the server's Date header for aligning with the hub's energy log)
  - usage (prompt / completion / reasoning tokens), vLLM request id, finish reason
  - the answer text (the ReAct step) and logprob summaries of reasoning, answer and the Action part
  - each agent's system prompt (role, tools, rules) once per file
Reasoning text and full prompts are not stored (size); use --debug-log for those.
Per-task vLLM counter deltas (tokens, prefix cache, preemptions) come from the server's /metrics.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import litellm

_ROLE = re.compile(r"^You are (.+?)\. ")
_COUNTERS = ("prompt_tokens_total", "generation_tokens_total", "prefix_cache_hits_total",
             "prefix_cache_queries_total", "num_preemptions_total")

_path: Path | None = None
_task: str | None = None
_seen_prompts: set[str] = set()
_completion = litellm.completion


def start(path: Path) -> None:
    """Route LLM calls through the recorder and write them to path (one file per architecture run)."""
    global _path
    _path = path
    _seen_prompts.clear()
    litellm.completion = _recorded_completion


def set_task(task_id: str | None) -> None:
    global _task
    _task = task_id


def _write(record: dict) -> None:
    with open(_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _summary(logprobs: list[float]) -> dict:
    if not logprobs:
        return {}
    return {"n": len(logprobs), "mean": round(sum(logprobs) / len(logprobs), 4), "min": round(min(logprobs), 4)}


def _logprob_summaries(tokens: list, n_reasoning: int) -> dict:
    """vLLM returns logprobs for reasoning and answer tokens in order; the reasoning ones come first."""
    reasoning, answer = tokens[:n_reasoning], tokens[n_reasoning:]
    result = {"reasoning": _summary([t.logprob for t in reasoning]), "answer": _summary([t.logprob for t in answer])}
    action_at = "".join(t.token for t in answer).find("Action:")
    if action_at >= 0:  # the tool choice and its arguments
        pos, action = 0, []
        for t in answer:
            if pos + len(t.token) > action_at:
                action.append(t.logprob)
            pos += len(t.token)
        result["action"] = _summary(action)
    return result


def _recorded_completion(*args, **kwargs):
    if _path is None:
        return _completion(*args, **kwargs)
    if kwargs.get("logprobs") is None:  # crewai passes logprobs=None by default
        kwargs["logprobs"] = True
    messages = kwargs.get("messages") or []
    system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    role = m.group(1) if (m := _ROLE.match(system)) else None
    if system and role not in _seen_prompts:
        _seen_prompts.add(role)
        _write({"type": "prompt", "agent": role, "text": system})

    record = {"type": "call", "task": _task, "agent": role, "start": datetime.now(timezone.utc).isoformat()}
    t0 = time.monotonic()
    try:
        response = _completion(*args, **kwargs)
    except Exception as e:
        _write({**record, "duration_s": round(time.monotonic() - t0, 3), "error": type(e).__name__})
        raise
    record["duration_s"] = round(time.monotonic() - t0, 3)

    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    n_reasoning = getattr(details, "reasoning_tokens", None) or 0
    choice = response.choices[0]
    content = choice.message.content or ""
    headers = (getattr(response, "_hidden_params", {}) or {}).get("additional_headers") or {}
    logprobs = getattr(choice, "logprobs", None)
    tokens = getattr(logprobs, "content", None) or []
    _write({
        **record,
        "server_date": headers.get("llm_provider-date"),
        "request_id": response.id,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": n_reasoning,
        "content": content,
        "logprobs": _logprob_summaries(tokens, n_reasoning),
    })
    return response


# ── vLLM server ───────────────────────────────────────────────────────────


def _server(path: str) -> str | None:
    base = os.getenv("OPENAI_COMPAT_BASE_URL")
    if os.getenv("LLM_PROVIDER") != "openai_compat" or not base:
        return None
    try:
        r = httpx.get(base.removesuffix("/v1") + path, timeout=10, verify=False,
                      headers={"Authorization": f"Bearer {os.getenv('OPENAI_COMPAT_API_KEY', '')}"})
        return r.text if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def vllm_counters() -> dict[str, float]:
    """Cumulative vLLM counters; the per-task difference is the server-side cost of that task."""
    text = _server("/metrics") or ""
    totals = {}
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        key = name.split("{")[0].removeprefix("vllm:")
        if key in _COUNTERS:
            totals[key] = totals.get(key, 0) + float(value)
    return totals


def counter_delta(before: dict, after: dict) -> dict:
    return {k.removesuffix("_total"): after[k] - before[k] for k in after if k in before}


def server_info() -> dict:
    """Served model and vLLM version, for the run metadata."""
    info = {}
    try:
        model = json.loads(_server("/v1/models") or "{}")["data"][0]
        info = {"served_model": model.get("root"), "max_model_len": model.get("max_model_len")}
    except (ValueError, KeyError, IndexError):
        pass
    try:
        info["vllm_version"] = json.loads(_server("/version") or "{}").get("version")
    except ValueError:
        pass
    return info
