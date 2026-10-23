"""
Compact Task Trace
==================
Per-task facts for fault analysis, without the generated text:
  - tool calls (agent, tool, args, truncated output) from crewai events
  - cluster state diff (deployments, services, configmaps, ingresses, pods)
Full LLM text stays in the optional --debug-log.
"""

import hashlib
import json
import re

from crewai.events import (
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolValidateInputErrorEvent,
    crewai_event_bus,
)

from src.tools import run_cmd

MAX_CHARS = 200

_events: list | None = None  # active recorder buffer, None when not recording


def _short(value) -> str:
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"


def _on_tool_event(source, event) -> None:
    if _events is None:
        return
    finished = isinstance(event, ToolUsageFinishedEvent)
    out = event.output if finished else event.error
    out = str(out).split("\nYou ONLY have access to the following tools")[0]  # crewai's periodic reminder
    _events.append((event.timestamp, {
        "agent": event.agent_role,
        "tool": event.tool_name,
        "args": _short(event.tool_args),
        "out": _short(out),
        **({"duration_s": round((event.finished_at - event.started_at).total_seconds(), 3)} if finished
           else {"event": event.type}),
    }))


for _cls in (ToolUsageFinishedEvent, ToolUsageErrorEvent, ToolValidateInputErrorEvent, ToolSelectionErrorEvent):
    crewai_event_bus.on(_cls)(_on_tool_event)


_IN_CONTAINER = re.compile(r"\s*(?:kubectl\s+)?(?:exec|cp|debug|attach)\b")
_REFUSED = ("Denied", "INVALID")  # kubectl verb allow-lists (P1) and the manifest validation gate (P3)


def fault_facts(steps: list[dict]) -> dict:
    """Trace facts the cluster diff cannot show: commands run inside containers (changes to a container's
    filesystem or processes are invisible to the diff) and calls an enforcement point refused."""
    facts = {
        "in_container": [s["args"] for s in steps if _IN_CONTAINER.match(s["args"])],
        "refused": [f"{s['tool']}: {s['args']}" for s in steps if s["out"].startswith(_REFUSED)],
    }
    return {k: v for k, v in facts.items() if v}


class TraceRecorder:
    """Collects tool calls emitted while the context is active, in call order."""

    def __enter__(self):
        global _events
        _events = []
        self.steps: list[dict] = []
        return self

    def __exit__(self, *exc):
        global _events
        crewai_event_bus.flush()
        self.steps = [step for _, step in sorted(_events, key=lambda e: e[0])]
        _events = None
        return False


# ── Cluster state ─────────────────────────────────────────────────────────


def _fingerprint(item: dict) -> dict:
    kind = item["kind"]
    if kind == "Deployment":
        spec = item["spec"]
        return {
            "generation": item["metadata"].get("generation"),
            "replicas": spec.get("replicas"),
            "images": [c["image"] for c in spec["template"]["spec"]["containers"]],
        }
    if kind == "Pod":
        return {}
    body = json.dumps(item.get("spec", item.get("data")), sort_keys=True)
    return {"hash": hashlib.sha1(body.encode()).hexdigest()[:8]}


def snapshot() -> dict[str, dict]:
    raw = run_cmd(["kubectl", "get", "deploy,svc,cm,ingress,pod,servicemonitor,hpa", "-A", "-o", "json"], timeout=30)
    try:
        items = json.loads(raw)["items"]
    except (ValueError, KeyError):
        return {}
    return {
        f"{i['kind']}/{i['metadata'].get('namespace', '')}/{i['metadata']['name']}": _fingerprint(i)
        for i in items
        if "deletionTimestamp" not in i["metadata"]  # terminating = already gone
    }


def state_diff(before: dict, after: dict) -> dict:
    """{resource: "added" | "removed" | {field: [before, after]}} for changed resources only."""
    diff = {}
    for key in before.keys() | after.keys():
        if key not in before:
            diff[key] = "added"
        elif key not in after:
            diff[key] = "removed"
        elif before[key] != after[key]:
            diff[key] = {f: [before[key].get(f), after[key].get(f)]
                         for f in before[key] if before[key].get(f) != after[key].get(f)}
    return dict(sorted(diff.items()))
