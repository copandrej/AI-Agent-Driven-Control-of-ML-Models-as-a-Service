"""
kagent evaluation wrapper.
Same output schema as eval/runner.py. Reuses all helper functions from it.

Prerequisites:
    # Install CRDs and chart:
    helm install kagent-crds oci://ghcr.io/kagent-dev/kagent/helm/kagent-crds \\
        --version 0.9.4 --namespace kagent --create-namespace
    helm install kagent oci://ghcr.io/kagent-dev/kagent/helm/kagent \\
        --version 0.9.4 --namespace kagent \\
        --set providers.default=openAI --set providers.openAI.apiKey=placeholder

    # Configure ModelConfig CRD (replaces default-model-config):
    # The Helm chart does NOT support baseUrl — must use ModelConfig CRD.
    # See research notes; needs CoreDNS host.docker.internal alias + socat bridge.

    # Install kagent CLI:
    curl -sL https://github.com/kagent-dev/kagent/releases/download/v0.9.4/kagent-linux-amd64 \\
        -o ~/.local/bin/kagent && chmod +x ~/.local/bin/kagent

    # Port-forward the controller before running this script:
    kubectl -n kagent port-forward svc/kagent-controller 8083:8083 &

Usage:
    uv run python eval/kagent_runner.py --agent k8s-agent
    uv run python eval/kagent_runner.py --agent k8s-agent --levels 1 2
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LITELLM_LOG", "CRITICAL")

# kagent may be in ~/.local/bin
_LOCAL_BIN = Path.home() / ".local" / "bin"
os.environ["PATH"] = f"{_LOCAL_BIN}:{os.environ.get('PATH', '')}"

from eval.runner import (
    RESULTS_DIR,
    check_cluster_state,
    check_response_keywords,
    cleanup_eval_resources,
    cleanup_task,
    compute_summary,
    load_tasks,
    print_running_tally,
    print_task_done,
    print_task_start,
    setup_preconditions,
    suppress_agent_output,
)


def _detect_model() -> str:
    """Read the kagent ModelConfig CRD to identify which LLM is in use."""
    try:
        out = subprocess.run(
            ["kubectl", "-n", "kagent", "get", "modelconfig", "default-model-config",
             "-o", "jsonpath={.spec.model}"],
            capture_output=True, text=True, timeout=10
        )
        m = out.stdout.strip()
        return m.replace(":", "-").replace(".", "-").replace("/", "-") or "unknown"
    except Exception:
        return "unknown"


def make_output_path(agent: str, levels: list[int] | None) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    levels_tag = "L" + "".join(str(l) for l in sorted(levels)) if levels else "Lall"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _detect_model()
    return RESULTS_DIR / f"results_kagent_{agent}_{model_tag}_{levels_tag}_{ts}.json"


def _extract_response(raw_stdout: str) -> str:
    """Parse kagent's JSON output and extract the final answer.
    The CLI returns a Task object with artifacts[0].parts[0].text being the final
    assistant message. Thinking-model responses include a </think> block we strip."""
    try:
        data = json.loads(raw_stdout)
        text = data["artifacts"][0]["parts"][0]["text"]
        # Strip everything up to and including </think>
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()
    except (json.JSONDecodeError, KeyError, IndexError):
        return raw_stdout.strip()


def run_kagent_task(prompt: str, agent: str, log_path: Path | None) -> dict:
    cmd = ["kagent", "invoke", "-t", prompt, "--agent", agent, "-o", "json", "--timeout", "180s"]
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"--- {prompt[:60]} ---\nstdout: {result.stdout}\nstderr: {result.stderr}\n")
        if result.returncode != 0 and not result.stdout.strip():
            response = f"Error: kagent CLI returned {result.returncode}: {result.stderr.strip()[:200]}"
        else:
            response = _extract_response(result.stdout) or "(no output)"
    except subprocess.TimeoutExpired:
        response = "Error: timeout"
    except FileNotFoundError:
        response = "Error: kagent CLI not installed"
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


def run_evaluation(agent: str, levels: list[int] | None,
                   tasks_path: str, log_path: Path | None) -> dict:
    tasks = load_tasks(tasks_path)
    if levels:
        tasks = [t for t in tasks if t["level"] in levels]

    output_path = make_output_path(agent, levels)
    levels_str = " ".join(f"L{l}" for l in sorted(levels)) if levels else "L1 L2"
    print(f"\n{'='*70}")
    print(f"  arch=kagent  agent={agent}  levels={levels_str}  tasks={len(tasks)}")
    print(f"{'='*70}")
    cleanup_eval_resources()

    metadata = {
        "architecture": "kagent",
        "llm_model": agent,  # agent name encodes the model config
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks_file": tasks_path,
        "levels": levels or [1, 2],
        "status": "in_progress",
    }

    start_time = time.time()
    results = []
    failed = 0

    for i, task in enumerate(tasks, start=1):
        print_task_start(i, len(tasks), task["id"], task["description"])
        cleanup_task(task)
        setup_preconditions(task, log_path)

        out = run_kagent_task(task["prompt"], agent, log_path)

        kw = check_response_keywords(out["response"], task["success_criteria"]["response_keywords"])
        cl = check_cluster_state(task["success_criteria"].get("cluster_check"), out["response"])
        success = kw and (cl is None or cl["passed"]) and not out["response"].startswith("Error:")

        print_task_done(i, len(tasks), task["id"], task["description"], success, out["elapsed"])
        cleanup_task(task)

        record = {
            "task_id": task["id"], "level": task["level"], "category": task["category"],
            "description": task["description"], "success": success,
            "elapsed_seconds": out["elapsed"], "response": out["response"][:2000],
        }
        if cl is not None:
            record["cluster_check"] = cl
        if not success:
            record["error"] = out["response"][:300]
        results.append(record)
        if not success:
            failed += 1

        summary = compute_summary(results)
        duration = round(time.time() - start_time, 1)
        payload = {"metadata": {**metadata, "total_duration_s": duration, "status": "in_progress"},
                   "summary": summary, "tasks": results}
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(output_path)
        print_running_tally(i, len(tasks), failed)

    duration = round(time.time() - start_time, 1)
    summary = compute_summary(results)
    payload = {"metadata": {**metadata, "total_duration_s": duration, "status": "complete"},
               "summary": summary, "tasks": results}
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  RESULT  {summary['passed']}/{summary['total_tasks']} ({summary['success_rate']}%)  |  {duration}s")
    print(f"  Results → {output_path}\n{'='*70}\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kagent evaluation runner")
    parser.add_argument("--agent", required=True, help="kagent agent name (must be deployed in cluster)")
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", default="eval/tasks.yaml")
    parser.add_argument("--debug-log", action="store_true")
    args = parser.parse_args()

    log_path = None
    if args.debug_log:
        from eval.runner import LOGS_DIR
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"debug_kagent_{ts}.log"

    run_evaluation(args.agent, args.levels, args.tasks, log_path)
