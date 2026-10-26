"""
kubectl-ai evaluation wrapper.
Same output schema as eval/runner.py. Reuses all helper functions from it.

Prerequisites:
    # Download binary to ~/.local/bin (no sudo needed):
    curl -fsSL https://github.com/GoogleCloudPlatform/kubectl-ai/releases/download/v0.0.31/kubectl-ai_Linux_x86_64.tar.gz \\
        | tar xz -C ~/.local/bin kubectl-ai

Usage:
    # Ollama (only supported provider — OpenAI-compat endpoints fail due to
    # tool_choice:"auto" incompatibility and ReAct shim JSON format mismatch):
    OLLAMA_HOST=http://193.2.205.14:11434 \\
        uv run python eval/kubectl_ai_runner.py --model qwen3.5:9b-64k --provider ollama
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LITELLM_LOG", "CRITICAL")

# kubectl-ai may be in ~/.local/bin which isn't always on PATH
_LOCAL_BIN = Path.home() / ".local" / "bin"
os.environ["PATH"] = f"{_LOCAL_BIN}:{os.environ.get('PATH', '')}"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

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


def make_output_path(model: str, levels: list[int] | None) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    model_tag = model.replace(":", "-").replace(".", "-").replace("/", "-")
    levels_tag = "L" + "".join(str(l) for l in sorted(levels)) if levels else "Lall"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"results_kubectl-ai_{model_tag}_{levels_tag}_{ts}.json"


def run_kubectl_ai_task(prompt: str, model: str, provider: str, log_path: Path | None) -> dict:
    cmd = ["kubectl-ai", "--quiet", "--skip-permissions", "--model", model]
    if provider == "ollama":
        cmd += ["--llm-provider", "ollama"]
    else:
        # OpenAI-compat (HPC vLLM): use tool-use-shim because vLLM's native
        # tool-call parsers don't reliably handle kubectl-ai's complex tool
        # schemas with embedded JSON in descriptions. The shim does text-based
        # ReAct tool calling which works with any model that can follow instructions.
        cmd += ["--llm-provider", "openai", "--enable-tool-use-shim"]
    cmd.append(prompt)

    start = time.time()
    try:
        # whole task; multi-step tasks on larger models take far longer than 180 s
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=int(os.getenv("KUBECTL_AI_TIMEOUT", "1800")))
        raw = (result.stdout + result.stderr).strip()
        response = _ANSI.sub("", raw) or "(no output)"
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"--- {prompt[:60]} ---\n{raw}\n")
    except subprocess.TimeoutExpired:
        response = "Error: timeout"
    except FileNotFoundError:
        response = "Error: kubectl-ai not installed — run: curl -fsSL .../kubectl-ai_Linux_x86_64.tar.gz | tar xz -C ~/.local/bin kubectl-ai"
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


def run_evaluation(model: str, provider: str, levels: list[int] | None,
                   tasks_path: str, log_path: Path | None) -> dict:
    import json
    tasks = load_tasks(tasks_path)
    if levels:
        tasks = [t for t in tasks if t["level"] in levels]

    output_path = make_output_path(model, levels)
    levels_str = " ".join(f"L{l}" for l in sorted(levels)) if levels else "L1 L2"
    print(f"\n{'='*70}")
    print(f"  arch=kubectl-ai  model={model}  levels={levels_str}  tasks={len(tasks)}")
    print(f"{'='*70}")
    cleanup_eval_resources()

    metadata = {
        "architecture": "kubectl-ai",
        "llm_model": model,
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

        out = run_kubectl_ai_task(task["prompt"], model, provider, log_path)

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
    parser = argparse.ArgumentParser(description="kubectl-ai evaluation runner")
    parser.add_argument("--model", required=True, help="Model name (e.g. Qwen-Qwen3.5-122B-A10B)")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"],
                        help="LLM provider type (default: openai for openai-compat)")
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", default="eval/tasks.yaml")
    parser.add_argument("--debug-log", action="store_true")
    args = parser.parse_args()

    log_path = None
    if args.debug_log:
        from eval.runner import LOGS_DIR
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"debug_kubectl-ai_{ts}.log"

    run_evaluation(args.model, args.provider, args.levels, args.tasks, log_path)
