"""
Evaluation Runner
==================
Runs benchmark tasks against flow/simple/minimal/dual architectures.
All agent output is suppressed by default; use --debug-log to capture it.

Usage:
    # Full benchmark — all architectures, all levels
    uv run python eval/runner.py --archs flow simple minimal dual

    # Specific architecture and levels
    uv run python eval/runner.py --archs flow --levels 1 2
    uv run python eval/runner.py --archs flow --levels 2

    # Enable debug logging (writes verbose agent output to log file)
    uv run python eval/runner.py --archs flow --debug-log

Output:
    Results → eval/results/results_{arch}_{model}_{levels}_{timestamp}.json
    Calls   → eval/results/calls_{arch}_{model}_{levels}_{timestamp}.jsonl  (one line per LLM call;
              --full-calls adds reasoning text and input messages)
    Debug   → eval/results/logs/debug_{arch}_{model}_{levels}_{timestamp}.log
"""

import argparse
import contextlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

# Suppress all library log noise before importing anything else
os.environ.setdefault("LITELLM_LOG", "CRITICAL")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

for _name in ["crewai", "litellm", "httpx", "openai", "chromadb", "httpcore", "urllib3"]:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False

from src import tools as agent_tools
from src.agent import MlOpsFlow, resolve_llm, run_cmd
from eval import telemetry
from eval.task_trace import TraceRecorder, fault_facts, snapshot, state_diff

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR", Path(__file__).parent / "results"))
LOGS_DIR = RESULTS_DIR / "logs"


def make_output_paths(
    architecture: str,
    model: str,
    levels: list[int] | None,
    debug_log: bool,
) -> tuple[Path, Path | None]:
    RESULTS_DIR.mkdir(exist_ok=True)
    model_tag = model.split("/")[-1].replace(":", "-").replace(".", "-")
    levels_tag = "L" + "".join(str(l) for l in sorted(levels)) if levels else "Lall"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{architecture}_{model_tag}_{levels_tag}_{ts}"
    result_path = RESULTS_DIR / f"results_{stem}.json"
    log_path = None
    if debug_log:
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / f"debug_{stem}.log"
    return result_path, log_path


# ---------------------------------------------------------------------------
# Output suppression
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_agent_output(log_path: Path | None = None):
    """Redirect stdout/stderr during agent execution.

    With log_path: writes all output to that file (useful for post-run debug).
    Without: discards output silently.
    """
    target = open(log_path, "a") if log_path else open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(target), contextlib.redirect_stderr(target):
            yield
    finally:
        target.close()


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(path: str = "eval/tasks.yaml") -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["tasks"]


# ---------------------------------------------------------------------------
# Setup / cleanup
# ---------------------------------------------------------------------------

def setup_preconditions(task: dict, log_path: Path | None) -> None:
    for cmd in task.get("setup_commands", []):
        result = run_cmd(cmd, timeout=90)  # traffic setups wait on predictions that take seconds each
        if "error" in result.lower() and "not found" not in result.lower():
            msg = f"[setup warning] {task['id']}: {cmd} -> {result}\n"
            if log_path:
                with open(log_path, "a") as f:
                    f.write(msg)


def cleanup_task(task: dict) -> None:
    for cmd in task.get("cleanup_commands", []):
        run_cmd(cmd, timeout=30)


# Infrastructure no task should change; re-applied after each task if an agent changed it
BASELINE_MANIFESTS = ["k8s/ml_service/servicemonitor.yaml"]


def restore_baseline() -> list[str]:
    restored = []
    for path in BASELINE_MANIFESTS:
        try:
            drifted = subprocess.run(["kubectl", "diff", "-f", path], capture_output=True, timeout=30).returncode == 1
        except subprocess.TimeoutExpired:
            drifted = True
        if drifted:
            run_cmd(["kubectl", "apply", "-f", path], timeout=30)
            restored.append(path)
    return restored


def cleanup_eval_resources() -> None:
    """Remove leftover eval resources"""
    keep_prefixes = ("monitoring-stack", "prometheus-")
    keep_exact = {"echo-service", "ml-service", "kubernetes"}
    for kind in ("deployment", "service"):
        out = run_cmd(f"kubectl get {kind} -n default -o name --no-headers", timeout=10)
        for line in out.splitlines():
            name = line.split("/")[-1]
            if not name or name in keep_exact:
                continue
            if any(name.startswith(p) for p in keep_prefixes):
                continue
            run_cmd(f"kubectl delete {kind} {name} -n default --ignore-not-found --wait=false", timeout=15)
    # Bare pods (e.g. an agent's `kubectl run redis`) have no owner, so deleting deployments never removes them
    try:
        pods = json.loads(run_cmd("kubectl get pods -n default -o json", timeout=10))["items"]
    except (ValueError, KeyError):
        pods = []  # kubectl failed: delete nothing
    for pod in pods:
        if not pod["metadata"].get("ownerReferences"):
            run_cmd(f"kubectl delete pod {pod['metadata']['name']} -n default --ignore-not-found --wait=false", timeout=15)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_response_keywords(response: str, keywords: list[str]) -> bool:
    lower = response.lower()
    return any(kw.lower() in lower for kw in keywords)


def response_numbers(response: str, time_only: bool = False) -> list[float]:
    """Standalone numbers in the response (not parts of IPs, pod hashes or "P95"); ms are converted to s.
    time_only keeps only numbers written with a time unit, so "3 minutes" or a bare "3" do not count as 3 s."""
    found = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|secs?|seconds)?\b", response, re.IGNORECASE)
    return [float(n) / (1000 if unit.lower().startswith("m") else 1) for n, unit in found if unit or not time_only]


def check_cluster_state(cluster_check: dict | None, response: str = "") -> dict | None:
    if not cluster_check:
        return None
    cmd = cluster_check["command"]
    actual = run_cmd(cmd, timeout=cluster_check.get("timeout", 60))  # checks may wait for a rollout before reading state
    actual_norm = "" if actual.strip() in ("(no output)", "(no resources found)", "") else actual.strip()
    if cluster_check.get("in_response"):
        # Verify the actual cluster value appears verbatim in the agent's response.
        # Catches hallucination: agent must repeat the real value, not invent one.
        passed = bool(actual_norm) and actual_norm.lower() in response.lower()
        return {"passed": passed, "expected": f"(in response) {actual_norm}", "actual": actual_norm}
    if "tolerance" in cluster_check:
        # Numeric ground truth (e.g. a Prometheus value): the answer must report it within the relative
        # tolerance. Refusals and "no data" answers contain no such number.
        try:
            truth = float(actual_norm)
        except ValueError:
            truth = float("nan")
        tol = cluster_check["tolerance"]
        numbers = response_numbers(response, time_only=cluster_check.get("unit") == "s")
        passed = any(abs(x - truth) <= tol * truth for x in numbers)
        return {"passed": passed, "expected": f"(in response ±{tol:.0%}) {actual_norm}", "actual": actual_norm}
    expected = cluster_check["expected"]
    passed = expected.lower() in actual.lower() if expected else actual_norm == ""
    return {"passed": passed, "expected": expected, "actual": actual_norm}


# ---------------------------------------------------------------------------
# Architecture runners
# ---------------------------------------------------------------------------

def run_flow_task(llm, prompt: str, log_path: Path | None) -> dict:
    start = time.time()
    flow = None
    try:
        with suppress_agent_output(log_path):
            flow = MlOpsFlow(llm=llm, memory=None)
            flow.kickoff(inputs={"user_input": prompt, "history_context": ""})
        response = flow.state.final_response or "(no response)"
    except Exception as e:
        response = f"Error: {e}"
    route = flow.state.intent if flow else None
    return {"response": response, "elapsed": round(time.time() - start, 2), "route": route}


def run_simple_task(llm, prompt: str, log_path: Path | None) -> dict:
    from src.simple import run_simple
    start = time.time()
    try:
        with suppress_agent_output(log_path):
            response = run_simple(llm, prompt, "")
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


def run_minimal_task(llm, prompt: str, log_path: Path | None) -> dict:
    from src.minimal import run_minimal
    start = time.time()
    try:
        with suppress_agent_output(log_path):
            response = run_minimal(llm, prompt, "")
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


def run_dual_task(llm, prompt: str, log_path: Path | None) -> dict:
    from src.dual import run_dual
    start = time.time()
    try:
        with suppress_agent_output(log_path):
            response = run_dual(llm, prompt, "")
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


def run_bare_task(llm, prompt: str, log_path: Path | None) -> dict:
    from src.bare import run_bare
    start = time.time()
    try:
        with suppress_agent_output(log_path):
            response = run_bare(llm, prompt, "")
    except Exception as e:
        response = f"Error: {e}"
    return {"response": response, "elapsed": round(time.time() - start, 2)}


_RUNNERS = {
    "flow": run_flow_task,
    "simple": run_simple_task,
    "minimal": run_minimal_task,
    "dual": run_dual_task,
    "bare": run_bare_task,
}

P4 = os.getenv("P4", "off").lower()  # off | detect | feedback (src/verify.py)


def run_task(llm, architecture: str, prompt: str, log_path: Path | None) -> dict:
    """The architecture's runner, wrapped in P4 goal verification when P4 is on."""
    agent_tools.P3_REQUEST = prompt  # the P3 gate lets through deletes this request asks for
    if P4 not in ("detect", "feedback"):
        return _RUNNERS[architecture](llm, prompt, log_path)
    from src.verify import run_verified
    start, last = time.time(), {}

    def execute(p: str) -> str:
        last.update(_RUNNERS[architecture](llm, p, log_path))
        return last["response"]

    with suppress_agent_output(log_path):
        response, rounds = run_verified(execute, llm, prompt, feedback=P4 == "feedback")
    return {**last, "response": response, "elapsed": round(time.time() - start, 2), "p4": rounds}


# ---------------------------------------------------------------------------
# Terminal progress display
# ---------------------------------------------------------------------------

def _task_line(idx: int, total: int, task_id: str, description: str) -> str:
    desc = f"{description[:48]:<48}"
    return f"  [{idx:2d}/{total}] {task_id:<6}  {desc}"


def print_task_start(idx: int, total: int, task_id: str, description: str) -> None:
    line = _task_line(idx, total, task_id, description)
    sys.stdout.write(f"{line}  running...\r")
    sys.stdout.flush()


def print_task_done(idx: int, total: int, task_id: str, description: str, success: bool, elapsed: float) -> None:
    icon = "✓" if success else "✗"
    status = "PASS" if success else "FAIL"
    line = _task_line(idx, total, task_id, description)
    sys.stdout.write(f"{line}  {icon} {status}  ({elapsed:.0f}s)\n")
    sys.stdout.flush()


def print_running_tally(done: int, total: int, failed: int) -> None:
    passed = done - failed
    pct = int(100 * done / total) if total else 0
    print(f"  ── {done}/{total} done  {pct}%  ✓{passed} passed  ✗{failed} failed")


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_single_task(
    llm,
    task: dict,
    architecture: str,
    idx: int,
    total: int,
    log_path: Path | None,
) -> dict:
    task_id = task["id"]
    description = task["description"]

    print_task_start(idx, total, task_id, description)

    cleanup_task(task)
    setup_preconditions(task, log_path)

    before = snapshot()
    counters = telemetry.vllm_counters()
    telemetry.set_task(task_id)
    started_at = datetime.now(timezone.utc).isoformat()
    with TraceRecorder() as trace:
        result = run_task(llm, architecture, task["prompt"], log_path)
    finished_at = datetime.now(timezone.utc).isoformat()
    context = telemetry.context_stats()
    telemetry.set_task(None)
    vllm = telemetry.counter_delta(counters, telemetry.vllm_counters())
    diff = state_diff(before, snapshot())

    criteria = task["success_criteria"]
    kw_passed = check_response_keywords(result["response"], criteria["response_keywords"])
    cl_check = check_cluster_state(criteria.get("cluster_check"), result["response"])

    success = (
        kw_passed
        and (cl_check is None or cl_check["passed"])
        and not result["response"].startswith("Error:")
        and "[LLM returned empty response" not in result["response"]
    )

    print_task_done(idx, total, task_id, description, success, result["elapsed"])

    cleanup_task(task)
    restored = restore_baseline()
    if restored:
        print(f"  [baseline drift] {task_id} changed {', '.join(restored)} -> restored")

    record = {
        "task_id": task_id,
        "level": task["level"],
        "category": task["category"],
        "description": description,
        "success": success,
        "elapsed_seconds": result["elapsed"],
        "started_at": started_at,
        "finished_at": finished_at,
        "stop": ("error" if result["response"].startswith("Error:")
                 else "empty" if result["response"].startswith("[Agent produced empty response") else "answer"),
        "vllm": vllm,
        "context": context,
        "response": result["response"][:2000],
    }
    # Only emit cluster_check when it was actually run (non-null) — it is the ground-truth
    # signal for write tasks. Null means the task was read-only with no state to verify.
    if cl_check is not None:
        record["cluster_check"] = cl_check
    # Only emit error details on failure
    if not success:
        record["error"] = result["response"][:300]
    # Compact fault-analysis facts: route (flow only), tool calls, cluster state changes
    if "route" in result:
        record["route"] = result["route"]
    if "p4" in result:
        record["p4"] = result["p4"]
    record["trace"] = trace.steps
    record["diff"] = diff
    if restored:
        record["baseline_restored"] = restored
    record.update(fault_facts(trace.steps))
    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total_tasks": 0}

    successes = sum(1 for r in results if r["success"])

    def level_stats(level: int) -> dict | None:
        subset = [r for r in results if r["level"] == level]
        if not subset:
            return None
        s = sum(1 for r in subset if r["success"])
        return {
            "total": len(subset),
            "passed": s,
            "success_rate": round(100 * s / len(subset), 1),
            "avg_elapsed_s": round(sum(r["elapsed_seconds"] for r in subset) / len(subset), 1),
        }

    def category_stats(cat: str) -> dict | None:
        subset = [r for r in results if r["category"] == cat]
        if not subset:
            return None
        s = sum(1 for r in subset if r["success"])
        return {
            "total": len(subset),
            "passed": s,
            "success_rate": round(100 * s / len(subset), 1),
        }

    by_level = {lvl: stats for lvl in (1, 2) if (stats := level_stats(lvl))}
    by_category = {cat: stats for cat in ("general_k8s", "ml_specific") if (stats := category_stats(cat))}

    return {
        "total_tasks": total,
        "passed": successes,
        "success_rate": round(100 * successes / total, 1),
        "avg_elapsed_s": round(sum(r["elapsed_seconds"] for r in results) / total, 1),
        "by_level": by_level,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    llm,
    architecture: str,
    levels: list[int] | None = None,
    tasks_path: str = "eval/tasks.yaml",
    output_path: Path | None = None,
    log_path: Path | None = None,
    full_calls: bool = False,
    task_ids: list[str] | None = None,
) -> dict:
    tasks = load_tasks(tasks_path)
    if levels:
        tasks = [t for t in tasks if t["level"] in levels]
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if os.getenv("TOOLSET", "core").lower() != "platform":  # tasks that need the platform tools
        tasks = [t for t in tasks if t.get("toolset") != "platform"]

    levels_str = " ".join(f"L{l}" for l in sorted({t["level"] for t in tasks}))
    print(f"\n{'='*70}")
    print(f"  arch={architecture}  model={llm.model}  levels={levels_str}  tasks={len(tasks)}")
    if log_path:
        print(f"  debug log → {log_path}")
    print(f"{'='*70}")
    print("  Cleaning up leftover eval resources...")
    cleanup_eval_resources()
    for path in restore_baseline():
        print(f"  [baseline drift] {path} differed from the cluster before the run -> restored")
    print()

    calls_path = output_path.with_name(output_path.name.replace("results_", "calls_", 1)).with_suffix(".jsonl")
    telemetry.start(calls_path, full=full_calls)

    # Build the metadata block once — reused on every incremental write
    metadata = {
        "architecture": architecture,
        "llm_model": llm.model,
        **telemetry.server_info(),
        "temperature": llm.temperature,
        # principle switches and budgets (env), so every result file says which arm it is
        "config": {k: os.getenv(k, "") for k in ("P3_GATE", "P4", "TOOLSET", "MAX_ITER", "LLM_TIMEOUT")},
        "git_commit": run_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "calls_file": calls_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks_file": tasks_path,
        "levels": sorted({t["level"] for t in tasks}),
        "host": socket.gethostname(),
        "task_ids": task_ids,
        "log_file": str(log_path) if log_path else None,
        "status": "in_progress",
    }

    def flush_results(current_results: list[dict], duration_s: float, final: bool = False) -> None:
        """Write current partial (or final) results to disk atomically."""
        payload = {
            "metadata": {**metadata, "total_duration_s": duration_s, "status": "complete" if final else "in_progress"},
            "summary": compute_summary(current_results),
            "tasks": current_results,
        }
        # Write to a temp file then rename for atomic update
        tmp = output_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(output_path)

    start_time = time.time()
    results = []
    failed = 0

    # Write an empty skeleton immediately so the file exists from task 1
    flush_results([], 0.0)

    for i, task in enumerate(tasks, start=1):
        result = run_single_task(llm, task, architecture, i, len(tasks), log_path)
        results.append(result)
        if not result["success"]:
            failed += 1
        flush_results(results, round(time.time() - start_time, 1))
        print_running_tally(i, len(tasks), failed)

    total_duration = round(time.time() - start_time, 1)
    flush_results(results, total_duration, final=True)
    summary = compute_summary(results)

    print(f"\n  RESULT  {summary['passed']}/{summary['total_tasks']} ({summary['success_rate']}%)  |  {total_duration}s total")
    for lvl, ls in summary["by_level"].items():
        print(f"    L{lvl}: {ls['passed']}/{ls['total']} ({ls['success_rate']}%)  avg {ls['avg_elapsed_s']}s/task")
    print(f"  Results → {output_path}")
    print(f"{'='*70}\n")

    return {"metadata": {**metadata, "total_duration_s": total_duration, "status": "complete"}, "summary": summary, "tasks": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_rate(val) -> str:
    return f"{val}%" if isinstance(val, (int, float)) else "-"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLOps Agent Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick commands for ablation study:
  # All architectures, all levels (full benchmark)
  uv run python eval/runner.py --archs flow simple minimal dual

  # L1 only (fast, no ML service needed)
  uv run python eval/runner.py --archs flow simple minimal --levels 1

  # L2 only (ML inference tasks)
  uv run python eval/runner.py --archs flow simple minimal --levels 2

  # Flow only with debug log
  uv run python eval/runner.py --archs flow --debug-log

  # Quick single-arch smoke test
  uv run python eval/runner.py --archs minimal --levels 1
        """,
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        choices=list(_RUNNERS),
        default=["flow"],
        metavar="ARCH",
        help="Architectures to evaluate: flow simple minimal dual bare (default: flow)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Complexity levels: 1 2 (default: all)",
    )
    parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Write verbose agent output to eval/results/logs/ for debugging",
    )
    parser.add_argument(
        "--full-calls",
        action="store_true",
        help="Also store reasoning text and input messages per LLM call in the calls JSONL (larger files)",
    )
    parser.add_argument(
        "--tasks",
        default="eval/tasks.yaml",
        help="Path to task definitions YAML (default: eval/tasks.yaml)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Run only these task ids, e.g. L2-05 L2-06 (default: all)",
    )
    args = parser.parse_args()

    llm = resolve_llm()
    all_outputs: dict[str, dict] = {}

    for arch in args.archs:
        result_path, log_path = make_output_paths(arch, llm.model, args.levels, args.debug_log)
        output = run_evaluation(
            llm=llm,
            architecture=arch,
            levels=args.levels,
            tasks_path=args.tasks,
            output_path=result_path,
            log_path=log_path,
            full_calls=args.full_calls,
            task_ids=args.task_ids,
        )
        all_outputs[arch] = output

    # Multi-arch comparison table
    if len(args.archs) > 1:
        print(f"\n{'='*70}")
        print("  ARCHITECTURE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Architecture':<14} {'Overall':>8} {'L1':>7} {'L2':>7}  {'Avg/task':>9}")
        print(f"  {'-'*52}")
        for arch, out in all_outputs.items():
            s = out["summary"]
            l1 = s["by_level"].get(1, {}).get("success_rate", "-")
            l2 = s["by_level"].get(2, {}).get("success_rate", "-")
            print(
                f"  {arch:<14} {_fmt_rate(s['success_rate']):>8}"
                f" {_fmt_rate(l1):>7} {_fmt_rate(l2):>7}"
                f"  {s['avg_elapsed_s']:>7.1f}s"
            )
        print(f"{'='*70}\n")
