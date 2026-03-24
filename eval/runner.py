"""
Evaluation Runner for MLOps Agent
===================================
Automated benchmark harness that executes eval/tasks.yaml against
the Flow, Simple, or Minimal architecture and produces structured results.

Usage:
    uv run python eval/runner.py --tasks eval/tasks.yaml --arch flow
    uv run python eval/runner.py --arch simple --levels 1 2
    uv run python eval/runner.py --arch minimal --output results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from src.agent import MlOpsFlow, build_context, resolve_llm, run_cmd


def load_tasks(path: str = "eval/tasks.yaml") -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["tasks"]


def setup_preconditions(task: dict) -> bool:
    for cmd in task.get("setup_commands", []):
        result = run_cmd(cmd, timeout=30)
        if "error" in result.lower() and "not found" not in result.lower():
            print(f"  Setup warning: {cmd} -> {result}")
    return True


def cleanup_task(task: dict) -> None:
    for cmd in task.get("cleanup_commands", []):
        run_cmd(cmd, timeout=30)


def check_response_keywords(response: str, keywords: list[str]) -> dict:
    lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return {"passed": len(matched) > 0, "matched": matched, "total": len(keywords)}


def check_cluster_state(cluster_check: dict | None) -> dict | None:
    if not cluster_check:
        return None
    cmd = cluster_check["command"]
    expected = cluster_check["expected"]
    actual = run_cmd(cmd, timeout=15)
    passed = expected.lower() in actual.lower() if expected else actual.strip() == ""
    return {"passed": passed, "expected": expected, "actual": actual.strip()}


def run_flow_task(llm, prompt: str, history_context: str = "") -> dict:
    flow = MlOpsFlow(llm=llm, memory=None)
    start = time.time()
    try:
        flow.kickoff(inputs={"user_input": prompt, "history_context": history_context})
        response = flow.state.final_response or "(no response)"
        intent = flow.state.intent
    except Exception as e:
        response = f"Error: {e}"
        intent = "error"
    elapsed = time.time() - start
    return {"response": response, "intent": intent, "elapsed": round(elapsed, 2)}


def run_simple_task(llm, prompt: str, history_context: str = "") -> dict:
    from src.simple import run_simple
    start = time.time()
    try:
        response = run_simple(llm, prompt, history_context)
        intent = "single_agent"
    except Exception as e:
        response = f"Error: {e}"
        intent = "error"
    elapsed = time.time() - start
    return {"response": response, "intent": intent, "elapsed": round(elapsed, 2)}


def run_minimal_task(llm, prompt: str, history_context: str = "") -> dict:
    from src.minimal import run_minimal
    start = time.time()
    try:
        response = run_minimal(llm, prompt, history_context)
        intent = "minimal"
    except Exception as e:
        response = f"Error: {e}"
        intent = "error"
    elapsed = time.time() - start
    return {"response": response, "intent": intent, "elapsed": round(elapsed, 2)}


def run_single_task(llm, task: dict, architecture: str = "flow") -> dict:
    task_id = task["id"]
    print(f"\n{'='*60}")
    print(f"Task {task_id}: {task['description']} [{architecture}]")
    print(f"{'='*60}")

    setup_preconditions(task)

    runners = {"flow": run_flow_task, "simple": run_simple_task, "minimal": run_minimal_task}
    result = runners[architecture](llm, task["prompt"])

    criteria = task["success_criteria"]
    kw_check = check_response_keywords(result["response"], criteria["response_keywords"])
    cl_check = check_cluster_state(criteria.get("cluster_check"))

    intent_correct = (
        result["intent"] == task["expected_intent"]
        if architecture == "flow"
        else True  # simple/minimal have no intent routing
    )

    success = (
        kw_check["passed"]
        and (cl_check is None or cl_check["passed"])
        and intent_correct
        and "Error:" not in result["response"][:20]
    )

    status = "PASS" if success else "FAIL"
    print(f"  Intent: {result['intent']} (expected: {task['expected_intent']}) {'OK' if intent_correct else 'MISS'}")
    print(f"  Keywords: {kw_check['matched']} {'OK' if kw_check['passed'] else 'MISS'}")
    if cl_check:
        print(f"  Cluster: {cl_check['actual'][:80]} {'OK' if cl_check['passed'] else 'MISS'}")
    print(f"  Elapsed: {result['elapsed']}s")
    print(f"  Result: {status}")

    cleanup_task(task)

    return {
        "task_id": task_id,
        "level": task["level"],
        "category": task["category"],
        "description": task["description"],
        "prompt": task["prompt"],
        "architecture": architecture,
        "intent_expected": task["expected_intent"],
        "intent_actual": result["intent"],
        "intent_correct": intent_correct,
        "response": result["response"][:2000],
        "response_check": kw_check,
        "cluster_check": cl_check,
        "success": success,
        "elapsed_seconds": result["elapsed"],
        "manual_commands_saved": task.get("manual_commands_saved", 0),
        "manual_lines_saved": task.get("manual_lines_saved", 0),
        "comparable_to": task.get("comparable_to", []),
        "error": None if success else result["response"][:200],
    }


def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total_tasks": 0}

    successes = sum(1 for r in results if r["success"])
    intent_correct = sum(1 for r in results if r["intent_correct"])

    def level_stats(level: int) -> dict:
        subset = [r for r in results if r["level"] == level]
        if not subset:
            return {"total": 0, "success_rate": 0.0}
        s = sum(1 for r in subset if r["success"])
        return {
            "total": len(subset),
            "passed": s,
            "success_rate": round(100 * s / len(subset), 1),
            "avg_elapsed": round(sum(r["elapsed_seconds"] for r in subset) / len(subset), 2),
        }

    def category_stats(cat: str) -> dict:
        subset = [r for r in results if r["category"] == cat]
        if not subset:
            return {"total": 0, "success_rate": 0.0}
        s = sum(1 for r in subset if r["success"])
        return {
            "total": len(subset),
            "passed": s,
            "success_rate": round(100 * s / len(subset), 1),
        }

    return {
        "total_tasks": total,
        "passed": successes,
        "success_rate": round(100 * successes / total, 1),
        "intent_accuracy": round(100 * intent_correct / total, 1),
        "avg_elapsed": round(sum(r["elapsed_seconds"] for r in results) / total, 2),
        "by_level": {1: level_stats(1), 2: level_stats(2), 3: level_stats(3)},
        "by_category": {
            "general_k8s": category_stats("general_k8s"),
            "ml_specific": category_stats("ml_specific"),
        },
        "total_manual_commands_saved": sum(
            r["manual_commands_saved"] for r in results if r["success"]
        ),
        "total_manual_lines_saved": sum(
            r["manual_lines_saved"] for r in results if r["success"]
        ),
    }


def run_evaluation(
    tasks_path: str = "eval/tasks.yaml",
    architecture: str = "flow",
    output_path: str | None = None,
    levels: list[int] | None = None,
) -> dict:
    llm = resolve_llm()
    tasks = load_tasks(tasks_path)

    if levels:
        tasks = [t for t in tasks if t["level"] in levels]

    print(f"\nEvaluation: {len(tasks)} tasks, architecture={architecture}")
    print(f"LLM: {llm.model}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []
    for task in tasks:
        result = run_single_task(llm, task, architecture)
        results.append(result)

    total_duration = round(time.time() - start_time, 2)
    summary = compute_summary(results)

    output = {
        "metadata": {
            "architecture": architecture,
            "llm_model": llm.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": total_duration,
            "tasks_file": tasks_path,
        },
        "summary": summary,
        "tasks": results,
    }

    if not output_path:
        model_tag = llm.model.replace("/", "_").replace(":", "_")
        output_path = f"eval_results_{architecture}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Architecture: {architecture}")
    print(f"LLM: {llm.model}")
    print(f"Total: {summary['total_tasks']} tasks, {summary['passed']} passed ({summary['success_rate']}%)")
    print(f"Intent accuracy: {summary['intent_accuracy']}%")
    print(f"Avg elapsed: {summary['avg_elapsed']}s per task")
    for lvl in (1, 2, 3):
        ls = summary["by_level"].get(lvl, {})
        if ls.get("total", 0) > 0:
            print(f"  L{lvl}: {ls['passed']}/{ls['total']} ({ls['success_rate']}%)")
    print(f"Total duration: {total_duration}s")
    print(f"Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Agent Evaluation Runner")
    parser.add_argument("--tasks", default="eval/tasks.yaml", help="Path to task definitions")
    parser.add_argument("--arch", choices=["flow", "simple", "minimal"], default="flow",
                        help="Architecture to evaluate")
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Filter by complexity level (1 2 3)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    run_evaluation(
        tasks_path=args.tasks,
        architecture=args.arch,
        output_path=args.output,
        levels=args.levels,
    )
