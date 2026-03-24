"""
Autonomous Watch Mode (Level 3)
================================
Periodic monitoring loop that checks cluster and ML service health,
detects issues, and takes corrective action autonomously.

Two-tier decision system:
  Tier 1: Deterministic rules (no LLM, fast) for common issues
  Tier 2: LLM-based decisions via MlOpsFlow for ambiguous cases

Usage:
    uv run python watch.py --interval 30 --dry-run
    uv run python watch.py --interval 15 --latency-threshold 0.1 --max-replicas 5
    uv run python watch.py --log watch_log.jsonl
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from .agent import run_cmd

DEFAULT_INTERVAL = 30
DEFAULT_LATENCY_THRESHOLD = 0.1
DEFAULT_MAX_REPLICAS = 5
DEFAULT_LOG_FILE = "watch_log.jsonl"

SERVICES = [
    {"name": "mnist-service", "namespace": "default", "health_url": "http://localhost:30081/health", "metrics_url": "http://localhost:30081/metrics"},
]


def log_decision(decision: dict, log_file: str) -> None:
    with open(log_file, "a") as f:
        f.write(json.dumps(decision) + "\n")


def make_decision(check_type: str, observation: str, action: str, action_type: str,
                  result: str, tier: str, executed: bool, elapsed: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "check_type": check_type,
        "observation": observation,
        "action": action,
        "action_type": action_type,
        "result": result,
        "decision_tier": tier,
        "was_executed": executed,
        "elapsed_seconds": round(elapsed, 2),
    }


def check_pod_health(namespace: str = "default") -> list[dict]:
    out = run_cmd(f"kubectl get pods -n {namespace} --no-headers")
    issues = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name, ready, status = parts[0], parts[1], parts[2]
        if status in ("CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull"):
            issues.append({"pod": name, "status": status, "namespace": namespace})
        elif status == "Running" and "/" in ready:
            current, total = ready.split("/")
            if current != total:
                issues.append({"pod": name, "status": f"NotReady ({ready})", "namespace": namespace})
    return issues


def check_service_health(url: str) -> dict:
    start = time.time()
    try:
        with urlopen(Request(url), timeout=10) as r:
            body = r.read().decode()
            latency = time.time() - start
            return {"healthy": True, "response": body[:500], "latency_ms": round(latency * 1000, 1)}
    except Exception as e:
        latency = time.time() - start
        return {"healthy": False, "response": str(e), "latency_ms": round(latency * 1000, 1)}


def parse_p95_from_metrics(metrics_text: str, metric_name: str = "mnist_request_latency_seconds") -> float | None:
    """Estimate P95 from Prometheus histogram buckets scraped from /metrics."""
    buckets = []
    for line in metrics_text.splitlines():
        if line.startswith(f"{metric_name}_bucket"):
            match = re.search(r'le="([^"]+)".*\}\s+(\d+\.?\d*)', line)
            if match and match.group(1) != "+Inf":
                buckets.append((float(match.group(1)), float(match.group(2))))
    if not buckets:
        return None
    buckets.sort()
    total = buckets[-1][1] if buckets else 0
    if total == 0:
        return None
    target = total * 0.95
    for bound, count in buckets:
        if count >= target:
            return bound
    return buckets[-1][0]


def check_latency(metrics_url: str, threshold: float) -> dict:
    try:
        with urlopen(Request(metrics_url), timeout=10) as r:
            text = r.read().decode()
        p95 = parse_p95_from_metrics(text)
        if p95 is None:
            return {"exceeded": False, "p95": None, "threshold": threshold, "note": "no data"}
        return {"exceeded": p95 > threshold, "p95": round(p95, 4), "threshold": threshold}
    except Exception as e:
        return {"exceeded": False, "p95": None, "threshold": threshold, "note": str(e)}


def handle_pod_issue(issue: dict, dry_run: bool) -> dict:
    start = time.time()
    pod, status, ns = issue["pod"], issue["status"], issue["namespace"]
    observation = f"Pod {pod} in {ns} is {status}"

    if status in ("CrashLoopBackOff", "Error"):
        action = f"kubectl delete pod {pod} -n {ns}"
        action_type = "restart"
    elif status in ("ImagePullBackOff", "ErrImagePull"):
        action = f"kubectl describe pod {pod} -n {ns}"
        action_type = "diagnose"
    else:
        action = f"kubectl describe pod {pod} -n {ns}"
        action_type = "diagnose"

    if dry_run:
        result = "(dry run)"
        executed = False
    else:
        result = run_cmd(action, timeout=15)
        executed = True

    return make_decision("pod_health", observation, action, action_type,
                         result[:500], "deterministic", executed, time.time() - start)


def handle_high_latency(service: dict, latency_info: dict, max_replicas: int, dry_run: bool) -> dict | None:
    if not latency_info["exceeded"]:
        return None
    start = time.time()
    name, ns = service["name"], service["namespace"]
    observation = f"P95 latency {latency_info['p95']}s exceeds threshold {latency_info['threshold']}s for {name}"

    current = run_cmd(f"kubectl get deployment {name} -n {ns} -o jsonpath='{{.spec.replicas}}'")
    try:
        current_replicas = int(current)
    except ValueError:
        return make_decision("latency", observation, "none", "none",
                             f"Cannot read replicas: {current}", "deterministic", False, time.time() - start)

    if current_replicas >= max_replicas:
        return make_decision("latency", observation, "none", "at_max_replicas",
                             f"Already at max replicas ({max_replicas})", "deterministic", False, time.time() - start)

    new_replicas = current_replicas + 1
    action = f"kubectl scale deployment {name} -n {ns} --replicas={new_replicas}"
    if dry_run:
        result = "(dry run)"
        executed = False
    else:
        result = run_cmd(action, timeout=15)
        executed = True

    return make_decision("latency", observation, action, "scale_up",
                         result[:500], "deterministic", executed, time.time() - start)


def handle_service_down(service: dict, health_info: dict, dry_run: bool) -> dict:
    start = time.time()
    name, ns = service["name"], service["namespace"]
    observation = f"Service {name} health check failed: {health_info['response'][:200]}"
    action = f"kubectl rollout restart deployment {name} -n {ns}"

    if dry_run:
        result = "(dry run)"
        executed = False
    else:
        result = run_cmd(action, timeout=15)
        executed = True

    return make_decision("service_health", observation, action, "restart",
                         result[:500], "deterministic", executed, time.time() - start)


def watch_iteration(services: list[dict], latency_threshold: float,
                    max_replicas: int, dry_run: bool, log_file: str) -> list[dict]:
    decisions = []

    for svc in services:
        pod_issues = check_pod_health(svc["namespace"])
        for issue in pod_issues:
            if svc["name"] in issue["pod"]:
                d = handle_pod_issue(issue, dry_run)
                decisions.append(d)
                log_decision(d, log_file)
                print(f"  [{d['action_type']}] {d['observation']}")

        health = check_service_health(svc["health_url"])
        if not health["healthy"]:
            d = handle_service_down(svc, health, dry_run)
            decisions.append(d)
            log_decision(d, log_file)
            print(f"  [{d['action_type']}] {d['observation']}")

        latency = check_latency(svc["metrics_url"], latency_threshold)
        d = handle_high_latency(svc, latency, max_replicas, dry_run)
        if d:
            decisions.append(d)
            log_decision(d, log_file)
            print(f"  [{d['action_type']}] {d['observation']}")

    if not decisions:
        d = make_decision("routine", "All checks passed", "none", "none", "healthy", "deterministic", False, 0)
        log_decision(d, log_file)

    return decisions


def watch_loop(interval: int, latency_threshold: float, max_replicas: int,
               dry_run: bool, log_file: str) -> None:
    print(f"Watch mode started (interval={interval}s, threshold={latency_threshold}s, "
          f"max_replicas={max_replicas}, dry_run={dry_run})")
    print(f"Logging to: {log_file}\n")

    iteration = 0
    while True:
        iteration += 1
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] Iteration {iteration}")

        try:
            watch_iteration(SERVICES, latency_threshold, max_replicas, dry_run, log_file)
        except Exception as e:
            print(f"  Error in watch iteration: {e}")

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLOps Autonomous Watch Mode (Level 3)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Seconds between checks")
    parser.add_argument("--latency-threshold", type=float, default=DEFAULT_LATENCY_THRESHOLD, help="P95 latency threshold in seconds")
    parser.add_argument("--max-replicas", type=int, default=DEFAULT_MAX_REPLICAS, help="Max replicas safety cap")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions without executing")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Decision log file path")
    args = parser.parse_args()

    watch_loop(args.interval, args.latency_threshold, args.max_replicas, args.dry_run, args.log)


if __name__ == "__main__":
    main()
