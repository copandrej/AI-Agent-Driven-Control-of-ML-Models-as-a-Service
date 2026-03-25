"""
Watch Mode Evaluation (Level 3 Fault Injection)
=================================================
Injects faults into the cluster and measures how the watch mode
detects and remediates them. Produces MTTD/MTTR metrics.

Usage:
    uv run python eval/watch_eval.py --scenario pod_crash
    uv run python eval/watch_eval.py --scenario high_latency --concurrency 20
    uv run python eval/watch_eval.py --all --output eval_watch_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from src.agent import run_cmd
from src.watch import SERVICES, watch_iteration


def inject_pod_crash(service_name: str = "ml-service", namespace: str = "default") -> float:
    """Kill the first pod of a deployment. Returns injection timestamp."""
    pods = run_cmd(f"kubectl get pods -n {namespace} -l app={service_name} -o name")
    if not pods or "No resources" in pods:
        print(f"  No pods found for {service_name}, deploying first...")
        run_cmd("kubectl apply -f k8s/ml_service/deployment.yaml")
        time.sleep(8)
        pods = run_cmd(f"kubectl get pods -n {namespace} -l app={service_name} -o name")

    pod = pods.strip().splitlines()[0]
    inject_time = time.time()
    run_cmd(f"kubectl delete {pod} -n {namespace} --grace-period=0 --force")
    print(f"  Injected: killed {pod}")
    return inject_time


def inject_bad_image(service_name: str = "ml-service", namespace: str = "default") -> float:
    """Set a nonexistent image tag to trigger ImagePullBackOff."""
    inject_time = time.time()
    run_cmd(f"kubectl set image deployment/{service_name} {service_name}=nonexistent:broken -n {namespace}")
    print(f"  Injected: set bad image on {service_name}")
    return inject_time


def restore_good_image(service_name: str = "ml-service", namespace: str = "default") -> None:
    run_cmd(f"kubectl set image deployment/{service_name} {service_name}={service_name}:latest -n {namespace}")
    run_cmd(f"kubectl rollout status deployment/{service_name} -n {namespace} --timeout=60s")


def inject_high_latency(concurrency: int = 20, requests: int = 100) -> float:
    """Generate load to spike latency. Returns injection timestamp."""
    load_test = os.path.join(os.path.dirname(__file__), "load_test.py")
    inject_time = time.time()
    env = {**os.environ, "LOAD_TEST_URL": "http://localhost:30081/predict"}
    proc = subprocess.Popen(
        [sys.executable, load_test, "--concurrency", str(concurrency),
         "--requests", str(requests), "--output", "/dev/null"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    print(f"  Injected: load test (concurrency={concurrency}, requests={requests}, pid={proc.pid})")
    return inject_time


def wait_for_detection(log_file: str, inject_time: float, action_types: list[str],
                       timeout: int = 120) -> dict | None:
    """Wait until the watch log contains a relevant action after inject_time."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if entry.get("action_type") in action_types:
                        entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        if entry_time >= inject_time - 1:
                            return entry
                    if entry.get("action_type") == "diagnose" and any(at in str(entry) for at in action_types):
                        entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        if entry_time >= inject_time - 1:
                            return entry
        time.sleep(2)
    return None


def run_scenario(name: str, inject_fn, action_types: list[str],
                 watch_interval: int = 5, latency_threshold: float = 0.05,
                 max_replicas: int = 5, timeout: int = 120) -> dict:
    log_file = tempfile.mktemp(suffix=".jsonl", prefix=f"watch_eval_{name}_")
    print(f"\nScenario: {name}")
    print(f"  Log: {log_file}")

    stop_event = threading.Event()

    def watch_thread():
        while not stop_event.is_set():
            try:
                watch_iteration(SERVICES, latency_threshold, max_replicas, False, log_file)
            except Exception as e:
                print(f"  Watch error: {e}")
            for _ in range(watch_interval * 10):
                if stop_event.is_set():
                    return
                time.sleep(0.1)

    t = threading.Thread(target=watch_thread, daemon=True)
    t.start()

    time.sleep(watch_interval + 1)

    inject_time = inject_fn()

    detection = wait_for_detection(log_file, inject_time, action_types, timeout)

    stop_event.set()
    t.join(timeout=5)

    if detection:
        detect_time = datetime.fromisoformat(detection["timestamp"]).timestamp()
        mttd = round(detect_time - inject_time, 2)
        return {
            "scenario": name,
            "detected": True,
            "mttd_seconds": mttd,
            "action_type": detection["action_type"],
            "action": detection["action"],
            "observation": detection["observation"],
            "log_file": log_file,
        }
    else:
        return {
            "scenario": name,
            "detected": False,
            "mttd_seconds": None,
            "action_type": None,
            "action": None,
            "observation": "Not detected within timeout",
            "log_file": log_file,
        }


def run_pod_crash_scenario(watch_interval: int = 5) -> dict:
    return run_scenario(
        name="pod_crash",
        inject_fn=inject_pod_crash,
        action_types=["restart", "diagnose"],
        watch_interval=watch_interval,
    )


def run_bad_image_scenario(watch_interval: int = 5) -> dict:
    result = run_scenario(
        name="bad_image",
        inject_fn=inject_bad_image,
        action_types=["diagnose", "restart"],
        watch_interval=watch_interval,
    )
    restore_good_image()
    return result


def run_high_latency_scenario(concurrency: int = 20, watch_interval: int = 5) -> dict:
    result = run_scenario(
        name="high_latency",
        inject_fn=lambda: inject_high_latency(concurrency=concurrency),
        action_types=["scale_up"],
        watch_interval=watch_interval,
        latency_threshold=0.05,
    )
    run_cmd("kubectl scale deployment ml-service --replicas=1")

    return result


def main():
    parser = argparse.ArgumentParser(description="Watch Mode Fault Injection Evaluation")
    parser.add_argument("--scenario", choices=["pod_crash", "bad_image", "high_latency"],
                        help="Run a specific scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--concurrency", type=int, default=20, help="Load test concurrency for latency scenario")
    parser.add_argument("--interval", type=int, default=5, help="Watch check interval in seconds")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    scenarios = {
        "pod_crash": lambda: run_pod_crash_scenario(args.interval),
        "bad_image": lambda: run_bad_image_scenario(args.interval),
        "high_latency": lambda: run_high_latency_scenario(args.concurrency, args.interval),
    }

    results = []
    if args.all:
        for name, fn in scenarios.items():
            results.append(fn())
    elif args.scenario:
        results.append(scenarios[args.scenario]())
    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print("WATCH EVALUATION RESULTS")
    print(f"{'='*60}")
    for r in results:
        status = f"DETECTED in {r['mttd_seconds']}s" if r["detected"] else "NOT DETECTED"
        print(f"  {r['scenario']}: {status}")
        if r["detected"]:
            print(f"    Action: {r['action_type']} — {r['action']}")

    if args.output:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
