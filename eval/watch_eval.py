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


def _wait_for_stable_cluster(service_name: str = "ml-service", timeout: int = 60) -> None:
    """Reset deployment to 1 replica"""
    run_cmd(f"kubectl scale deployment/{service_name} --replicas=1")
    run_cmd(f"kubectl rollout status deployment/{service_name} --timeout={timeout}s")
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = run_cmd(f"kubectl get pods -l app={service_name} --no-headers")
        unstable = any(
            parts[2] in ("Terminating", "ContainerCreating", "Pending", "Init", "Error")
            for line in out.splitlines() if len(parts := line.split()) >= 3
        )
        if not unstable:
            return
        time.sleep(2)


def run_scenario(name: str, inject_fn, action_types: list[str],
                 watch_interval: int = 5, latency_threshold: float = 1.0,
                 max_replicas: int = 5, timeout: int = 240) -> dict:
    log_file = tempfile.mktemp(suffix=".jsonl", prefix=f"watch_eval_{name}_")
    print(f"\nScenario: {name}")
    print(f"  Log: {log_file}")

    # Settle the cluster before starting the watch loop — prevents pre-existing
    # rollouts / Terminating pods from being flagged as "detections" before
    # inject_time, which would produce negative MTTD values.
    _wait_for_stable_cluster()

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


# Any non-trivial action counts as a valid detection: the system reacted.
_DETECTION_ACTIONS = ["restart", "diagnose", "scale_up"]


def run_pod_crash_scenario(watch_interval: int = 5) -> dict:
    return run_scenario(
        name="pod_crash",
        inject_fn=inject_pod_crash,
        action_types=_DETECTION_ACTIONS,
        watch_interval=watch_interval,
    )


def run_bad_image_scenario(watch_interval: int = 5) -> dict:
    result = run_scenario(
        name="bad_image",
        inject_fn=inject_bad_image,
        action_types=_DETECTION_ACTIONS,
        watch_interval=watch_interval,
    )
    restore_good_image()
    return result


def run_high_latency_scenario(concurrency: int = 20, watch_interval: int = 5) -> dict:
    result = run_scenario(
        name="high_latency",
        inject_fn=lambda: inject_high_latency(concurrency=concurrency, requests=300),
        action_types=_DETECTION_ACTIONS,
        watch_interval=watch_interval,
        latency_threshold=1.0,
    )
    # Reset replicas and wait for rollout so the next scenario starts clean.
    run_cmd("kubectl scale deployment ml-service --replicas=1")
    run_cmd("kubectl rollout status deployment/ml-service --timeout=60s")
    return result


def aggregate_runs(name: str, runs: list[dict]) -> dict:
    import statistics as _s
    detected = [r for r in runs if r["detected"]]
    mttds = [r["mttd_seconds"] for r in detected]
    return {
        "scenario": name,
        "runs": len(runs),
        "detected_count": len(detected),
        "detection_rate": round(len(detected) / len(runs), 3),
        "mttd_mean_s": round(_s.mean(mttds), 2) if mttds else None,
        "mttd_std_s": round(_s.stdev(mttds), 2) if len(mttds) > 1 else 0.0,
        "raw": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Watch Mode Fault Injection Evaluation")
    parser.add_argument("--scenario", choices=["pod_crash", "bad_image", "high_latency"],
                        help="Run a specific scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions per scenario (default: 1)")
    parser.add_argument("--concurrency", type=int, default=20, help="Load test concurrency for latency scenario")
    parser.add_argument("--interval", type=int, default=5, help="Watch check interval in seconds")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    scenarios = {
        "pod_crash": lambda: run_pod_crash_scenario(args.interval),
        "bad_image": lambda: run_bad_image_scenario(args.interval),
        "high_latency": lambda: run_high_latency_scenario(args.concurrency, args.interval),
    }

    to_run = list(scenarios.keys()) if args.all else ([args.scenario] if args.scenario else [])
    if not to_run:
        parser.print_help()
        return

    aggregated = []
    for name in to_run:
        print(f"\nRunning {args.runs} repetition(s) of '{name}'...")
        runs = [scenarios[name]() for _ in range(args.runs)]
        aggregated.append(aggregate_runs(name, runs))

    print(f"\n{'='*60}")
    print(f"WATCH EVALUATION RESULTS  ({args.runs} run(s) per scenario)")
    print(f"{'='*60}")
    for a in aggregated:
        rate = f"{a['detected_count']}/{a['runs']} detected"
        mttd = f"MTTD {a['mttd_mean_s']}±{a['mttd_std_s']}s" if a["mttd_mean_s"] else "NOT DETECTED"
        print(f"  {a['scenario']:<16}  {rate}  {mttd}")

    if args.output:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs_per_scenario": args.runs,
            "scenarios": aggregated,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
