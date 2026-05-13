"""
Aggregate multiple eval/runner.py result JSONs for the same (arch, model).
Groups by (architecture, llm_model) from metadata inside each file.

Usage:
    uv run python eval/aggregate.py eval/results/
    uv run python eval/aggregate.py eval/results/ --out eval/results/aggregated/
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_and_group(results_dir: Path) -> dict:
    groups = defaultdict(list)
    for f in sorted(results_dir.glob("results_*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get("metadata", {}).get("status") != "complete":
                continue
            meta = data["metadata"]
            key = (meta["architecture"], meta["llm_model"])
            groups[key].append(data)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return groups


def aggregate(runs: list[dict]) -> dict:
    rates = [r["summary"]["success_rate"] for r in runs]
    task_ids = [t["task_id"] for t in runs[0]["tasks"]]

    per_task = {}
    for tid in task_ids:
        passes = sum(
            1 for r in runs for t in r["tasks"] if t["task_id"] == tid and t["success"]
        )
        per_task[tid] = {"pass_count": passes, "pass_rate": round(passes / len(runs), 3)}

    # per-level stats across runs
    by_level = {}
    for lvl in (1, 2, 3):
        lvl_rates = [
            r["summary"]["by_level"][str(lvl)]["success_rate"]
            for r in runs
            if str(lvl) in r["summary"].get("by_level", {})
        ]
        if lvl_rates:
            by_level[lvl] = {
                "mean": round(statistics.mean(lvl_rates), 1),
                "std": round(statistics.stdev(lvl_rates) if len(lvl_rates) > 1 else 0.0, 1),
                "runs": len(lvl_rates),
            }

    meta = runs[-1]["metadata"]
    return {
        "metadata": {
            "architecture": meta["architecture"],
            "llm_model": meta["llm_model"],
            "num_runs": len(runs),
            "levels": meta["levels"],
        },
        "summary": {
            "success_rate_mean": round(statistics.mean(rates), 1),
            "success_rate_std": round(statistics.stdev(rates) if len(rates) > 1 else 0.0, 1),
            "by_level": by_level,
        },
        "per_task": per_task,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing results_*.json files")
    parser.add_argument("--out", default=None, help="Output directory (default: same as results_dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir
    out_dir.mkdir(exist_ok=True)

    groups = load_and_group(results_dir)
    if not groups:
        print("No complete result files found.")
        return

    print(f"{'arch':<10} {'model':<45} {'runs':>4}  {'mean%':>6}  {'std':>5}")
    print("-" * 75)
    for (arch, model), runs in sorted(groups.items()):
        result = aggregate(runs)
        model_tag = model.split("/")[-1].replace(":", "-").replace(".", "-")
        out_path = out_dir / f"aggregate_{arch}_{model_tag}_{len(runs)}runs.json"
        out_path.write_text(json.dumps(result, indent=2))
        s = result["summary"]
        print(f"{arch:<10} {model:<45} {len(runs):>4}  {s['success_rate_mean']:>5.1f}%  ±{s['success_rate_std']:>4.1f}")
    print(f"\nWrote {len(groups)} aggregate files to {out_dir}/")


if __name__ == "__main__":
    main()
