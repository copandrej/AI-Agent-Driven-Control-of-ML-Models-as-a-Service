"""
Aggregate multiple eval/runner.py result JSONs for the same (arch, model).
Groups by (architecture, llm_model) from metadata inside each file.

Usage:
    uv run python eval/aggregate.py eval/results/
    uv run python eval/aggregate.py eval/results/microk8s/
    uv run python eval/aggregate.py eval/results/ --compare eval/results/microk8s/
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ARCH_LABELS = {
    "flow": "Multi-Agent",
    "simple": "Single-Agent",
    "dual": "Dual-Agent",
    "minimal": "Minimal",
    "bare": "Bare",
}
MODEL_LABELS = {
    "Qwen-Qwen3.5-122B-A10B": "Qwen3.5-122B",
    "MiniMaxAI-MiniMax-M2.5": "MiniMax-M2.5",
    "qwen3.5:9b-64k": "Qwen3.5-9B",
    "qwen3.5:9b": "Qwen3.5-9B",
}
ARCH_ORDER = ["bare", "minimal", "simple", "dual", "flow"]
MODEL_ORDER = ["qwen3.5:9b-64k", "qwen3.5:9b", "Qwen-Qwen3.5-122B-A10B", "MiniMaxAI-MiniMax-M2.5"]


def model_label(raw: str) -> str:
    name = raw.split("/")[-1]
    return MODEL_LABELS.get(name, name)


def arch_label(raw: str) -> str:
    return ARCH_LABELS.get(raw, raw)


def load_and_group(results_dir: Path) -> dict:
    groups = defaultdict(list)
    for f in sorted(results_dir.glob("results_*.json")):
        if "aggregate" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if data.get("metadata", {}).get("status") != "complete":
                continue
            meta = data["metadata"]
            key = (meta["architecture"], meta["llm_model"].split("/")[-1])
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
            }

    meta = runs[-1]["metadata"]
    return {
        "metadata": {
            "architecture": meta["architecture"],
            "llm_model": meta["llm_model"].split("/")[-1],
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


def fmt(mean, std) -> str:
    return f"{mean:.1f}±{std:.1f}"


def print_table(groups: dict, label: str = "") -> dict:
    results = {}
    for (arch, model), runs in groups.items():
        results[(arch, model)] = aggregate(runs)

    # collect all models present, in preferred order
    models_present = []
    for m in MODEL_ORDER:
        if any(model == m for (_, model) in results):
            models_present.append(m)
    for (_, model) in results:
        if model not in models_present:
            models_present.append(model)

    header = f"  {'Architecture':<16}"
    for m in models_present:
        ml = model_label(m)
        header += f"  {ml:^22}"

    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
    print('='*70)
    print(f"  {'Architecture':<16}" + "".join(f"  {'Overall / L1 / L2':^22}" for _ in models_present))
    print(f"  {'':<16}" + "".join(f"  {model_label(m):^22}" for m in models_present))
    print('-'*70)

    for arch in ARCH_ORDER:
        row = f"  {arch_label(arch):<16}"
        for m in models_present:
            r = results.get((arch, m))
            if r:
                s = r["summary"]
                l1 = s["by_level"].get(1, {})
                l2 = s["by_level"].get(2, {})
                overall = fmt(s["success_rate_mean"], s["success_rate_std"])
                l1s = fmt(l1["mean"], l1["std"]) if l1 else "---"
                l2s = fmt(l2["mean"], l2["std"]) if l2 else "---"
                cell = f"{overall} / {l1s} / {l2s}"
                row += f"  {cell:<22}"
            else:
                row += f"  {'---':^22}"
        print(row)

    print('='*70)
    print(f"  Format: Overall±std / L1±std / L2±std  ({sum(len(v) for v in groups.values())} total runs)")
    return results


def write_aggregates(groups: dict, out_dir: Path) -> None:
    for (arch, model), runs in groups.items():
        result = aggregate(runs)
        model_tag = model.replace(":", "-").replace(".", "-").replace("/", "-")
        for old in out_dir.glob(f"aggregate_{arch}_{model_tag}_*runs.json"):
            old.unlink()
        out_path = out_dir / f"aggregate_{arch}_{model_tag}_{len(runs)}runs.json"
        out_path.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with results_*.json files")
    parser.add_argument("--compare", default=None, help="Second directory to compare (e.g. microk8s)")
    parser.add_argument("--out", default=None, help="Output directory for aggregate JSONs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir

    groups = load_and_group(results_dir)
    if not groups:
        print("No complete result files found.")
        return

    label = results_dir.name.upper() if results_dir.name != "results" else "KIND CLUSTER"
    print_table(groups, label)
    write_aggregates(groups, out_dir)
    print(f"\n  Wrote {len(groups)} aggregate JSON files → {out_dir}/\n")

    if args.compare:
        cmp_dir = Path(args.compare)
        cmp_groups = load_and_group(cmp_dir)
        if cmp_groups:
            cmp_label = cmp_dir.name.upper()
            print_table(cmp_groups, cmp_label)
            write_aggregates(cmp_groups, cmp_dir)
            print(f"\n  Wrote {len(cmp_groups)} aggregate JSON files → {cmp_dir}/\n")


if __name__ == "__main__":
    main()
