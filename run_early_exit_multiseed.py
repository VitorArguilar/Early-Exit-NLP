"""Treina heads por 10 épocas em três seeds e agrega os experimentos Early Exit."""

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data") / "matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG = logging.getLogger(__name__)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def label_for(row):
    if row["strategy"] == "fixed":
        return f"fixed_{row['depth']}"
    return row["strategy"] + "_selected"


def flatten(task, seed, row):
    return {
        "task": task,
        "seed": seed,
        "strategy_label": label_for(row),
        "strategy": row["strategy"],
        "depth": row.get("depth"),
        "threshold": row.get("threshold"),
        "patience": row.get("patience"),
        "tolerance": row.get("tolerance"),
        "macro_f1": row.get("macro_f1"),
        "accuracy": row.get("accuracy"),
        "pearson": row.get("pearson"),
        "mse": row.get("mse"),
        "latency_ms_per_pair": row["latency_ms_per_pair"],
        "speedup": row["speedup"],
        "mean_layers": row["mean_layers"],
        **{f"exit_percent_{layer}": row["exit_percent"][str(layer)] for layer in (3, 6, 9, 12)},
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["strategy_label"])].append(row)
    summary = []
    for (task, label), entries in sorted(grouped.items()):
        result = {"task": task, "strategy_label": label, "runs": len(entries)}
        for key in ("macro_f1", "accuracy", "pearson", "mse", "latency_ms_per_pair", "speedup", "mean_layers", "exit_percent_3", "exit_percent_6", "exit_percent_9", "exit_percent_12"):
            values = [float(row[key]) for row in entries if row.get(key) is not None]
            if values:
                result[key + "_mean"] = statistics.mean(values)
                result[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def chart(summary, task, destination):
    metric = "macro_f1" if task == "rte" else "pearson"
    rows = [row for row in summary if row["task"] == task]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.errorbar(
            row["latency_ms_per_pair_mean"],
            row[metric + "_mean"],
            xerr=row["latency_ms_per_pair_std"],
            yerr=row[metric + "_std"],
            fmt="o",
            capsize=3,
            label=f"{row['strategy_label']} (n={row['runs']})",
        )
    ax.set(xlabel="Latência média (ms/par)", ylabel=metric, title=f"{task.upper()}: 10 épocas dos heads, 3 seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def aggregate(root, tasks, seeds):
    validation_rows = []
    test_rows = []
    selections = []
    qualitative = []
    for task in tasks:
        for seed in seeds:
            run_root = root / f"seed-{seed}" / task
            report = json.loads((run_root / "evaluation" / "report.json").read_text(encoding="utf-8"))
            if report.get("head_seed") != seed or report.get("head_epochs") != 10 or not report.get("predictions_exported"):
                raise ValueError(f"Metadados incompatíveis em {run_root}")
            validation_rows.extend(flatten(task, seed, row) for row in report["validation"])
            test_rows.extend(flatten(task, seed, row) for row in report["test"])
            for strategy, chosen in report["selected_on_validation"].items():
                selections.append({"task": task, "seed": seed, "strategy": strategy, "selected": chosen is not None, "parameters": json.dumps({key: chosen[key] for key in ("threshold", "patience", "tolerance") if key in chosen}, sort_keys=True) if chosen else ""})
            sample_path = run_root / "evaluation" / "qualitative_examples.csv"
            with sample_path.open(newline="", encoding="utf-8") as handle:
                qualitative.extend(csv.DictReader(handle))
    summary = summarize(test_rows)
    write_csv(root / "all_validation.csv", validation_rows)
    write_csv(root / "all_test.csv", test_rows)
    write_csv(root / "selection_by_seed.csv", selections)
    write_csv(root / "qualitative_examples_combined.csv", qualitative)
    write_csv(root / "summary.csv", summary)
    (root / "summary.json").write_text(json.dumps({"seeds": seeds, "tasks": tasks, "head_epochs": 10, "test_summary": summary, "note": "Médias e desvios amostrais são descritivos. Cada seed altera apenas o treino dos heads; o checkpoint BERTimbau Base é o mesmo. Hiperparâmetros de saída são escolhidos na validação de cada run."}, indent=2, allow_nan=False), encoding="utf-8")
    for task in tasks:
        chart(summary, task, root / f"{task}_quality_vs_latency.png")


def run_command(command):
    LOG.info("Executando: %s", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("rte", "sts"), default=["rte", "sts"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/early_exit_10epochs_3seeds"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(set(args.seeds)) != len(args.seeds) or args.repeats < 1:
        raise ValueError("Seeds devem ser distintas e repeats deve ser positivo")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for seed in args.seeds:
            run_root = args.output_dir / f"seed-{seed}"
            head_file = run_root / task / "heads.pt"
            config_file = run_root / task / "heads_config.json"
            report_file = run_root / task / "evaluation" / "report.json"
            sample_file = run_root / task / "evaluation" / "qualitative_examples.csv"
            head_ready = head_file.is_file() and config_file.is_file()
            if head_ready:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                head_ready = config.get("seed") == seed and config.get("epochs") == 10
            if not head_ready:
                run_command([sys.executable, "train_early_exit_heads.py", "--task", task, "--epochs", "10", "--seed", str(seed), "--output-dir", str(run_root)])
            report_ready = report_file.is_file() and sample_file.is_file()
            if report_ready:
                report = json.loads(report_file.read_text(encoding="utf-8"))
                report_ready = report.get("head_seed") == seed and report.get("head_epochs") == 10 and report.get("predictions_exported")
            if not report_ready:
                run_command([sys.executable, "evaluate_early_exit.py", "--task", task, "--heads-dir", str(run_root), "--repeats", str(args.repeats), "--export-predictions"])
    aggregate(args.output_dir, args.tasks, args.seeds)
    LOG.info("Agregados salvos em %s", args.output_dir)


if __name__ == "__main__":
    main()
