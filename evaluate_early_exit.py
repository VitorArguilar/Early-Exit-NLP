"""Seleciona early exit na validação e compara as estratégias no teste."""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data") / "matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from early_exit import EXITS, EarlyExitBert


LOGGER = logging.getLogger(__name__)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def score(task, gold, predictions):
    if task == "rte":
        predicted = predictions.argmax(axis=1)
        return {
            "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(gold, predicted)),
        }
    predicted = np.clip(predictions.reshape(-1), 1, 5)
    pearson = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
    return {"pearson": pearson, "mse": float(mean_squared_error(gold, predicted))}


def evaluate(model, batches, task, spec, device, repeats=3, collect=False):
    # Tokenização e transferência para GPU ficam fora da medição para todas as estratégias.
    prepared = []
    for batch in batches:
        prepared.append(({key: value.to(device) for key, value in batch.items() if key != "labels"}, batch["labels"].numpy()))
    with torch.inference_mode():
        model.predict(**prepared[0][0], **spec)
    synchronize(device)
    elapsed = []
    with torch.inference_mode():
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            for inputs, _ in prepared:
                model.predict(**inputs, **spec)
            synchronize(device)
            elapsed.append(time.perf_counter() - started)
    predictions = []
    exit_layers = []
    gold = []
    with torch.inference_mode():
        for inputs, labels in prepared:
            result = model.predict(**inputs, **spec)
            predictions.append(result.predictions.cpu().numpy())
            exit_layers.append(result.layers.cpu().numpy())
            gold.append(labels)
    predictions = np.concatenate(predictions)
    exit_layers = np.concatenate(exit_layers)
    gold = np.concatenate(gold)
    quality = score(task, gold, predictions)
    record = {
        **quality,
        "latency_ms_per_pair": float(1000 * np.median(elapsed) / len(gold)),
        "mean_layers": float(exit_layers.mean()),
        "exit_percent": {str(layer): float(100 * np.mean(exit_layers == layer)) for layer in EXITS},
    }
    if collect:
        return record, predictions, exit_layers
    return record


def specifications(task):
    fixed = [{"strategy": "fixed", "depth": layer} for layer in EXITS]
    if task == "rte":
        confidence = [{"strategy": "confidence", "threshold": float(round(value, 2))} for value in np.arange(0.1, 1.0, 0.1)]
        stability = [{"strategy": "stability", "patience": value} for value in (2, 3)]
    else:
        confidence = []
        stability = [{"strategy": "stability", "patience": patience, "tolerance": tolerance} for patience in (2, 3) for tolerance in (0.1, 0.2, 0.3, 0.5)]
    return fixed + confidence + stability


def select(candidates, baseline_quality, baseline_latency, metric, quality_drop):
    feasible = [row for row in candidates if row[metric] >= baseline_quality - quality_drop and row["latency_ms_per_pair"] <= baseline_latency / 1.05]
    if not feasible:
        return None
    return min(feasible, key=lambda row: (row["latency_ms_per_pair"], -row[metric]))


def chart(rows, metric, destination, title, x_metric="mean_layers"):
    colors = {"fixed": "#4063a2", "confidence": "#d07b2a", "stability": "#37936d"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in colors:
        subset = [row for row in rows if row["strategy"] == strategy]
        if subset:
            ax.scatter([row[x_metric] for row in subset], [row[metric] for row in subset], label=strategy, color=colors[strategy], s=50)
    ax.set(xlabel="Camadas médias executadas" if x_metric == "mean_layers" else "Latência (ms/par)", ylabel=metric, title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--heads-dir", type=Path, default=Path("outputs/early_exit"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--max-quality-drop", type=float, default=None, help="queda absoluta permitida na métrica principal")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.repeats < 1:
        raise ValueError("batch-size e repeats devem ser positivos")
    model_path = args.model or Path("outputs/epoch_sweep") / args.task / "epoch-7"
    head_dir = args.heads_dir / args.task
    config = json.loads((head_dir / "heads_config.json").read_text(encoding="utf-8"))
    if Path(config["base_model"]).resolve() != model_path.resolve():
        raise ValueError("Os heads foram treinados com outro checkpoint base")
    if config["max_length"] != 128:
        LOGGER.info("Usando max_length=%d salvo no treinamento dos heads", config["max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Hardware: %s%s | batch size: %d", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "", args.batch_size)
    model = EarlyExitBert(model_path)
    if model.task != args.task:
        raise ValueError("Checkpoint de outra tarefa")
    model.load_heads(head_dir / "heads.pt")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(Path("outputs/epoch_sweep") / args.task / "best_model")
    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("validation", "test")}
    raw = load_dataset("json", data_files=files, cache_dir=str(args.data_dir / ".cache"))
    label = "entailment_judgment" if args.task == "rte" else "relatedness_score"

    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=config["max_length"])
        result["labels"] = batch[label]
        return result

    encoded = raw.map(tokenize, batched=True, remove_columns=raw["validation"].column_names)
    collator = DataCollatorWithPadding(tokenizer)
    validation = list(DataLoader(encoded["validation"], batch_size=args.batch_size, collate_fn=collator))
    test = list(DataLoader(encoded["test"], batch_size=args.batch_size, collate_fn=collator))
    metric = "macro_f1" if args.task == "rte" else "pearson"
    quality_drop = args.max_quality_drop if args.max_quality_drop is not None else (0.01 if args.task == "rte" else 0.02)
    if quality_drop < 0:
        raise ValueError("max-quality-drop deve ser não negativo")
    specs = specifications(args.task)
    val_rows = []
    for spec in specs:
        metrics = evaluate(model, validation, args.task, spec, device, args.repeats)
        val_rows.append({**spec, **metrics})
        LOGGER.info("Validação %s: %s=%.4f, latência=%.3f ms/par, camadas=%.2f", spec, metric, metrics[metric], metrics["latency_ms_per_pair"], metrics["mean_layers"])
    baseline = next(row for row in val_rows if row["strategy"] == "fixed" and row["depth"] == 12)
    for row in val_rows:
        row["speedup"] = baseline["latency_ms_per_pair"] / row["latency_ms_per_pair"]
    selected = {strategy: select([row for row in val_rows if row["strategy"] == strategy], baseline[metric], baseline["latency_ms_per_pair"], metric, quality_drop) for strategy in ("confidence", "stability")}
    selected_specs = [spec for spec in specs if spec["strategy"] == "fixed"]
    selected_specs += [{key: value for key, value in row.items() if key in {"strategy", "threshold", "patience", "tolerance"}} for row in selected.values() if row is not None]

    test_rows = []
    detailed = {}
    for spec in selected_specs:
        metrics, predictions, exit_layers = evaluate(model, test, args.task, spec, device, args.repeats, collect=True)
        row = {**spec, **metrics}
        test_rows.append(row)
        detailed[json.dumps(spec, sort_keys=True)] = (predictions, exit_layers)
        LOGGER.info("Teste %s: %s=%.4f, latência=%.3f ms/par, camadas=%.2f", spec, metric, metrics[metric], metrics["latency_ms_per_pair"], metrics["mean_layers"])
    test_baseline = next(row for row in test_rows if row["strategy"] == "fixed" and row["depth"] == 12)
    for row in test_rows:
        row["speedup"] = test_baseline["latency_ms_per_pair"] / row["latency_ms_per_pair"]

    destination = head_dir / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    for strategy in ("confidence", "stability"):
        if selected[strategy] is None:
            (destination / f"{strategy}_test_examples.csv").unlink(missing_ok=True)
    for name, rows in (("validation", val_rows), ("test", test_rows)):
        with (destination / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            columns = ["strategy", "depth", "threshold", "patience", "tolerance", metric, "accuracy" if args.task == "rte" else "mse", "latency_ms_per_pair", "speedup", "mean_layers", "exit_percent"]
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "exit_percent": json.dumps(row["exit_percent"])})
        chart(rows, metric, destination / f"{name}_quality_vs_layers.png", f"{args.task.upper()} - {name}")
        chart(rows, metric, destination / f"{name}_quality_vs_latency.png", f"{args.task.upper()} - {name}", x_metric="latency_ms_per_pair")

    example_analysis = {}
    for strategy in ("confidence", "stability"):
        chosen = selected[strategy]
        if chosen is None:
            continue
        spec = {key: value for key, value in chosen.items() if key in {"strategy", "threshold", "patience", "tolerance"}}
        predictions, layers = detailed[json.dumps(spec, sort_keys=True)]
        grouped = {layer: {"lengths": [], "gold": [], "predictions": []} for layer in EXITS}
        with (destination / f"{strategy}_test_examples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sentence_pair_id", "token_length", "gold", "prediction", "exit_layer"])
            for index, example in enumerate(raw["test"]):
                prediction = int(predictions[index].argmax()) if args.task == "rte" else float(np.clip(predictions[index, 0], 1, 5))
                token_length = len(encoded["test"][index]["input_ids"])
                exit_layer = int(layers[index])
                writer.writerow([example["sentence_pair_id"], token_length, example[label], prediction, exit_layer])
                group = grouped[exit_layer]
                group["lengths"].append(token_length)
                group["gold"].append(example[label])
                group["predictions"].append(prediction)
        example_analysis[strategy] = {
            str(layer): {
                "count": len(grouped[layer]["lengths"]),
                "mean_token_length": float(np.mean(grouped[layer]["lengths"])) if grouped[layer]["lengths"] else None,
                "accuracy": float(accuracy_score(grouped[layer]["gold"], grouped[layer]["predictions"])) if args.task == "rte" and grouped[layer]["gold"] else None,
                "mse": float(mean_squared_error(grouped[layer]["gold"], grouped[layer]["predictions"])) if args.task == "sts" and grouped[layer]["gold"] else None,
            } for layer in EXITS
        }

    if args.export_predictions:
        all_examples = []
        for spec in selected_specs:
            predictions, layers = detailed[json.dumps(spec, sort_keys=True)]
            spec_name = spec["strategy"] + (f"_{spec['depth']}" if spec["strategy"] == "fixed" else "_selected")
            for index, example in enumerate(raw["test"]):
                gold_value = example[label]
                prediction = int(predictions[index].argmax()) if args.task == "rte" else float(np.clip(predictions[index, 0], 1, 5))
                error = int(prediction != gold_value) if args.task == "rte" else abs(prediction - gold_value)
                all_examples.append({
                    "task": args.task,
                    "seed": config["seed"],
                    "strategy": spec_name,
                    "parameters": json.dumps(spec, sort_keys=True),
                    "sentence_pair_id": example["sentence_pair_id"],
                    "premise": example["premise"],
                    "hypothesis": example["hypothesis"],
                    "token_length": len(encoded["test"][index]["input_ids"]),
                    "gold": gold_value,
                    "prediction": prediction,
                    "error": error,
                    "exit_layer": int(layers[index]),
                })
        columns = list(all_examples[0])
        with (destination / "all_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_examples)
        sampled = []
        for spec_name in sorted({row["strategy"] for row in all_examples}):
            for exit_layer in EXITS:
                group = [row for row in all_examples if row["strategy"] == spec_name and row["exit_layer"] == exit_layer]
                if not group:
                    continue
                ordered = sorted(group, key=lambda row: (row["error"], row["sentence_pair_id"]))
                sampled.extend(ordered[:2])
                sampled.extend(ordered[-2:])
        sampled = list({(row["strategy"], row["sentence_pair_id"]): row for row in sampled}.values())
        with (destination / "qualitative_examples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sampled)

    candidates = [row for row in val_rows if row["strategy"] != "fixed" or row["depth"] != 12]
    recommendation = select(candidates, baseline[metric], baseline["latency_ms_per_pair"], metric, quality_drop) or baseline
    report = {
        "task": args.task,
        "base_model": str(model_path.resolve()),
        "head_seed": config["seed"],
        "head_epochs": config["epochs"],
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "selection_metric": metric,
        "max_quality_drop": quality_drop,
        "min_validation_speedup": 1.05,
        "selected_on_validation": selected,
        "validation": val_rows,
        "test": test_rows,
        "recommended_on_validation": recommendation,
        "exit_group_analysis": example_analysis,
        "predictions_exported": args.export_predictions,
        "note": "A recomendação foi escolhida somente pela validação. O custo mede apenas o forward dos lotes já tokenizados e transferidos ao dispositivo. Diferenças pequenas de latência podem refletir ruído de medição.",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info("Relatório salvo em %s", destination)


if __name__ == "__main__":
    main()
