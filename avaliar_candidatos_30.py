"""Avalia no teste todos os candidatos da validação, sem mudar a seleção original."""

import argparse
import csv
import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from early_exit import EarlyExitBert
from evaluate_early_exit import evaluate


REPO = Path(__file__).resolve().parent
ROOT = REPO / "backup"
SPEC_FIELDS = ("strategy", "depth", "threshold", "patience", "tolerance")


def spec_key(row):
    return tuple(row.get(field) for field in SPEC_FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    args = parser.parse_args()
    task = args.task
    directory = ROOT / "outputs" / "long_overfit" / "selected_heads" / task
    report = json.loads((directory / "evaluation" / "report.json").read_text())
    config = json.loads((directory / "heads_config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Esta avaliação requer a GPU configurada")
    model = EarlyExitBert(ROOT / "outputs" / "epoch_sweep" / task / f"epoch-{config['base_epoch']}")
    model.load_heads(directory / "heads.pt")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "outputs" / "epoch_sweep" / task / "best_model")
    dataset = load_dataset("json", data_files={"test": str(REPO / "data" / "assin2" / "test.jsonl")},
                           cache_dir=str(REPO / "data" / "assin2" / ".cache"))["test"]
    label = "entailment_judgment" if task == "rte" else "relatedness_score"
    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)
        result["labels"] = batch[label]
        return result
    encoded = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    batches = list(DataLoader(encoded, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer)))
    existing = {spec_key(row): row for row in report["test"]}
    baseline_latency = next(row["latency_ms_per_pair"] for row in report["test"]
                            if row["strategy"] == "fixed" and row.get("depth") == 12)
    output = []
    for candidate in report["validation"]:
        spec = {key: candidate[key] for key in SPEC_FIELDS if key in candidate}
        key = spec_key(spec)
        if key in existing:
            row = dict(existing[key])
            origin = "teste_original"
        else:
            row = {**spec, **evaluate(model, batches, task, spec, device, repeats=1)}
            row["speedup"] = baseline_latency / row["latency_ms_per_pair"]
            origin = "teste_adicional"
        output.append({**row, "origem": origin})
        print(task, origin, spec, row.get("macro_f1", row.get("pearson")), flush=True)
    destination = directory / "evaluation" / "test_all_candidates.csv"
    columns = [*SPEC_FIELDS, "macro_f1" if task == "rte" else "pearson",
               "accuracy" if task == "rte" else "mse", "latency_ms_per_pair", "speedup", "mean_layers", "origem"]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    print(destination)


if __name__ == "__main__":
    main()
