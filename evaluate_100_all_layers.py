"""Avalia os novos 12 heads: saídas fixas, confiança e previsões consecutivas."""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from train_100_heads import LAYERS, make_heads


LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
FIELDS = ["task", "split", "strategy", "depth", "threshold", "patience", "tolerance",
          "macro_f1", "accuracy", "pearson", "mse", "latency_ms_per_pair", "speedup",
          "mean_layers", "exit_percent", "selected_on_validation"]


@dataclass
class Prediction:
    logits: torch.Tensor
    layers: torch.Tensor


class AllLayerExit(torch.nn.Module):
    def __init__(self, base_path, heads_path, task):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(base_path)
        self.heads = make_heads(self.base)
        payload = torch.load(heads_path, map_location="cpu", weights_only=True)
        if payload["task"] != task or tuple(payload["layers"]) != LAYERS:
            raise ValueError("Checkpoint de heads incompatível")
        if Path(payload["base_model"]).parts[-5:] != Path(base_path).parts[-5:]:
            raise ValueError("Os heads foram treinados com outro encoder")
        self.heads.load_state_dict(payload["state_dict"])
        self.task = task

    @torch.inference_mode()
    def predict(self, input_ids, attention_mask, token_type_ids=None, strategy="fixed",
                depth=12, threshold=0.3, patience=2, tolerance=0.2):
        if strategy == "base":
            result = self.base(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
            return Prediction(result.logits, torch.full((len(input_ids),), 12, device=input_ids.device))
        if strategy not in {"fixed", "confidence", "stability"}:
            raise ValueError(strategy)
        if strategy == "confidence" and self.task != "rte":
            raise ValueError("Confiança de classes só se aplica ao RTE")
        if depth not in LAYERS or not 0 <= threshold <= 1 or patience < 2 or tolerance < 0:
            raise ValueError("Parâmetros de saída inválidos")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        batch = len(input_ids)
        active = torch.arange(batch, device=input_ids.device)
        hidden = self.base.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        mask = self.base.bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        logits_out = torch.empty(batch, self.base.config.num_labels, device=input_ids.device)
        layer_out = torch.empty(batch, dtype=torch.int64, device=input_ids.device)
        previous = torch.zeros(batch, device=input_ids.device)
        streak = torch.ones(batch, dtype=torch.int64, device=input_ids.device)
        for layer, block in enumerate(self.base.bert.encoder.layer, start=1):
            hidden = block(hidden, attention_mask=mask)[0]
            if strategy == "fixed" and layer < depth:
                continue
            logits = self.heads[str(layer)](hidden[:, 0]) if layer < 12 or strategy == "fixed" else \
                self.base.classifier(self.base.bert.pooler(hidden))
            if strategy == "fixed" or layer == 12:
                should_exit = torch.ones(len(active), dtype=torch.bool, device=input_ids.device)
            elif strategy == "confidence":
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / np.log(2)
                should_exit = entropy <= threshold
            else:
                current = logits.argmax(-1).float() if self.task == "rte" else logits.squeeze(-1)
                stable = (current == previous[active]) if self.task == "rte" else \
                    (current - previous[active]).abs() <= tolerance
                if layer == 1:
                    stable = torch.zeros_like(stable)
                streak[active] = torch.where(stable, streak[active] + 1, 1)
                previous[active] = current
                should_exit = streak[active] >= patience
            leaving = active[should_exit]
            logits_out[leaving] = logits[should_exit]
            layer_out[leaving] = layer
            if strategy == "fixed" or layer == 12:
                break
            keep = ~should_exit
            if not bool(keep.any()):
                break
            active, hidden, mask = active[keep], hidden[keep], mask[keep]
        return Prediction(logits_out, layer_out)


def score(task, gold, logits):
    if task == "rte":
        predicted = logits.argmax(axis=-1)
        return {"macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(gold, predicted))}
    predicted = np.clip(logits.reshape(-1), 1, 5)
    pearson = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
    return {"pearson": pearson, "mse": float(mean_squared_error(gold, predicted))}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, prepared, task, spec, device, repeats):
    with torch.inference_mode():
        model.predict(**prepared[0][0], **spec)
        times = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            for inputs, _ in prepared:
                model.predict(**inputs, **spec)
            synchronize(device)
            times.append(time.perf_counter() - start)
        predictions, layers, gold = [], [], []
        for inputs, labels in prepared:
            result = model.predict(**inputs, **spec)
            predictions.append(result.logits.cpu().numpy())
            layers.append(result.layers.cpu().numpy())
            gold.append(labels)
    predictions = np.concatenate(predictions)
    layers = np.concatenate(layers)
    gold = np.concatenate(gold)
    return {**score(task, gold, predictions),
            "latency_ms_per_pair": float(1000 * np.median(times) / len(gold)),
            "mean_layers": float(layers.mean()),
            "exit_percent": {str(layer): float(100 * np.mean(layers == layer)) for layer in LAYERS}}


def specifications(task):
    fixed = [{"strategy": "fixed", "depth": layer} for layer in LAYERS]
    confidence = ([{"strategy": "confidence", "threshold": round(x / 10, 1)} for x in range(1, 10)]
                  if task == "rte" else [])
    stability = ([{"strategy": "stability", "patience": x} for x in (2, 3)] if task == "rte" else
                 [{"strategy": "stability", "patience": p, "tolerance": t}
                  for p in (2, 3) for t in (0.1, 0.2, 0.3, 0.5)])
    return [{"strategy": "base"}, *fixed, *confidence, *stability]


def select(rows, metric, baseline_quality, baseline_latency, max_drop):
    choices = {}
    for strategy in ("fixed", "confidence", "stability"):
        feasible = [row for row in rows if row["strategy"] == strategy and
                    row[metric] >= baseline_quality - max_drop and
                    row["latency_ms_per_pair"] <= baseline_latency / 1.05]
        choices[strategy] = min(feasible, key=lambda row: (row["latency_ms_per_pair"], -row[metric])) if feasible else None
    return choices


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "exit_percent": json.dumps(row["exit_percent"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeats < 1 or args.batch_size < 1:
        raise ValueError("repeats e batch-size devem ser positivos")
    task = args.task
    task_root = ROOT / "outputs" / "100epochs"
    base_path = task_root / "base" / task / "best_model"
    heads_path = task_root / "heads" / task / "best_heads.pt"
    destination = task_root / "evaluation" / task
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("%s | GPU=%s | repetições=%d", task, device, args.repeats)
    model = AllLayerExit(base_path, heads_path, task).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    data_dir = ROOT / "data" / "assin2"
    files = {split: str(data_dir / f"{split}.jsonl") for split in ("validation", "test")}
    raw = load_dataset("json", data_files=files, cache_dir=str(data_dir / ".cache"))
    label = "entailment_judgment" if task == "rte" else "relatedness_score"
    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)
        result["labels"] = batch[label]
        return result
    data = raw.map(tokenize, batched=True, remove_columns=raw["validation"].column_names)
    prepared = {}
    for split in ("validation", "test"):
        loader = DataLoader(data[split], batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
        prepared[split] = [({key: value.to(device) for key, value in batch.items() if key != "labels"},
                            batch["labels"].numpy()) for batch in loader]
    specs = specifications(task)
    metric = "macro_f1" if task == "rte" else "pearson"
    outputs = {}
    for split in ("validation", "test"):
        rows = []
        for spec in specs:
            result = evaluate(model, prepared[split], task, spec, device, args.repeats)
            rows.append({"task": task, "split": split, **spec, **result})
            LOG.info("%s %s %s=%.4f latência=%.3f ms/par", split, spec, metric,
                     result[metric], result["latency_ms_per_pair"])
        baseline = next(row for row in rows if row["strategy"] == "base")
        for row in rows:
            row["speedup"] = baseline["latency_ms_per_pair"] / row["latency_ms_per_pair"]
        outputs[split] = rows
    baseline = next(row for row in outputs["validation"] if row["strategy"] == "base")
    drop = 0.01 if task == "rte" else 0.02
    selected = select(outputs["validation"], metric, baseline[metric], baseline["latency_ms_per_pair"], drop)
    def key(row):
        return tuple(row.get(field) for field in ("strategy", "depth", "threshold", "patience", "tolerance"))
    selected_keys = {key(row) for row in selected.values() if row is not None}
    for split in outputs:
        for row in outputs[split]:
            row["selected_on_validation"] = key(row) in selected_keys
        write_rows(destination / f"{split}.csv", outputs[split])
    report = {"task": task, "base_checkpoint": str(base_path.resolve()),
              "heads_checkpoint": str(heads_path.resolve()), "head_layers": list(LAYERS),
              "metric": metric, "max_quality_drop": drop, "min_speedup": 1.05,
              "selected_on_validation": selected, "repeats": args.repeats,
              "batch_size": args.batch_size, "test_candidates_exploratory": True,
              "validation": outputs["validation"], "test": outputs["test"]}
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(task, "selecionados:", {k: (v and key(v)) for k, v in selected.items()})


if __name__ == "__main__":
    main()
