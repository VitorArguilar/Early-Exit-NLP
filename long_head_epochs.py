"""Explora 30 épocas de treino dos heads, com registros a cada 2 épocas."""

import argparse
import csv
import json
import logging
import os
import random
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from early_exit import EXITS, EarlyExitBert


CHECKPOINTS = tuple(range(2, 31, 2))
LAYERS = EXITS[:-1]
LOG = logging.getLogger(__name__)


def evaluate_heads(model, loader, task, device):
    model.heads.eval()
    model.base.eval()
    collected = {layer: {"pred": [], "loss_sum": 0.0} for layer in LAYERS}
    gold = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model.forward_heads(**batch)
            gold.append(labels.cpu().numpy())
            for layer in LAYERS:
                output = logits[layer]
                if task == "rte":
                    loss = torch.nn.functional.cross_entropy(output, labels.long(), reduction="sum")
                else:
                    loss = torch.nn.functional.mse_loss(output.squeeze(-1), labels.float(), reduction="sum")
                collected[layer]["loss_sum"] += loss.item()
                collected[layer]["pred"].append(output.cpu().numpy())
    gold = np.concatenate(gold)
    result = {}
    for layer in LAYERS:
        predictions = np.concatenate(collected[layer]["pred"])
        row = {"loss": collected[layer]["loss_sum"] / len(gold)}
        if task == "rte":
            predicted = predictions.argmax(-1)
            row["macro_f1"] = float(f1_score(gold, predicted, average="macro", zero_division=0))
            row["accuracy"] = float(accuracy_score(gold, predicted))
        else:
            predicted = np.clip(predictions.reshape(-1), 1, 5)
            row["pearson"] = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
            row["mse"] = float(mean_squared_error(gold, predicted))
        result[str(layer)] = row
    return result


def cache_features(model, loader, device):
    """Guarda apenas os vetores CLS; o encoder está congelado e em modo eval."""
    all_features, all_labels = [], []
    model.base.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            hidden, mask = model._start(**batch)
            features = []
            for index, block in enumerate(model.base.bert.encoder.layer[:9], start=1):
                hidden = block(hidden, attention_mask=mask)[0]
                if index in LAYERS:
                    features.append(hidden[:, 0].detach().cpu())
            all_features.append(torch.stack(features, dim=1))
            all_labels.append(labels.cpu())
    return TensorDataset(torch.cat(all_features), torch.cat(all_labels))


def evaluate_cached(model, loader, task, device):
    model.heads.eval()
    collected = {layer: {"pred": [], "loss_sum": 0.0} for layer in LAYERS}
    gold = []
    with torch.inference_mode():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            gold.append(labels.cpu().numpy())
            for position, layer in enumerate(LAYERS):
                output = model.heads[str(layer)](features[:, position])
                if task == "rte":
                    loss = torch.nn.functional.cross_entropy(output, labels.long(), reduction="sum")
                else:
                    loss = torch.nn.functional.mse_loss(output.squeeze(-1), labels.float(), reduction="sum")
                collected[layer]["loss_sum"] += loss.item()
                collected[layer]["pred"].append(output.cpu().numpy())
    gold = np.concatenate(gold)
    result = {}
    for layer in LAYERS:
        predictions = np.concatenate(collected[layer]["pred"])
        row = {"loss": collected[layer]["loss_sum"] / len(gold)}
        if task == "rte":
            predicted = predictions.argmax(-1)
            row["macro_f1"] = float(f1_score(gold, predicted, average="macro", zero_division=0))
            row["accuracy"] = float(accuracy_score(gold, predicted))
        else:
            predicted = np.clip(predictions.reshape(-1), 1, 5)
            row["pearson"] = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
            row["mse"] = float(mean_squared_error(gold, predicted))
        result[str(layer)] = row
    return result


def plot(rows, task, destination):
    metric = "macro_f1" if task == "rte" else "pearson"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for layer in LAYERS:
        series = [row for row in rows if row["layer"] == layer]
        epochs = [row["epoch"] for row in series]
        line, = axes[0].plot(epochs, [row["validation_" + metric] for row in series], marker="o", label=f"Camada {layer}")
        axes[1].plot(epochs, [row["train_loss"] for row in series], marker="o", linestyle="--", color=line.get_color(), label=f"Treino {layer}")
        axes[1].plot(epochs, [row["validation_loss"] for row in series], marker="o", color=line.get_color(), label=f"Validação {layer}")
    axes[0].set(xlabel="Épocas dos heads", ylabel=metric, title="Qualidade na validação", xticks=CHECKPOINTS)
    axes[1].set(xlabel="Épocas dos heads", ylabel="Perda", title="Treino × validação", xticks=CHECKPOINTS)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/long_overfit/heads"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.learning_rate <= 0 or args.max_length < 1:
        raise ValueError("batch-size, learning-rate e max-length devem ser positivos")
    model_path = args.model or Path("outputs/epoch_sweep") / args.task / "epoch-7"
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(model_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Tarefa: %s | dispositivo: %s%s", args.task, device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    model = EarlyExitBert(model_path).to(device)
    if model.task != args.task:
        raise ValueError("Checkpoint incompatível com a tarefa")
    model.freeze_base()
    tokenizer = AutoTokenizer.from_pretrained(Path("outputs/epoch_sweep") / args.task / "best_model")
    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if any(not Path(path).is_file() for path in files.values()):
        raise FileNotFoundError("Execute prepare_data.py antes desta exploração")
    raw = load_dataset("json", data_files=files, cache_dir=str(args.data_dir / ".cache"))
    label = "entailment_judgment" if args.task == "rte" else "relatedness_score"

    def tokenize(batch):
        output = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=args.max_length)
        output["labels"] = batch[label]
        return output

    data = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer)
    train_raw_loader = DataLoader(data["train"], batch_size=args.batch_size, collate_fn=collator)
    val_raw_loader = DataLoader(data["validation"], batch_size=args.batch_size, collate_fn=collator)
    LOG.info("Extraindo vetores CLS do encoder congelado uma vez por split")
    train_features = cache_features(model, train_raw_loader, device)
    val_features = cache_features(model, val_raw_loader, device)
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    train_eval_loader = DataLoader(train_features, batch_size=args.batch_size)
    val_loader = DataLoader(val_features, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=args.learning_rate)
    destination = args.output_dir / args.task
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    online_losses = []
    for epoch in range(1, max(CHECKPOINTS) + 1):
        model.heads.train()
        model.base.eval()
        total = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = {layer: model.heads[str(layer)](features[:, position])
                      for position, layer in enumerate(LAYERS)}
            if args.task == "rte":
                loss = sum(torch.nn.functional.cross_entropy(logits[layer], labels.long()) for layer in LAYERS) / len(LAYERS)
            else:
                loss = sum(torch.nn.functional.mse_loss(logits[layer].squeeze(-1), labels.float()) for layer in LAYERS) / len(LAYERS)
            loss.backward()
            optimizer.step()
            total += loss.item()
        mean_loss = total / len(train_loader)
        online_losses.append(mean_loss)
        LOG.info("Época %d/%d | perda de treino online: %.5f", epoch, max(CHECKPOINTS), mean_loss)
        if epoch not in CHECKPOINTS:
            continue
        checkpoint = destination / f"epoch-{epoch}"
        model.save_heads(checkpoint / "heads.pt")
        train_metrics = evaluate_cached(model, train_eval_loader, args.task, device)
        val_metrics = evaluate_cached(model, val_loader, args.task, device)
        for layer in LAYERS:
            row = {"epoch": epoch, "layer": layer, "train_loss": train_metrics[str(layer)]["loss"], "validation_loss": val_metrics[str(layer)]["loss"]}
            for name in ("macro_f1", "accuracy") if args.task == "rte" else ("pearson", "mse"):
                row["train_" + name] = train_metrics[str(layer)][name]
                row["validation_" + name] = val_metrics[str(layer)][name]
            row["loss_gap"] = row["validation_loss"] - row["train_loss"]
            rows.append(row)
            with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            LOG.info("Camada %d, época %d | treino %s=%.4f | validação %s=%.4f | perdas %.4f/%.4f", layer, epoch, "F1" if args.task == "rte" else "Pearson", row["train_macro_f1" if args.task == "rte" else "train_pearson"], "F1" if args.task == "rte" else "Pearson", row["validation_macro_f1" if args.task == "rte" else "validation_pearson"], row["train_loss"], row["validation_loss"])

    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric = "validation_macro_f1" if args.task == "rte" else "validation_pearson"
    best = {str(layer): max((row for row in rows if row["layer"] == layer), key=lambda row: (row[metric], -row["epoch"]))["epoch"] for layer in LAYERS}
    result = {
        "task": args.task,
        "base_model": str(model_path.resolve()),
        "epochs_compared": list(CHECKPOINTS),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "online_train_loss": online_losses,
        "rows": rows,
        "best_validation_epoch_by_layer": best,
        "test_evaluated": False,
        "note": "Overfitting é sugerido quando a perda de validação aumenta enquanto a de treino diminui; diferenças pequenas devem ser interpretadas com cautela.",
    }
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    plot(rows, args.task, destination / "quality_and_loss.png")
    LOG.info("Exploração concluída em %s", destination)


if __name__ == "__main__":
    main()
