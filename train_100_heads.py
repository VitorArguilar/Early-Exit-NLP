"""Treina heads nas 12 camadas sobre o melhor encoder congelado, até 100 épocas."""

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


LOG = logging.getLogger(__name__)
LAYERS = tuple(range(1, 13))
FIELDS = ["epoch", "layer", "train_loss", "validation_loss", "train_primary",
          "validation_primary", "train_secondary", "validation_secondary",
          "mean_validation_primary", "improved_global", "epochs_without_improvement"]


def make_heads(base):
    heads = nn.ModuleDict({str(layer): nn.Sequential(
        nn.Linear(base.config.hidden_size, base.config.hidden_size), nn.Tanh(),
        nn.Linear(base.config.hidden_size, base.config.num_labels)) for layer in LAYERS})
    for head in heads.values():
        head[0].load_state_dict(base.bert.pooler.dense.state_dict())
        head[2].load_state_dict(base.classifier.state_dict())
    return heads


@torch.inference_mode()
def features(base, loader, device):
    base.eval()
    collected, labels = [], []
    for batch in loader:
        gold = batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = base.bert(**inputs, output_hidden_states=True, return_dict=True)
        collected.append(torch.stack([hidden[:, 0].detach().cpu().half()
                                      for hidden in outputs.hidden_states[1:]], dim=1))
        labels.append(gold)
    return TensorDataset(torch.cat(collected), torch.cat(labels))


def layer_loss(task, logits, labels):
    if task == "rte":
        return nn.functional.cross_entropy(logits, labels.long())
    return nn.functional.mse_loss(logits.squeeze(-1), labels.float())


@torch.inference_mode()
def evaluate(heads, loader, task, device):
    heads.eval()
    gold, predictions = [], {layer: [] for layer in LAYERS}
    loss_sum = {layer: 0.0 for layer in LAYERS}
    count = 0
    for feats, labels in loader:
        feats = feats.to(device).float()
        labels = labels.to(device)
        n = len(labels)
        count += n
        gold.append(labels.cpu().numpy())
        for layer in LAYERS:
            logits = heads[str(layer)](feats[:, layer - 1])
            loss_sum[layer] += float(layer_loss(task, logits, labels)) * n
            predictions[layer].append(logits.cpu().numpy())
    gold = np.concatenate(gold)
    result = {}
    for layer in LAYERS:
        logits = np.concatenate(predictions[layer])
        if task == "rte":
            predicted = logits.argmax(axis=-1)
            primary = float(f1_score(gold, predicted, average="macro", zero_division=0))
            secondary = float(accuracy_score(gold, predicted))
        else:
            predicted = np.clip(logits.reshape(-1), 1, 5)
            primary = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
            secondary = float(mean_squared_error(gold, predicted))
        result[layer] = {"loss": loss_sum[layer] / count, "primary": primary,
                         "secondary": secondary}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/100epochs/base"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/100epochs/heads"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if min(args.max_epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError("Hiperparâmetros devem ser positivos")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.base_dir / args.task / "best_model"
    destination = args.output_dir / args.task
    destination.mkdir(parents=True, exist_ok=True)
    base = AutoModelForSequenceClassification.from_pretrained(base_path).to(device).eval()
    if base.config.model_type != "bert" or base.config.num_hidden_layers != 12:
        raise ValueError("Esperado BERT com 12 camadas")
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    heads = make_heads(base).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    raw = load_dataset("json", data_files=files, cache_dir=str(args.data_dir / ".cache"))
    label = "entailment_judgment" if args.task == "rte" else "relatedness_score"
    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)
        result["labels"] = batch[label]
        return result
    data = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer)
    LOG.info("Extraindo CLS das 12 camadas com encoder congelado (%s)", device)
    train_data = features(base, DataLoader(data["train"], batch_size=args.batch_size, collate_fn=collator), device)
    val_data = features(base, DataLoader(data["validation"], batch_size=args.batch_size, collate_fn=collator), device)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    train_eval = DataLoader(train_data, batch_size=args.batch_size)
    val_eval = DataLoader(val_data, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate)
    history = []
    best_mean = float("-inf")
    best_epoch = None
    bad_epochs = 0
    best_by_layer = {layer: {"epoch": None, "score": float("-inf")} for layer in LAYERS}
    for epoch in range(1, args.max_epochs + 1):
        heads.train()
        online_losses = []
        for feats, labels in train_loader:
            feats, labels = feats.to(device).float(), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(layer_loss(args.task, heads[str(layer)](feats[:, layer - 1]), labels)
                       for layer in LAYERS) / len(LAYERS)
            loss.backward()
            optimizer.step()
            online_losses.append(float(loss.detach()))
        train_metrics = evaluate(heads, train_eval, args.task, device)
        val_metrics = evaluate(heads, val_eval, args.task, device)
        mean_score = float(np.mean([val_metrics[layer]["primary"] for layer in LAYERS]))
        improved = mean_score > best_mean
        bad_epochs = 0 if improved else bad_epochs + 1
        if improved:
            best_mean, best_epoch = mean_score, epoch
            checkpoint = destination / "best_heads.pt"
            temporary = destination / "best_heads.tmp"
            state = {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()}
            torch.save({"task": args.task, "layers": LAYERS, "state_dict": state,
                        "base_model": str(base_path.resolve()), "epoch": epoch,
                        "mean_validation_primary": mean_score}, temporary)
            temporary.replace(checkpoint)
        for layer in LAYERS:
            score = val_metrics[layer]["primary"]
            if score > best_by_layer[layer]["score"]:
                best_by_layer[layer] = {"epoch": epoch, "score": score}
            history.append({"epoch": epoch, "layer": layer,
                            "train_loss": train_metrics[layer]["loss"],
                            "validation_loss": val_metrics[layer]["loss"],
                            "train_primary": train_metrics[layer]["primary"],
                            "validation_primary": score,
                            "train_secondary": train_metrics[layer]["secondary"],
                            "validation_secondary": val_metrics[layer]["secondary"],
                            "mean_validation_primary": mean_score,
                            "improved_global": improved,
                            "epochs_without_improvement": bad_epochs})
        with (destination / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(history)
        progress = {"task": args.task, "last_epoch": epoch, "best_epoch": best_epoch,
                    "best_mean_validation": best_mean, "epochs_without_improvement": bad_epochs,
                    "patience": args.patience, "best_by_layer": best_by_layer,
                    "test_evaluated": False}
        (destination / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        LOG.info("Época %d | perda online=%.4f | média de validação=%.4f | melhor=%d (%.4f) | sem melhora=%d/%d",
                 epoch, np.mean(online_losses), mean_score, best_epoch, best_mean, bad_epochs, args.patience)
        if bad_epochs >= args.patience:
            break
    (destination / "summary.json").write_text(json.dumps({**progress, "epochs_completed": epoch,
        "selection_metric": "mean_macro_f1" if args.task == "rte" else "mean_pearson",
        "layers": LAYERS, "base_checkpoint": str(base_path.resolve()),
        "early_stopping_rule": "5 épocas consecutivas sem melhora da média nas 12 camadas"}, indent=2), encoding="utf-8")
    print((destination / "summary.json").read_text())


if __name__ == "__main__":
    main()
