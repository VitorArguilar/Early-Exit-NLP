"""Estuda uma trajetória de 30 épocas do BERTimbau, com registros a cada 2.

O teste permanece reservado: esta execução usa apenas treino e validação.
"""

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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


EPOCHS = tuple(range(2, 31, 2))
LOGGER = logging.getLogger(__name__)


class EpochCheckpoint(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_begin(self, args, state, control, **kwargs):
        LOGGER.info("Iniciando época %d de 30", int(state.epoch or 0) + 1)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch or 0)
        LOGGER.info("Época %d concluída", epoch)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = round(state.epoch or 0)
        if epoch in EPOCHS:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "progress.json").write_text(
                json.dumps({"epoch": epoch, "history": state.log_history, "latest": metrics}, indent=2),
                encoding="utf-8",
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/long_overfit/base"))
    parser.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_metrics(task):
    def compute(prediction):
        logits, gold = prediction
        if task == "rte":
            predicted = np.argmax(logits, axis=-1)
            return {
                "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(gold, predicted)),
            }
        predicted = np.clip(logits.reshape(-1), 1.0, 5.0)
        gold = gold.reshape(-1)
        pearson = float(pearsonr(gold, predicted).statistic) if np.std(predicted) > 0 and np.std(gold) > 0 else 0.0
        return {"pearson": pearson, "rmse": float(np.sqrt(mean_squared_error(gold, predicted)))}

    return compute


def history_for_epoch(history, epoch, prefix):
    matches = [row for row in history if "epoch" in row and abs(row["epoch"] - epoch) < 1e-6 and f"{prefix}_loss" in row]
    if len(matches) != 1:
        raise RuntimeError(f"Esperava uma métrica {prefix} na época {epoch}; encontrei {len(matches)}")
    return matches[0]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1 or args.learning_rate <= 0:
        raise ValueError("batch-size, max-length e learning-rate devem ser positivos")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    missing = [path for path in files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset ausente: {missing}. Execute prepare_data.py")
    raw = load_dataset("json", data_files=files, cache_dir=str(args.data_dir / ".cache"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label_column = "entailment_judgment" if args.task == "rte" else "relatedness_score"

    def preprocess(batch):
        encoded = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=args.max_length)
        encoded["labels"] = batch[label_column]
        return encoded

    data = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2 if args.task == "rte" else 1,
        problem_type="single_label_classification" if args.task == "rte" else "regression",
    )
    destination = args.output_dir / args.task
    destination.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(destination / "trainer"),
        num_train_epochs=max(EPOCHS),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_metrics(args.task),
        callbacks=[EpochCheckpoint(destination)],
    )
    device = trainer.args.device
    LOGGER.info("Dispositivo: %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU: %s | PyTorch: %s | CUDA: %s", torch.cuda.get_device_name(device), torch.__version__, torch.version.cuda)
    else:
        LOGGER.warning("CUDA indisponível: treinamento em %s", device)
    LOGGER.info("Tarefa: %s | treino: %d | validação: %d | teste: %d", args.task, len(raw["train"]), len(raw["validation"]), len(raw["test"]))
    trainer.train()

    primary = "macro_f1" if args.task == "rte" else "pearson"
    rows = []
    for epoch in EPOCHS:
        train_matches = [row for row in trainer.state.log_history if "loss" in row and "eval_loss" not in row and abs(row.get("epoch", -1) - epoch) < 1e-6]
        if len(train_matches) != 1:
            raise RuntimeError(f"Esperava uma perda de treino na época {epoch}; encontrei {len(train_matches)}")
        eval_log = history_for_epoch(trainer.state.log_history, epoch, "eval")
        row = {
            "epoch": epoch,
            "train_loss": float(train_matches[0]["loss"]),
            "validation_loss": float(eval_log["eval_loss"]),
            "validation_primary": float(eval_log[f"eval_{primary}"]),
        }
        if args.task == "rte":
            row["validation_accuracy"] = float(eval_log["eval_accuracy"])
        else:
            row["validation_rmse"] = float(eval_log["eval_rmse"])
        row["loss_gap"] = row["validation_loss"] - row["train_loss"]
        rows.append(row)

    # Em caso de empate, prefere a época anterior.
    selected = max(rows, key=lambda row: (row["validation_primary"], -row["epoch"]))
    LOGGER.info("Melhor validação: época %d (%s=%.4f)", selected["epoch"], primary, selected["validation_primary"])
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler": "constant",
        "max_length": args.max_length,
        "selection_metric": primary,
        "comparison": rows,
        "selected_epoch": selected["epoch"],
        "test_evaluated": False,
        "interpretation": "Compare a queda da métrica de validação e o aumento de validation_loss apesar da queda de train_loss. Isso sugere overfitting, mas não o comprova sozinho.",
    }
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
