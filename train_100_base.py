"""Fine-tuning do BERTimbau por até 100 épocas com early stopping."""

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
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments)


LOG = logging.getLogger(__name__)
FIELDS = ["epoch", "train_loss", "validation_loss", "validation_primary",
          "validation_secondary", "improved", "epochs_without_improvement"]


class TrackAndStop(TrainerCallback):
    def __init__(self, destination, tokenizer, metric, patience):
        self.destination = destination
        self.tokenizer = tokenizer
        self.metric = metric
        self.patience = patience
        self.rows = []
        self.best = float("-inf")
        self.best_epoch = None
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = round(state.epoch or 0)
        if epoch < 1:
            return control
        score = float(metrics[f"eval_{self.metric}"])
        improved = score > self.best
        self.bad_epochs = 0 if improved else self.bad_epochs + 1
        if improved:
            self.best, self.best_epoch = score, epoch
            model = kwargs["model"]
            weights = {key: tensor.detach().cpu().half() if tensor.is_floating_point() else tensor.detach().cpu()
                       for key, tensor in model.state_dict().items()}
            model.save_pretrained(self.destination / "best_model", state_dict=weights)
            self.tokenizer.save_pretrained(self.destination / "best_model")
            (self.destination / "best_model" / "checkpoint_meta.json").write_text(json.dumps({
                "epoch": epoch, "selection_metric": self.metric, "validation_score": score,
                "training_dtype": "float32", "saved_weights_dtype": "float16",
            }, indent=2), encoding="utf-8")
        train_logs = [entry for entry in state.log_history if "loss" in entry and "eval_loss" not in entry]
        train_loss = float(train_logs[-1]["loss"]) if train_logs else float("nan")
        secondary = "accuracy" if self.metric == "macro_f1" else "rmse"
        self.rows.append({"epoch": epoch, "train_loss": train_loss,
                          "validation_loss": float(metrics["eval_loss"]),
                          "validation_primary": score,
                          "validation_secondary": float(metrics[f"eval_{secondary}"]),
                          "improved": improved, "epochs_without_improvement": self.bad_epochs})
        with (self.destination / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        (self.destination / "progress.json").write_text(json.dumps({
            "last_epoch": epoch, "best_epoch": self.best_epoch, "best_validation": self.best,
            "epochs_without_improvement": self.bad_epochs, "patience": self.patience,
            "stopped_early": self.bad_epochs >= self.patience,
        }, indent=2), encoding="utf-8")
        LOG.info("Época %d | treino=%.4f validação=%.4f %s=%.4f | melhor=%d (%.4f) | sem melhora=%d/%d",
                 epoch, train_loss, metrics["eval_loss"], self.metric, score,
                 self.best_epoch, self.best, self.bad_epochs, self.patience)
        if self.bad_epochs >= self.patience:
            control.should_training_stop = True
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/100epochs/base"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if min(args.max_epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError("Hiperparâmetros devem ser positivos")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    destination = args.output_dir / args.task
    destination.mkdir(parents=True, exist_ok=True)
    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if any(not Path(path).is_file() for path in files.values()):
        raise FileNotFoundError("Faltam os splits ASSIN2 de treino/validação")
    raw = load_dataset("json", data_files=files, cache_dir=str(args.data_dir / ".cache"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label = "entailment_judgment" if args.task == "rte" else "relatedness_score"
    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)
        result["labels"] = batch[label]
        return result
    data = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2 if args.task == "rte" else 1,
        problem_type="single_label_classification" if args.task == "rte" else "regression")
    def metrics(prediction):
        logits, gold = prediction
        if args.task == "rte":
            predicted = np.argmax(logits, axis=-1)
            return {"macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(gold, predicted))}
        predicted = np.clip(logits.reshape(-1), 1, 5)
        gold = gold.reshape(-1)
        pearson = float(pearsonr(gold, predicted).statistic) if np.std(predicted) and np.std(gold) else 0.0
        return {"pearson": pearson, "rmse": float(np.sqrt(mean_squared_error(gold, predicted)))}
    primary = "macro_f1" if args.task == "rte" else "pearson"
    callback = TrackAndStop(destination, tokenizer, primary, args.patience)
    training = TrainingArguments(
        output_dir=str(destination / "trainer"), num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate, lr_scheduler_type="constant",
        eval_strategy="epoch", logging_strategy="epoch", save_strategy="no",
        report_to="none", disable_tqdm=True, seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(model=model, args=training, train_dataset=data["train"],
                      eval_dataset=data["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metrics, callbacks=[callback])
    LOG.info("%s | dispositivo=%s | treino=%d validação=%d | máximo=%d paciência=%d",
             args.task, trainer.args.device, len(raw["train"]), len(raw["validation"]),
             args.max_epochs, args.patience)
    trainer.train()
    summary = {"task": args.task, "seed": args.seed, "max_epochs": args.max_epochs,
               "patience": args.patience, "epochs_completed": len(callback.rows),
               "best_epoch": callback.best_epoch, "best_validation": callback.best,
               "selection_metric": primary, "checkpoint": "best_model",
               "test_evaluated": False, "training_dtype": "float32", "saved_weights_dtype": "float16"}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
