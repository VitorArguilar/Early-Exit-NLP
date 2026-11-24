"""Fine-tuning e avaliação do BERTimbau Base no ASSIN2."""

import argparse
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

logger = logging.getLogger(__name__)


class EpochLogger(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info("Iniciando época %d de %s", int(state.epoch or 0) + 1, args.num_train_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info("Época %d concluída", int(state.epoch or 0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rte", "sts"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("epochs, batch-size e max-length devem ser positivos")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = {split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    for path in files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path}: execute python prepare_data.py primeiro")
    raw = load_dataset("json", data_files=files)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label_col = "entailment_judgment" if args.task == "rte" else "relatedness_score"

    def preprocess(batch):
        encoded = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=args.max_length)
        encoded["labels"] = batch[label_col]
        return encoded

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2 if args.task == "rte" else 1,
        problem_type="single_label_classification" if args.task == "rte" else "regression",
    )

    def metrics(prediction):
        logits, gold = prediction
        if args.task == "rte":
            predicted = np.argmax(logits, axis=-1)
            return {
                "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(gold, predicted)),
            }
        predicted = np.clip(logits.reshape(-1), 1.0, 5.0)
        gold = gold.reshape(-1)
        pearson = float(pearsonr(gold, predicted).statistic) if np.std(predicted) > 0 and np.std(gold) > 0 else 0.0
        return {"pearson": pearson, "rmse": float(np.sqrt(mean_squared_error(gold, predicted)))}

    destination = args.output_dir / args.task
    training = TrainingArguments(
        output_dir=str(destination),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1" if args.task == "rte" else "pearson",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[EpochLogger()],
    )
    device = trainer.args.device
    logger.info("Dispositivo de treinamento: %s", device)
    if device.type == "cuda":
        logger.info("GPU em uso: %s (PyTorch %s, CUDA %s)", torch.cuda.get_device_name(device), torch.__version__, torch.version.cuda)
    else:
        logger.warning("CUDA não está em uso; o treinamento ocorrerá em %s", device)
    logger.info("Tarefa: %s | treino: %d pares | validação: %d pares | teste: %d pares", args.task, len(raw["train"]), len(raw["validation"]), len(raw["test"]))
    trainer.train()
    validation = trainer.evaluate(tokenized["validation"])
    test = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(str(destination / "best_model"))
    result = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "validation": validation,
        "test": test,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
