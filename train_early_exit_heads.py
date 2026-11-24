"""Treina somente as saídas intermediárias de um checkpoint BERTimbau já ajustado."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from early_exit import EXITS, EarlyExitBert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rte", "sts"), required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assin2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/early_exit"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size e learning-rate devem ser positivos")
    model_path = args.model or Path("outputs/epoch_sweep") / args.task / "epoch-7"
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint ausente: {model_path}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Dispositivo: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    model = EarlyExitBert(model_path).to(device)
    if model.task != args.task:
        raise ValueError("O checkpoint não corresponde à tarefa solicitada")
    model.freeze_base()
    tokenizer = AutoTokenizer.from_pretrained("outputs/epoch_sweep/" + args.task + "/best_model")
    train_file = args.data_dir / "train.jsonl"
    if not train_file.is_file():
        raise FileNotFoundError(f"Dataset ausente: {train_file}")
    dataset = load_dataset("json", data_files={"train": str(train_file)}, cache_dir=str(args.data_dir / ".cache"))["train"]
    column = "entailment_judgment" if args.task == "rte" else "relatedness_score"

    def tokenize(batch):
        result = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=args.max_length)
        result["labels"] = batch[column]
        return result

    encoded = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=DataCollatorWithPadding(tokenizer))
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.heads.train()
        model.base.eval()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_heads(**batch)
            if args.task == "rte":
                loss = sum(torch.nn.functional.cross_entropy(logits[layer], labels.long()) for layer in EXITS[:-1]) / 3
            else:
                loss = sum(torch.nn.functional.mse_loss(logits[layer].squeeze(-1), labels.float()) for layer in EXITS[:-1]) / 3
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(loader)
        losses.append(mean_loss)
        logging.info("Época %d/%d concluída | perda dos heads: %.5f", epoch, args.epochs, mean_loss)

    destination = args.output_dir / args.task
    destination.mkdir(parents=True, exist_ok=True)
    model.save_heads(destination / "heads.pt")
    metadata = {
        "task": args.task,
        "base_model": str(model_path.resolve()),
        "base_epoch": 7,
        "trained_layers": list(EXITS[:-1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_loss": losses,
        "encoder_frozen": True,
    }
    (destination / "heads_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Heads salvos em %s", destination / "heads.pt")


if __name__ == "__main__":
    main()
