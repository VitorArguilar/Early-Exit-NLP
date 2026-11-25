"""Monta heads selecionados na validação do estudo de 30 épocas."""

import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent / "backup"
SELECTION = ROOT / "final" / "tabela_epocas_selecionadas_30.csv"


def main():
    with SELECTION.open(newline="", encoding="utf-8-sig") as file:
        choices = list(csv.DictReader(file))
    for task in ("rte", "sts"):
        epochs = {int(row["camada"]): int(row["epoca_selecionada"])
                  for row in choices if row["tarefa"] == task and row["componente"] == "head"}
        state = {}
        for layer, epoch in epochs.items():
            source = ROOT / "outputs" / "long_overfit" / "heads" / task / f"epoch-{epoch}" / "heads.pt"
            payload = torch.load(source, map_location="cpu", weights_only=True)
            if payload["task"] != task:
                raise ValueError(f"Checkpoint incompatível: {source}")
            prefix = f"{layer}."
            state.update({key: value for key, value in payload["state_dict"].items() if key.startswith(prefix)})
        destination = ROOT / "outputs" / "long_overfit" / "selected_heads" / task
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({"task": task, "state_dict": state}, destination / "heads.pt")
        config = {"task": task, "base_model": str((ROOT / "outputs" / "epoch_sweep" / task / "epoch-7").resolve()),
                  "base_epoch": 7, "trained_layers": [3, 6, 9], "epochs": "por_camada",
                  "selected_head_epochs": epochs, "batch_size": 16, "learning_rate": 0.001,
                  "max_length": 128, "seed": 42, "encoder_frozen": True}
        (destination / "heads_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(task, epochs, destination)


if __name__ == "__main__":
    main()
