"""Consolida históricos e gera gráficos da rodada com early stopping."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "outputs" / "100epochs"
FIGURES = ROOT / "figures"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_base(task, rows, summary):
    x = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(x, [float(row["train_loss"]) for row in rows], "o-", label="Treino")
    axes[0].plot(x, [float(row["validation_loss"]) for row in rows], "o-", label="Validação")
    axes[0].set(title="Perda", xlabel="Época", ylabel="Perda")
    axes[0].legend()
    metric = "Macro F1" if task == "rte" else "Pearson"
    quality = [float(row["validation_primary"]) for row in rows]
    axes[1].plot(x, quality, "o-", color="#2477ae")
    best = summary["best_epoch"]
    axes[1].scatter([best], [quality[x.index(best)]], s=100, color="#d47b25", zorder=5,
                    label=f"Melhor: época {best}")
    axes[1].set(title="Qualidade na validação", xlabel="Época", ylabel=metric)
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(f"Modelo base {task.upper()} · early stopping (paciência 5)", fontsize=15)
    save(fig, f"base_{task}_historico")


def plot_heads(task, rows, summary):
    epochs = sorted({int(row["epoch"]) for row in rows})
    values = np.full((12, len(epochs)), np.nan)
    for row in rows:
        values[int(row["layer"]) - 1, epochs.index(int(row["epoch"]))] = float(row["validation_primary"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image = axes[0].imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                           extent=[epochs[0] - .5, epochs[-1] + .5, .5, 12.5])
    axes[0].set(title="Qualidade por camada e época", xlabel="Época", ylabel="Camada",
                yticks=range(1, 13))
    fig.colorbar(image, ax=axes[0], label="Macro F1" if task == "rte" else "Pearson")
    means = np.nanmean(values, axis=0)
    axes[1].plot(epochs, means, color="#2477ae", label="Média das 12 camadas")
    best = summary["best_epoch"]
    axes[1].scatter([best], [means[epochs.index(best)]], s=100, color="#d47b25", zorder=5,
                    label=f"Checkpoint: época {best}")
    axes[1].set(title="Critério de seleção do checkpoint", xlabel="Época",
                ylabel="Média do Macro F1" if task == "rte" else "Média do Pearson")
    axes[1].grid(alpha=.2)
    axes[1].legend()
    fig.suptitle(f"Heads {task.upper()} · 12 camadas · early stopping (paciência 5)", fontsize=15)
    save(fig, f"heads_{task}_historico")


def main():
    records = []
    layer_records = []
    for component in ("base", "heads"):
        for task in ("rte", "sts"):
            directory = ROOT / component / task
            summary = json.loads((directory / "summary.json").read_text())
            rows = read_csv(directory / "history.csv")
            expected = summary["epochs_completed"] * (12 if component == "heads" else 1)
            assert len(rows) == expected, (component, task, len(rows), expected)
            if component == "base":
                plot_base(task, rows, summary)
                score = summary["best_validation"]
            else:
                plot_heads(task, rows, summary)
                score = summary["best_mean_validation"]
                for layer in range(1, 13):
                    series = [row for row in rows if int(row["layer"]) == layer]
                    selected = next(row for row in series if int(row["epoch"]) == summary["best_epoch"])
                    peak = max(series, key=lambda row: (float(row["validation_primary"]), -int(row["epoch"])))
                    layer_records.append({"tarefa": task, "camada": layer,
                                          "epoca_checkpoint_conjunto": summary["best_epoch"],
                                          "validacao_no_checkpoint": selected["validation_primary"],
                                          "perda_validacao_no_checkpoint": selected["validation_loss"],
                                          "epoca_pico_individual": peak["epoch"],
                                          "pico_individual_validacao": peak["validation_primary"],
                                          "pico_individual_salvo": int(peak["epoch"]) == summary["best_epoch"]})
            records.append({"tarefa": task, "componente": component,
                            "epocas_executadas": summary["epochs_completed"],
                            "melhor_epoca": summary["best_epoch"],
                            "metrica_selecao": summary["selection_metric"],
                            "melhor_validacao": score, "paciencia": summary["patience"],
                            "teste_avaliado": (ROOT / "evaluation" / task / "test.csv").is_file()})
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (ROOT / "heads_by_layer.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(layer_records[0]))
        writer.writeheader()
        writer.writerows(layer_records)
    print("4 históricos conferidos; resumo e 4 figuras gerados")


if __name__ == "__main__":
    main()
