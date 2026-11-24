"""Gera gráficos e resumo do estudo de 30 épocas a partir das métricas reais."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "long_overfit"
DEST = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_task(task):
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 13})
    base = read(SOURCE / "base" / task / "comparison.csv")
    heads = read(SOURCE / "heads" / task / "comparison.csv")
    metric = "macro_f1" if task == "rte" else "pearson"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot([int(r["epoch"]) for r in base], [float(r["validation_primary"]) for r in base], "o-")
    axes[0, 0].set(title="Modelo base · qualidade", xlabel="Épocas", ylabel=metric.upper())
    axes[0, 1].plot([int(r["epoch"]) for r in base], [float(r["train_loss"]) for r in base], "o-", label="Treino")
    axes[0, 1].plot([int(r["epoch"]) for r in base], [float(r["validation_loss"]) for r in base], "o-", label="Validação")
    axes[0, 1].set(title="Modelo base · perda", xlabel="Épocas", ylabel="Perda")
    for layer in (3, 6, 9):
        series = [r for r in heads if int(r["layer"]) == layer]
        axes[1, 0].plot([int(r["epoch"]) for r in series],
                        [float(r["validation_" + metric]) for r in series], "o-", label=f"Camada {layer}")
        axes[1, 1].plot([int(r["epoch"]) for r in series],
                        [float(r["validation_loss"]) for r in series], "o-", label=f"Camada {layer}")
    axes[1, 0].set(title="Heads · qualidade na validação", xlabel="Épocas", ylabel=metric.upper())
    axes[1, 1].set(title="Heads · perda na validação", xlabel="Épocas", ylabel="Perda")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xticks(range(2, 31, 4))
        if len(ax.lines) > 1:
            ax.legend(fontsize=11)
    fig.suptitle(f"Treino até 30 épocas · {task.upper()}", fontsize=23)
    fig.savefig(DEST / f"grafico_30_epocas_{task}.png", dpi=220)
    fig.savefig(DEST / f"grafico_30_epocas_{task}.svg")
    plt.close(fig)
    best_base = max(base, key=lambda r: (float(r["validation_primary"]), -int(r["epoch"])))
    best_heads = {str(layer): max((r for r in heads if int(r["layer"]) == layer),
                                  key=lambda r: (float(r["validation_" + metric]), -int(r["epoch"])))
                  for layer in (3, 6, 9)}
    return {"tarefa": task, "melhor_epoca_base": int(best_base["epoch"]),
            "melhor_qualidade_base": float(best_base["validation_primary"]),
            "ultima_qualidade_base": float(base[-1]["validation_primary"]),
            "perda_validacao_base_final": float(base[-1]["validation_loss"]),
            "melhores_heads": {layer: {"epoca": int(row["epoch"]),
                                         "qualidade": float(row["validation_" + metric])}
                                for layer, row in best_heads.items()}}


def write_combined_table():
    rows = []
    for task in ("rte", "sts"):
        metric = "macro_f1" if task == "rte" else "pearson"
        for item in read(SOURCE / "base" / task / "comparison.csv"):
            rows.append({"tarefa": task, "componente": "modelo_base", "camada": 12,
                         "epoca": item["epoch"], "metrica": metric,
                         "perda_treino": item["train_loss"],
                         "perda_validacao": item["validation_loss"],
                         "qualidade_validacao": item["validation_primary"],
                         "outra_metrica_validacao": item.get("validation_accuracy", item.get("validation_rmse", ""))})
        for item in read(SOURCE / "heads" / task / "comparison.csv"):
            rows.append({"tarefa": task, "componente": "head", "camada": item["layer"],
                         "epoca": item["epoch"], "metrica": metric,
                         "perda_treino": item["train_loss"],
                         "perda_validacao": item["validation_loss"],
                         "qualidade_treino": item["train_" + metric],
                         "qualidade_validacao": item["validation_" + metric],
                         "outra_metrica_validacao": item["validation_accuracy" if task == "rte" else "validation_mse"]})
    fields = ["tarefa", "componente", "camada", "epoca", "metrica", "perda_treino",
              "perda_validacao", "qualidade_treino", "qualidade_validacao", "outra_metrica_validacao"]
    with (DEST / "tabela_30_epocas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    summary = [plot_task(task) for task in ("rte", "sts")]
    count = write_combined_table()
    (DEST / "resumo_30_epocas.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{count} registros consolidados")
    print(json.dumps(summary, indent=2))
