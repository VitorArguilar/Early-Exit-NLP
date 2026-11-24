"""Consolida todas as avaliações salvas e gera gráficos sem executar novos testes."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEST = Path(__file__).resolve().parent
MULTI = OUT / "early_exit_10epochs_3seeds"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def add(rows, source, family, split, task, seed, head_epochs=None, base_epochs=None):
    for record in read_csv(source):
        row = {
            "familia": family,
            "split": split,
            "tarefa": task,
            "seed": seed,
            "epocas_encoder": base_epochs,
            "epocas_heads": head_epochs,
            "arquivo_origem": source.relative_to(ROOT).as_posix(),
        }
        row.update(record)
        if family == "epocas_heads_exploratorio":
            row["epocas_heads"] = row["epoch"]
        if family == "epocas_heads_30":
            row["epocas_heads"] = row["epoch"]
        if family == "epocas_encoder_30":
            row["epocas_encoder"] = row["epoch"]
        if "exit_percent" in row and row["exit_percent"]:
            for layer, percent in json.loads(row.pop("exit_percent")).items():
                row[f"exit_percent_{layer}"] = percent
        rows.append(row)


def consolidate():
    rows = []
    for task in ("rte", "sts"):
        initial = OUT / task / "metrics.json"
        if initial.exists():
            data = json.loads(initial.read_text(encoding="utf-8"))
            for split in ("validation", "test"):
                if split in data:
                    rows.append({"familia": "baseline_inicial", "split": split, "tarefa": task,
                                 "seed": data.get("seed"), "epocas_encoder": data.get("epochs"),
                                 "arquivo_origem": initial.relative_to(ROOT).as_posix(), **data[split]})
        sweep = OUT / "epoch_sweep" / task / "comparison.json"
        if sweep.exists():
            data = json.loads(sweep.read_text(encoding="utf-8"))
            for record in data["comparison"]:
                rows.append({"familia": "epocas_encoder", "split": "validation", "tarefa": task,
                             "seed": data["seed"], "epocas_encoder": record["epoch"],
                             "selecionado": record["epoch"] == data["selected_epoch"],
                             "arquivo_origem": sweep.relative_to(ROOT).as_posix(), **record})
            rows.append({"familia": "epocas_encoder", "split": "test", "tarefa": task,
                         "seed": data["seed"], "epocas_encoder": data["selected_epoch"],
                         "selecionado": True, "arquivo_origem": sweep.relative_to(ROOT).as_posix(),
                         **data["test_selected_only"]})
        explore = OUT / "head_epoch_exploration" / task / "comparison.csv"
        if explore.exists():
            add(rows, explore, "epocas_heads_exploratorio", "validation", task, 42, base_epochs=7)
        standard = OUT / "early_exit" / task / "evaluation"
        for split in ("validation", "test"):
            source = standard / f"{split}.csv"
            if source.exists():
                add(rows, source, "early_exit_original", split, task, 42, head_epochs=5, base_epochs=7)
        long_base = OUT / "long_overfit" / "base" / task / "comparison.csv"
        if long_base.exists():
            add(rows, long_base, "epocas_encoder_30", "validation", task, 42)
        long_heads = OUT / "long_overfit" / "heads" / task / "comparison.csv"
        if long_heads.exists():
            add(rows, long_heads, "epocas_heads_30", "validation", task, 42, base_epochs=7)
    for split in ("validation", "test"):
        for record in read_csv(MULTI / f"all_{split}.csv"):
            row = {"familia": "early_exit_10ep_3seeds", "split": split,
                   "epocas_encoder": 7, "epocas_heads": 10,
                   "arquivo_origem": (MULTI / f"all_{split}.csv").relative_to(ROOT).as_posix()}
            row.update(record)
            row["tarefa"] = row.pop("task")
            rows.append(row)
    write_csv(DEST / "tabela_todos_os_testes.csv", rows)
    write_csv(DEST / "tabela_comparativa_tres_seeds.csv", read_csv(MULTI / "summary.csv"))
    return rows


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def charts(rows):
    plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 18, "figure.facecolor": "white"})
    colors = {"fixed": "#2477ae", "confidence": "#e38723", "stability": "#a44b9b"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, task in enumerate(("rte", "sts")):
        metric = "macro_f1" if task == "rte" else "pearson"
        for j, split in enumerate(("validation", "test")):
            ax = axes[i, j]
            subset = [r for r in rows if r["familia"] == "early_exit_10ep_3seeds"
                      and r["tarefa"] == task and r["split"] == split]
            for strategy in ("fixed", "confidence", "stability"):
                series = [r for r in subset if r["strategy"] == strategy]
                if not series:
                    continue
                ax.scatter([f(r, "latency_ms_per_pair") for r in series],
                           [f(r, metric) for r in series], c=colors[strategy], s=70,
                           alpha=0.68, label={"fixed": "Fixa", "confidence": "Confiança",
                                              "stability": "Estabilidade"}[strategy])
            ax.set(title=f"{task.upper()} · {'validação' if j == 0 else 'teste'} · {len(subset)} execuções",
                   xlabel="Latência (ms/par)", ylabel="Macro F1" if task == "rte" else "Pearson")
            ax.grid(alpha=.2)
            ax.legend(fontsize=15)
    fig.suptitle("Todas as configurações: 10 épocas dos heads, 3 seeds", fontsize=24)
    fig.savefig(DEST / "grafico_todos_early_exit.png", dpi=210)
    fig.savefig(DEST / "grafico_todos_early_exit.svg")
    plt.close(fig)

    for task in ("rte", "sts"):
        metric = "macro_f1" if task == "rte" else "pearson"
        for split in ("validation", "test"):
            subset = [r for r in rows if r["familia"] == "early_exit_10ep_3seeds"
                      and r["tarefa"] == task and r["split"] == split]
            fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
            for strategy, label in (("fixed", "Fixa"), ("confidence", "Confiança"),
                                    ("stability", "Estabilidade")):
                series = [r for r in subset if r["strategy"] == strategy]
                if series:
                    ax.scatter([f(r, "latency_ms_per_pair") for r in series],
                               [f(r, metric) for r in series], s=110,
                               color=colors[strategy], alpha=.75, label=label)
            ax.set(title=f"{task.upper()} · {'validação' if split == 'validation' else 'teste'}",
                   xlabel="Latência (ms/par)", ylabel="Macro F1" if task == "rte" else "Pearson")
            ax.grid(alpha=.2)
            ax.legend(loc="lower right", fontsize=18)
            name = f"grafico_early_exit_{task}_{split}_slide"
            fig.savefig(DEST / f"{name}.png", dpi=240)
            fig.savefig(DEST / f"{name}.svg")
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, task in zip(axes, ("rte", "sts")):
        metric = "macro_f1" if task == "rte" else "pearson"
        for split, marker in (("validation", "o"), ("test", "x")):
            subset = [r for r in rows if r["familia"] == "early_exit_original"
                      and r["tarefa"] == task and r["split"] == split]
            for strategy in ("fixed", "confidence", "stability"):
                series = [r for r in subset if r["strategy"] == strategy]
                if series:
                    ax.scatter([f(r, "latency_ms_per_pair") for r in series],
                               [f(r, metric) for r in series], color=colors[strategy],
                               marker=marker, s=90, label=f"{strategy} · {split}")
        ax.set(title=task.upper(), xlabel="Latência (ms/par)",
               ylabel="Macro F1" if task == "rte" else "Pearson")
        ax.grid(alpha=.2)
        ax.legend(fontsize=14)
    fig.suptitle("Experimento original: todas as configurações avaliadas")
    fig.savefig(DEST / "grafico_todos_early_exit_original.png", dpi=210)
    fig.savefig(DEST / "grafico_todos_early_exit_original.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, task in zip(axes, ("rte", "sts")):
        data = [r for r in rows if r["familia"] == "epocas_encoder" and r["split"] == "validation"
                and r["tarefa"] == task]
        ax.plot([f(r, "epoch") for r in data], [f(r, "validation_primary") for r in data],
                "o-", label="F1" if task == "rte" else "Pearson")
        ax.set(title=task.upper(), xlabel="Épocas do encoder", ylabel="F1" if task == "rte" else "Pearson",
               xticks=[3, 5, 7, 9])
        ax.grid(alpha=.2)
    fig.suptitle("Comparação completa das épocas do modelo base")
    fig.savefig(DEST / "grafico_todas_epocas_encoder.png", dpi=210)
    fig.savefig(DEST / "grafico_todas_epocas_encoder.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, task in zip(axes, ("rte", "sts")):
        data = [r for r in rows if r["familia"] == "epocas_heads_exploratorio" and r["tarefa"] == task]
        for layer in (3, 6, 9):
            series = [r for r in data if int(r["layer"]) == layer]
            ax.plot([f(r, "epoch") for r in series],
                    [f(r, "validation_macro_f1" if task == "rte" else "validation_pearson") for r in series],
                    "o-", label=f"Camada {layer}")
        ax.set(title=task.upper(), xlabel="Épocas dos heads", ylabel="F1" if task == "rte" else "Pearson",
               xticks=[5, 7, 10, 15])
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Exploração separada: todas as épocas dos heads")
    fig.savefig(DEST / "grafico_todas_epocas_heads.png", dpi=210)
    fig.savefig(DEST / "grafico_todas_epocas_heads.svg")
    plt.close(fig)


if __name__ == "__main__":
    records = consolidate()
    charts(records)
    print(f"{len(records)} linhas exportadas para {DEST / 'tabela_todos_os_testes.csv'}")
