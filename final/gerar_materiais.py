"""Gera figuras e tabelas da apresentação a partir dos CSVs agregados."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "early_exit_10epochs_3seeds"
DEST = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup(rows, task, strategy):
    return next(row for row in rows if row["task"] == task and row["strategy_label"] == strategy)


def number(row, field):
    return float(row[field])


def main():
    summary = read_csv(SOURCE / "summary.csv")
    selections = read_csv(SOURCE / "selection_by_seed.csv")
    selected = [
        ("RTE", "Modelo completo (12)", lookup(summary, "rte", "fixed_12"), "macro_f1"),
        ("RTE", "Saída fixa (9)", lookup(summary, "rte", "fixed_9"), "macro_f1"),
        ("RTE", "Confiança selecionada", lookup(summary, "rte", "confidence_selected"), "macro_f1"),
        ("STS", "Modelo completo (12)", lookup(summary, "sts", "fixed_12"), "pearson"),
        ("STS", "Saída fixa (9)", lookup(summary, "sts", "fixed_9"), "pearson"),
    ]
    table = []
    for task, strategy, row, metric in selected:
        table.append({
            "Tarefa": task,
            "Estratégia": strategy,
            "Métrica": "Macro F1" if task == "RTE" else "Pearson",
            "Qualidade média": f"{number(row, metric + '_mean'):.4f}",
            "Desvio qualidade": f"{number(row, metric + '_std'):.4f}",
            "Latência média (ms/par)": f"{number(row, 'latency_ms_per_pair_mean'):.3f}",
            "Speedup médio": f"{number(row, 'speedup_mean'):.2f}",
            "Camadas médias": f"{number(row, 'mean_layers_mean'):.2f}",
        })
    write_csv(DEST / "tabela_principal.csv", table)
    write_csv(DEST / "tabela_selecao_seeds.csv", [
        {"Tarefa": row["task"].upper(), "Seed": row["seed"], "Estratégia": row["strategy"], "Selecionada": row["selected"], "Parâmetros": row["parameters"]}
        for row in selections
    ])

    plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 18, "figure.facecolor": "white"})
    colors = {"fixed_12": "#26364a", "fixed_9": "#2b73b9", "confidence_selected": "#dc7c27"}
    labels = {"fixed_12": "Completo (12)", "fixed_9": "Fixo (9)", "confidence_selected": "Confiança"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, task, metric in ((axes[0], "rte", "macro_f1"), (axes[1], "sts", "pearson")):
        strategies = ("fixed_12", "fixed_9", "confidence_selected") if task == "rte" else ("fixed_12", "fixed_9")
        for strategy in strategies:
            row = lookup(summary, task, strategy)
            axis.errorbar(
                number(row, "speedup_mean"), number(row, metric + "_mean"),
                xerr=number(row, "speedup_std"), yerr=number(row, metric + "_std"),
                fmt="o", markersize=13, capsize=6, color=colors[strategy], label=labels[strategy],
            )
        baseline = number(lookup(summary, task, "fixed_12"), metric + "_mean")
        margin = 0.01 if task == "rte" else 0.02
        axis.axhline(baseline - margin, color="#888888", linestyle="--", linewidth=1, label="Margem de qualidade")
        axis.set(xlabel="Speedup médio (×)", ylabel="Macro F1" if task == "rte" else "Pearson", title="Inferência textual (RTE)" if task == "rte" else "Similaridade (STS)")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower left", fontsize=17)
    axes[0].set(xlim=(0.96, 1.39), ylim=(0.845, 0.900))
    axes[1].set(xlim=(0.96, 1.39), ylim=(0.805, 0.842))
    fig.suptitle("Early Exit: qualidade versus velocidade", fontsize=26, weight="bold")
    fig.tight_layout()
    fig.savefig(DEST / "grafico_tradeoff.png", dpi=240)
    fig.savefig(DEST / "grafico_tradeoff.svg")
    plt.close(fig)

    display_rows = [
        [row["Tarefa"], row["Estratégia"], f"{row['Qualidade média']} ± {row['Desvio qualidade']}", f"{row['Latência média (ms/par)']}", f"{row['Speedup médio']}×"]
        for row in table
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    rendered = ax.table(
        cellText=display_rows,
        colLabels=["Tarefa", "Estratégia", "F1 / Pearson", "ms/par", "Speedup"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.10, 0.30, 0.25, 0.17, 0.18],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(19)
    rendered.scale(1, 2.4)
    for (row_index, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("#d6dce3")
        if row_index == 0:
            cell.set_facecolor("#26364a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f2f5f8")
    fig.tight_layout()
    fig.savefig(DEST / "tabela_principal.png", dpi=240, bbox_inches="tight")
    fig.savefig(DEST / "tabela_principal.svg", bbox_inches="tight")
    plt.close(fig)

    all_test = read_csv(SOURCE / "all_test.csv")
    strategy_colors = {"fixed": "#2b73b9", "confidence": "#dc7c27", "stability": "#9b4e9d"}
    for task in ("rte", "sts"):
        fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        metric = "macro_f1" if task == "rte" else "pearson"
        task_rows = [row for row in all_test if row["task"] == task]
        for strategy in ("fixed", "confidence", "stability"):
            group = [row for row in task_rows if row["strategy"] == strategy]
            if group:
                ax.scatter([number(row, "latency_ms_per_pair") for row in group],
                           [number(row, metric) for row in group], s=110,
                           color=strategy_colors[strategy], label=strategy.capitalize())
        ax.set(title=f"Todas as execuções no teste · {task.upper()}",
               xlabel="Latência média (ms/par)", ylabel="Macro F1" if task == "rte" else "Pearson")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=18)
        fig.savefig(DEST / f"grafico_completo_{task}.png", dpi=240)
        fig.savefig(DEST / f"grafico_completo_{task}.svg")
        plt.close(fig)
    print("Materiais gerados em", DEST)


if __name__ == "__main__":
    main()
