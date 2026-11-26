"""Monta tabelas e gráficos de apresentação somente com a execução de 100 épocas."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "100epochs"
DEST = ROOT / "apresentação"


def plot_method(frame, baseline, method, path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    x_column, xlabel = {
        "fixed": ("depth", "Camada de saída"),
        "confidence": ("threshold", "Limiar de entropia normalizada"),
        "stability": ("tolerance", "Tolerância"),
    }[method]
    for row, task in enumerate(("rte", "sts")):
        metric = "macro_f1" if task == "rte" else "pearson"
        metric_label = "Macro F1" if task == "rte" else "Pearson"
        task_data = frame[frame.task == task]
        for column, (value, label) in enumerate(((metric, metric_label), ("speedup", "Aceleração relativa à base"))):
            ax = axes[row, column]
            axis_label = "Paciência (previsões consecutivas)" if method == "stability" and task == "rte" else xlabel
            ax.set(title=f"{task.upper()} · {label}", xlabel=axis_label, ylabel=label)
            ax.grid(alpha=0.25)
            if task_data.empty:
                ax.text(0.5, 0.5, "Não se aplica a STS", transform=ax.transAxes,
                        ha="center", va="center", fontsize=18)
                ax.set_axis_off()
                continue
            if method == "stability":
                colors = {"validation": "#b3262d", "test": "#2563a6"}
                for patience in (2, 3):
                    for split, split_label in (("validation", "Validação"),
                                               ("test", "Teste")):
                        current = task_data[(task_data.patience == patience) &
                                            (task_data.split == split)].sort_values(
                                                "patience" if task == "rte" else "tolerance")
                        if current.empty:
                            continue
                        x = current.patience if task == "rte" else current.tolerance
                        ax.plot(x, current[value], linestyle="-",
                                marker="o" if patience == 2 else "*",
                                color=colors[split], linewidth=2.5,
                                markersize=8 if patience == 2 else 13,
                                label=f"Paciência {patience} · {split_label}")
                if task == "rte":
                    ax.set_xticks((2, 3))
                else:
                    ax.set_xticks((0.1, 0.2, 0.3, 0.5))
            else:
                for split, split_label, color, style in (
                    ("validation", "Validação", "#2563a6", "--"),
                    ("test", "Teste", "#d47b25", "-"),
                ):
                    current = task_data[task_data.split == split].sort_values(x_column)
                    if current.empty:
                        continue
                    ax.plot(current[x_column], current[value], linestyle=style, marker="o",
                            color=color, linewidth=2.5, markersize=7, label=split_label)
                    selected = current[current.selected_on_validation]
                    if not selected.empty:
                        ax.scatter(selected[x_column], selected[value], marker="*", s=280,
                                   color=color, edgecolor="#26364a", linewidth=0.7,
                                   zorder=5, label=f"{split_label}: selecionado")
            if column == 0:
                baseline_colors = ({"validation": "#b3262d", "test": "#2563a6"}
                                   if method == "stability" else
                                   {"validation": "#334155", "test": "#94a3b8"})
                for split, split_label in (("validation", "Validação"),
                                           ("test", "Teste")):
                    reference = baseline[(baseline.task == task) & (baseline.split == split)]
                    if not reference.empty:
                        score = reference.iloc[0][metric]
                        ax.axhline(score, linestyle="--", color=baseline_colors[split], linewidth=2.2,
                                   label=f"Modelo base · {split_label}: {score:.3f}")
            if method == "fixed":
                ax.set_xticks(sorted(task_data[x_column].dropna().unique()))
            ax.legend(loc="best")
    fig.suptitle({"fixed": "Saída fixa", "confidence": "Confiança", "stability": "Previsões consecutivas"}[method],
                 fontsize=23, fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def configuration_label(row):
    if row.strategy == "base":
        return "Modelo base"
    if row.strategy == "fixed":
        return f"Saída fixa · camada {int(row.depth)}"
    if row.strategy == "confidence":
        return f"Confiança {row.threshold:.1f}"
    label = f"Paciência {int(row.patience)}"
    if row.task == "sts":
        label += f" · tol. {row.tolerance:.1f}"
    return label


def plot_exit_layers(frame, task, path):
    frame = frame[(frame.strategy != "fixed") & (frame.task == task)]
    fig, axes = plt.subplots(1, 2, figsize=(25, 12 if task == "rte" else 11), layout="constrained")
    for ax, (split, split_label) in zip(axes, (("validation", "Validação"), ("test", "Teste"))):
        current = frame[frame.split == split]
        matrix = [[json.loads(record.exit_percent)[str(layer)] for layer in range(1, 13)]
                  for record in current.itertuples()]
        image = ax.imshow(matrix, cmap="Blues", norm=PowerNorm(gamma=0.5, vmin=0, vmax=100),
                          aspect="auto", interpolation="nearest")
        ax.set(title=split_label, xlabel="Camada de saída")
        ax.title.set_fontsize(25)
        ax.xaxis.label.set_fontsize(21)
        ax.set_xticks(range(12), labels=range(1, 13))
        ax.set_yticks(range(len(current)), labels=[configuration_label(record)
                                                   for record in current.itertuples()])
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", labelsize=18)
        for index, strategy in enumerate(current.strategy.iloc[1:], start=1):
            if strategy != current.strategy.iloc[index - 1]:
                ax.axhline(index - 0.5, color="#26364a", linewidth=1.5)
    fig.suptitle(f"Distribuição das saídas por camada · {task.upper()}",
                 fontsize=31, fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, ticks=[0, 1, 5, 10, 25, 50, 75, 100])
    colorbar.set_label("Exemplos que saíram na camada (%)", fontsize=21)
    colorbar.ax.tick_params(labelsize=17)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 18,
                         "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 12})
    DEST.mkdir(exist_ok=True)
    figures = SOURCE / "figures"
    for task in ("rte", "sts"):
        for component in ("base", "heads"):
            for suffix in ("png", "svg"):
                name = f"{component}_{task}_historico.{suffix}"
                shutil.copy2(figures / name, DEST / name)
    summary = pd.read_csv(SOURCE / "summary.csv")
    summary.to_csv(DEST / "tabela_epocas_escolhidas.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(SOURCE / "heads_by_layer.csv", DEST / "tabela_epocas_heads_por_camada.csv")
    all_rows = []
    for task in ("rte", "sts"):
        for split in ("validation", "test"):
            all_rows.append(pd.read_csv(SOURCE / "evaluation" / task / f"{split}.csv"))
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(DEST / "tabela_total_metricas_solucoes.csv", index=False, encoding="utf-8-sig")
    selected = results[(results.split == "test") & ((results.strategy == "base") | results.selected_on_validation)]
    selected.to_csv(DEST / "tabela_resultados_selecionados.csv", index=False, encoding="utf-8-sig")
    for method, name in (("fixed", "saida_fixa"), ("confidence", "confianca"),
                         ("stability", "previsoes_consecutivas")):
        subset = results[results.strategy == method].copy()
        subset.to_csv(DEST / f"tabela_{name}.csv", index=False, encoding="utf-8-sig")
        plot_method(subset, results[results.strategy == "base"], method,
                    DEST / f"grafico_{name}")
    for task in ("rte", "sts"):
        plot_exit_layers(results, task, DEST / f"grafico_camadas_saida_{task}")
    (DEST / "README.md").write_text(
        "# Resultados dos novos treinos\n\n"
        "Todos os dados vêm exclusivamente de `outputs/100epochs`. Os históricos mostram "
        "treino e validação dos checkpoints novos. A época dos heads é selecionada pela "
        "média da métrica de validação nas 12 camadas; o pico individual de cada camada "
        "é apenas informativo e não representa outro checkpoint.\n\n"
        "As estrelas nos gráficos indicam a configuração selecionada usando apenas "
        "validação. A tabela total inclui todos os candidatos avaliados no teste para "
        "análise exploratória. A aceleração é a latência da base dividida pela latência "
        "da configuração, medidas nesta execução. Confiança só se aplica a RTE, pois "
        "STS é regressão. Cada gráfico apresenta RTE na primeira linha e STS "
        "na segunda, com qualidade e aceleração nas colunas. Validação e teste "
        "aparecem juntos em cada painel. Em previsões consecutivas, RTE usa "
        "paciência no eixo horizontal e STS usa tolerância, com curvas separadas "
        "para paciência 2 e 3. Nesse gráfico, vermelho indica validação e azul indica teste; "
        "círculos mostram paciência 2 e estrelas mostram paciência 3. O mapa de calor "
        "mostra a porcentagem de exemplos que saíram em cada camada para as "
        "configurações de confiança e previsões consecutivas em validação e teste, "
        "além do modelo base. As linhas horizontais tracejadas nos gráficos de "
        "qualidade indicam a qualidade do modelo base em validação e teste.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
