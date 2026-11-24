"""Monta tabelas e gráficos de apresentação somente com a execução de 100 épocas."""

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "100epochs"
DEST = ROOT / "apresentação"


def plot_method(frame, method, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for task, group in frame.groupby("task"):
        metric = "macro_f1" if task == "rte" else "pearson"
        if method == "fixed":
            x = group["depth"]
            xlabel = "Camada de saída"
        elif method == "confidence":
            x = group["threshold"]
            xlabel = "Limiar de entropia normalizada"
        else:
            x = group["mean_layers"]
            xlabel = "Média de camadas executadas"
        for split, style in (("validation", "--"), ("test", "-")):
            current = group[group.split == split]
            if current.empty:
                continue
            x_current = x.loc[current.index]
            order = x_current.argsort()
            label = f"{task.upper()} {split}"
            axes[0].plot(x_current.iloc[order], current[metric].iloc[order], style, marker="o", label=label)
            axes[1].plot(x_current.iloc[order], current.speedup.iloc[order], style, marker="o", label=label)
            selected = current[current.selected_on_validation]
            if not selected.empty:
                axes[0].scatter(x.loc[selected.index], selected[metric], marker="*", s=170, zorder=5)
                axes[1].scatter(x.loc[selected.index], selected.speedup, marker="*", s=170, zorder=5)
    for axis, ylabel in zip(axes, ("Macro-F1 (RTE) / Pearson (STS)", "Aceleração relativa à base")):
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle({"fixed": "Saída fixa", "confidence": "Confiança", "stability": "Previsões consecutivas"}[method])
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
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
        plot_method(subset, method, DEST / f"grafico_{name}")
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
        "STS é regressão.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
