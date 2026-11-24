"""Gera figuras e tabelas de apresentação apenas dos resultados salvos no repositório."""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "early_exit_10epochs_3seeds"
FINAL = ROOT / "final"
COLORS = {"rte": "#2563a6", "sts": "#d47b25", "fixed": "#2563a6",
          "confidence": "#d47b25", "stability": "#8b4da8"}
METRIC = {"rte": ("macro_f1", "Macro F1"), "sts": ("pearson", "Pearson")}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_csv(name, rows):
    path = DEST / name
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(DEST / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(DEST / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table_figure(name, rows, columns, title, subtitle, widths=None, font_size=11):
    values = [[str(row.get(key, "")) for key, _ in columns] for row in rows]
    height = max(3.2, 1.25 + 0.38 * len(values))
    fig, ax = plt.subplots(figsize=(15.5, height))
    ax.axis("off")
    fig.text(0.04, 0.97, title, fontsize=19, weight="bold", va="top", color="#172033")
    fig.text(0.04, 0.91, subtitle, fontsize=10, va="top", color="#526075")
    table = ax.table(cellText=values, colLabels=[label for _, label in columns],
                     cellLoc="center", colLoc="center", loc="center", colWidths=widths,
                     bbox=[0.01, 0.02, 0.98, 0.76])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1e8")
        if row == 0:
            cell.set_facecolor("#26364a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f8")
    save_figure(fig, name)


def quality(row):
    return float(row[METRIC[row["task"]][0]])


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def curves_and_epochs():
    for task in ("rte", "sts"):
        for suffix in ("", "_normalizado"):
            for extension in ("png", "svg"):
                source = FINAL / f"grafico_30_epocas_{task}{suffix}.{extension}"
                if source.is_file():
                    shutil.copy2(source, DEST / source.name)
    selected = read_csv(FINAL / "tabela_epocas_selecionadas_30.csv")
    rows = []
    for row in selected:
        rows.append({"Tarefa": row["tarefa"].upper(),
                     "Componente": "Modelo base" if row["componente"] == "modelo_base" else "Head",
                     "Camada": row["camada"], "Época": row["epoca_selecionada"],
                     "Métrica": "Macro F1" if row["tarefa"] == "rte" else "Pearson",
                     "Validação": f'{float(row["qualidade_validacao"]):.4f}',
                     "Perda": f'{float(row["perda_validacao"]):.4f}',
                     "Encoder dos heads": "7" if row["componente"] == "head" else "—"})
    save_csv("tabela_epocas_escolhidas.csv", rows)
    table_figure("tabela_epocas_escolhidas", rows,
                 [(key, key) for key in rows[0]], "Épocas escolhidas na validação",
                 "Estudo exploratório de 30 épocas · seed 42 · heads treinados sobre o encoder de 7 épocas",
                 widths=[.09, .15, .08, .07, .11, .12, .09, .16], font_size=10)


def total_metrics(summary):
    order = {"fixed_3": 0, "fixed_6": 1, "fixed_9": 2, "fixed_12": 3,
             "confidence_selected": 4}
    rows = []
    for row in sorted(summary, key=lambda r: (r["task"], order[r["strategy_label"]])):
        task = row["task"]
        metric, label = METRIC[task]
        strategy = row["strategy_label"].replace("fixed_", "Fixa ").replace("confidence_selected", "Confiança")
        other = "accuracy" if task == "rte" else "mse"
        rows.append({"Tarefa": task.upper(), "Estratégia": strategy, "Métrica": label,
                     "Qualidade média": f'{float(row[metric + "_mean"]):.4f}',
                     "DP qualidade": f'{float(row[metric + "_std"]):.4f}',
                     "Acurácia/MSE": f'{float(row[other + "_mean"]):.4f}',
                     "Latência ms/par": f'{float(row["latency_ms_per_pair_mean"]):.3f}',
                     "Speedup": f'{float(row["speedup_mean"]):.2f}×',
                     "Camadas médias": f'{float(row["mean_layers_mean"]):.2f}', "Seeds": row["runs"]})
    save_csv("tabela_total_metricas_solucoes.csv", rows)
    table_figure("tabela_total_metricas_solucoes", rows,
                 [(key, key) for key in rows[0]], "Métricas das soluções no teste",
                 "Early Exit · encoder 7 épocas · heads 10 épocas · médias e desvio padrão (DP) de 3 seeds",
                 widths=[.075, .12, .10, .12, .105, .11, .125, .095, .11, .06], font_size=9)


def fixed(summary):
    data = [r for r in summary if r["strategy_label"].startswith("fixed_")]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, task in zip(axes, ("rte", "sts")):
        group = sorted((r for r in data if r["task"] == task), key=lambda r: int(r["strategy_label"].split("_")[1]))
        metric, label = METRIC[task]
        depths = [int(r["strategy_label"].split("_")[1]) for r in group]
        scores = [float(r[metric + "_mean"]) for r in group]
        errors = [float(r[metric + "_std"]) for r in group]
        ax.errorbar(depths, scores, yerr=errors, marker="o", capsize=4, color=COLORS[task], lw=2)
        ax.set(title=task.upper(), xlabel="Camada de saída", ylabel=f"{label} no teste", xticks=depths)
        ax.grid(alpha=.25)
        for r, depth, score in zip(group, depths, scores):
            rows.append({"Tarefa": task.upper(), "Camada": depth, "Métrica": label,
                         "Qualidade teste": f"{score:.4f}", "DP": f'{float(r[metric + "_std"]):.4f}',
                         "Latência ms/par": f'{float(r["latency_ms_per_pair_mean"]):.3f}',
                         "Speedup": f'{float(r["speedup_mean"]):.2f}×', "Seeds": r["runs"]})
    fig.suptitle("Saída fixa · qualidade por camada", fontsize=17, weight="bold")
    save_figure(fig, "grafico_saida_fixa")
    save_csv("tabela_saida_fixa.csv", rows)
    table_figure("tabela_saida_fixa", rows, [(key, key) for key in rows[0]],
                 "Saída fixa · resultados no teste", "Encoder 7 épocas · heads 10 épocas · 3 seeds",
                 widths=[.10, .10, .13, .16, .09, .17, .12, .08], font_size=10)


def confidence(validation, test):
    data = [r for r in validation if r["strategy"] == "confidence" and r["task"] == "rte"]
    selected = [r for r in test if r["strategy"] == "confidence" and r["task"] == "rte"]
    grouped = defaultdict(list)
    for row in data:
        grouped[float(row["threshold"])].append(row)
    rows = []
    for threshold, group in sorted(grouped.items()):
        score, dp = mean_std([quality(r) for r in group])
        latency, _ = mean_std([float(r["latency_ms_per_pair"]) for r in group])
        rows.append({"Divisão": "Validação", "Limiar": f"{threshold:.1f}",
                     "Macro F1 médio": f"{score:.4f}", "DP": f"{dp:.4f}",
                     "Latência ms/par": f"{latency:.3f}", "Seeds": len(group)})
    score, dp = mean_std([quality(r) for r in selected])
    latency, _ = mean_std([float(r["latency_ms_per_pair"]) for r in selected])
    selected_thresholds = ", ".join(f'{r["seed"]}: {float(r["threshold"]):.1f}'
                                   for r in sorted(selected, key=lambda r: int(r["seed"])))
    rows.append({"Divisão": "Teste selecionado", "Limiar": selected_thresholds,
                 "Macro F1 médio": f"{score:.4f}", "DP": f"{dp:.4f}",
                 "Latência ms/par": f"{latency:.3f}", "Seeds": len(selected)})
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x = [float(r["Limiar"]) for r in rows[:-1]]
    y = [float(r["Macro F1 médio"]) for r in rows[:-1]]
    e = [float(r["DP"]) for r in rows[:-1]]
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=COLORS["confidence"], label="Validação")
    ax.axhline(score, color="#26364a", linestyle="--", label="Teste: 42→0,5; 123→0,3; 2026→0,5")
    ax.set(title="Confiança · RTE", xlabel="Limiar de confiança", ylabel="Macro F1", xticks=x)
    ax.grid(alpha=.25)
    ax.legend()
    save_figure(fig, "grafico_confianca")
    save_csv("tabela_confianca.csv", rows)
    table_figure("tabela_confianca", rows, [(key, key) for key in rows[0]],
                 "Confiança · RTE", "STS não usa confiança de classes · teste somente dos limiares escolhidos na validação",
                 widths=[.22, .14, .19, .10, .20, .10], font_size=10)


def stability(validation):
    data = [r for r in validation if r["strategy"] == "stability"]
    grouped = defaultdict(list)
    for row in data:
        grouped[(row["task"], int(row["patience"]), row["tolerance"] or "—")].append(row)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, task in zip(axes, ("rte", "sts")):
        metric, label = METRIC[task]
        if task == "rte":
            series = sorted(((patience, group) for (t, patience, _), group in grouped.items()
                             if t == "rte"), key=lambda item: item[0])
            xs = [patience for patience, _ in series]
            ys = [mean_std([quality(r) for r in group])[0] for _, group in series]
            errors = [mean_std([quality(r) for r in group])[1] for _, group in series]
            ax.errorbar(xs, ys, yerr=errors, fmt="o-", capsize=4,
                        color=COLORS["stability"], label="Média ± DP (3 seeds)")
            ax.set_xticks(xs)
            ax.set(title="RTE", xlabel="Paciência", ylabel=f"{label} na validação")
            ax.grid(alpha=.25)
            ax.legend()
            continue
        for patience in (2, 3):
            series = sorted(((float(tolerance) if tolerance != "—" else float(patience), group)
                             for (t, p, tolerance), group in grouped.items() if t == task and p == patience),
                            key=lambda item: item[0])
            if series:
                xs = [x for x, _ in series]
                ys = [mean_std([quality(r) for r in group])[0] for _, group in series]
                ax.plot(xs, ys, "o-", label=f"Paciência {patience}")
        ax.set(title=task.upper(), xlabel="Paciência" if task == "rte" else "Tolerância",
               ylabel=f"{label} na validação")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Previsões consecutivas · validação", fontsize=17, weight="bold")
    save_figure(fig, "grafico_previsoes_consecutivas")
    for (task, patience, tolerance), group in sorted(grouped.items()):
        score, dp = mean_std([quality(r) for r in group])
        latency, _ = mean_std([float(r["latency_ms_per_pair"]) for r in group])
        rows.append({"Tarefa": task.upper(), "Paciência": patience, "Tolerância": tolerance,
                     "Métrica": METRIC[task][1], "Qualidade média": f"{score:.4f}",
                     "DP": f"{dp:.4f}", "Latência ms/par": f"{latency:.3f}",
                     "Teste": "não avaliado"})
    save_csv("tabela_previsoes_consecutivas.csv", rows)
    table_figure("tabela_previsoes_consecutivas", rows, [(key, key) for key in rows[0]],
                 "Previsões consecutivas · validação",
                 "3 seeds · nenhuma configuração foi selecionada para teste no protocolo histórico de 10 épocas",
                 widths=[.09, .11, .13, .12, .16, .07, .16, .15], font_size=10)


def main():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
                         "figure.facecolor": "white"})
    summary = read_csv(SOURCE / "summary.csv")
    validation = read_csv(SOURCE / "all_validation.csv")
    test = read_csv(SOURCE / "all_test.csv")
    curves_and_epochs()
    total_metrics(summary)
    fixed(summary)
    confidence(validation, test)
    stability(validation)
    print("Arquivos gerados em", DEST)


if __name__ == "__main__":
    main()
