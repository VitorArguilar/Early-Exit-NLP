"""Consolida o teste dos heads selecionados na rodada de 30 épocas."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "long_overfit" / "selected_heads"
DEST = Path(__file__).resolve().parent / "tabela_teste_heads_selecionados_30.csv"


def main():
    rows = []
    for task in ("rte", "sts"):
        directory = SOURCE / task
        report = json.loads((directory / "evaluation" / "report.json").read_text())
        config = json.loads((directory / "heads_config.json").read_text())
        metric = "macro_f1" if task == "rte" else "pearson"
        for result in report["test"]:
            strategy = result["strategy"]
            layer = result.get("depth") if strategy == "fixed" else None
            rows.append({"tarefa": task, "estrategia": strategy, "camada_fixa": layer or "",
                         "limiar": result.get("threshold", ""), "paciencia": result.get("patience", ""),
                         "tolerancia": result.get("tolerance", ""), "metrica": metric,
                         "qualidade_teste": result[metric], "latencia_ms_par": result["latency_ms_per_pair"],
                         "speedup": result["speedup"], "camadas_medias": result["mean_layers"],
                         "epoca_encoder": 7, "epoca_head_3": config["selected_head_epochs"]["3"],
                         "epoca_head_6": config["selected_head_epochs"]["6"],
                         "epoca_head_9": config["selected_head_epochs"]["9"],
                         "repeticoes_latencia": report["repeats"]})
    with DEST.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} resultados em {DEST}")


if __name__ == "__main__":
    main()
