"""Seleciona épocas por métrica de validação e confere tabelas da rodada de 30 épocas."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "long_overfit"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def selected():
    result = []
    for task in ("rte", "sts"):
        metric = "macro_f1" if task == "rte" else "pearson"
        for component, layers in (("modelo_base", (12,)), ("head", (3, 6, 9))):
            source = SOURCE / ("base" if component == "modelo_base" else "heads") / task / "comparison.csv"
            records = read(source)
            for layer in layers:
                group = records if component == "modelo_base" else [r for r in records if int(r["layer"]) == layer]
                field = "validation_primary" if component == "modelo_base" else "validation_" + metric
                best = max(group, key=lambda r: (float(r[field]), -int(r["epoch"])))
                result.append({"tarefa": task, "componente": component, "camada": layer,
                               "epoca_selecionada": int(best["epoch"]), "metrica": metric,
                               "qualidade_validacao": best[field], "perda_validacao": best["validation_loss"],
                               "checkpoint_base_epocas": int(best["epoch"]) if component == "modelo_base" else 7,
                               "arquivo_origem": source.relative_to(ROOT).as_posix()})
    return result


def main():
    rows = selected()
    table = read(DEST / "tabela_30_epocas.csv")
    consolidated = read(DEST / "tabela_todos_os_testes.csv")
    summary = {r["tarefa"]: r for r in json.loads((DEST / "resumo_30_epocas.json").read_text())}
    for row in rows:
        task, component, layer = row["tarefa"], row["componente"], int(row["camada"])
        match = [r for r in table if r["tarefa"] == task and r["componente"] == component
                 and int(r["camada"]) == layer and int(r["epoca"]) == row["epoca_selecionada"]]
        assert len(match) == 1
        assert abs(float(match[0]["qualidade_validacao"]) - float(row["qualidade_validacao"])) < 1e-12
        family = "epocas_encoder_30" if component == "modelo_base" else "epocas_heads_30"
        match = [r for r in consolidated if r["familia"] == family and r["tarefa"] == task
                 and int(r["epoch"]) == row["epoca_selecionada"]
                 and (component == "modelo_base" or int(r["layer"]) == layer)]
        assert len(match) == 1
        field = "validation_primary" if component == "modelo_base" else "validation_" + row["metrica"]
        assert abs(float(match[0][field]) - float(row["qualidade_validacao"])) < 1e-12
        entry = summary[task]
        saved = (entry["melhor_epoca_base"], entry["melhor_qualidade_base"]) if component == "modelo_base" else (
            entry["melhores_heads"][str(layer)]["epoca"], entry["melhores_heads"][str(layer)]["qualidade"])
        assert saved[0] == row["epoca_selecionada"]
        assert abs(saved[1] - float(row["qualidade_validacao"])) < 1e-12
    output = DEST / "tabela_epocas_selecionadas_30.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} seleções conferidas em tabela_30_epocas.csv, tabela_todos_os_testes.csv e resumo_30_epocas.json")
    print(output)


if __name__ == "__main__":
    main()
