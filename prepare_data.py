"""Baixa ASSIN2 e registra os splits e a distribuição dos rótulos."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/assin2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nilc-nlp/assin2")
    expected = {"train": 6500, "validation": 500, "test": 2448}
    summary = {}
    for split, count in expected.items():
        rows = dataset[split]
        if len(rows) != count:
            raise ValueError(f"{split}: esperado {count}, recebido {len(rows)}")
        required = {"premise", "hypothesis", "relatedness_score", "entailment_judgment"}
        if not required.issubset(rows.column_names):
            raise ValueError(f"{split}: colunas ausentes: {required - set(rows.column_names)}")
        rows.to_json(args.output / f"{split}.jsonl", force_ascii=False)
        labels = rows["entailment_judgment"]
        scores = rows["relatedness_score"]
        if any(label not in (0, 1) for label in labels):
            raise ValueError(f"{split}: rótulo de inferência inválido")
        if any(not 1 <= score <= 5 for score in scores):
            raise ValueError(f"{split}: similaridade fora de [1, 5]")
        summary[split] = {"pairs": len(rows), "none": labels.count(0), "entailment": labels.count(1)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
