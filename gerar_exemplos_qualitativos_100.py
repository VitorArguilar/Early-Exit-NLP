"""Exemplos qualitativos da rodada de 100 épocas, usando apenas inferência.

Os textos vêm da cópia do conjunto de teste na rodada anterior. Os rótulos,
previsões e camadas abaixo são recalculados com os checkpoints atuais.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = ROOT / "outputs" / "100epochs"
SOURCE = ROOT / "backup" / "outputs" / "early_exit_10epochs_3seeds"
DEST = ROOT / "apresentação"


def test_pairs(task, limit):
    path = SOURCE / "seed-42" / task / "evaluation" / "all_test_predictions.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["strategy"] == "fixed_3"]
    rng = random.Random(42)
    selected = rng.sample(rows, min(limit, len(rows)))
    if task == "rte":
        with path.open(newline="", encoding="utf-8-sig") as file:
            historical = [row for row in csv.DictReader(file)
                          if row["strategy"] == "confidence_selected"]
        groups = defaultdict(list)
        for row in historical:
            groups[(row["exit_layer"], row["error"])].append(row)
        for group in groups.values():
            selected.extend(sorted(group, key=lambda row: len(row["premise"]) +
                                                  len(row["hypothesis"]))[:12])
    return list({row["sentence_pair_id"]: row for row in selected}.values())


def model_outputs(checkpoints, task, pairs, batch_size):
    base_dir = checkpoints / "base" / task / "best_model"
    model = AutoModelForSequenceClassification.from_pretrained(base_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    heads = nn.ModuleDict({str(layer): nn.Sequential(
        nn.Linear(model.config.hidden_size, model.config.hidden_size), nn.Tanh(),
        nn.Linear(model.config.hidden_size, model.config.num_labels))
        for layer in range(1, 13)})
    payload = torch.load(checkpoints / "heads" / task / "best_heads.pt",
                         map_location="cpu", weights_only=True)
    if payload["task"] != task or tuple(payload["layers"]) != tuple(range(1, 13)):
        raise ValueError("Checkpoint de heads incompatível")
    heads.load_state_dict(payload["state_dict"])
    heads.eval()
    all_logits = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = tokenizer([row["premise"] for row in batch],
                               [row["hypothesis"] for row in batch],
                               truncation=True, max_length=128, padding=True,
                               return_tensors="pt")
            hidden_states = model.bert(**inputs, output_hidden_states=True,
                                       return_dict=True).hidden_states
            logits = [heads[str(layer)](hidden_states[layer][:, 0]).numpy()
                      for layer in range(1, 12)]
            final = model.classifier(model.bert.pooler(hidden_states[12])).numpy()
            all_logits.extend(np.stack([*logits, final], axis=1))
            print(task, min(start + batch_size, len(pairs)), "/", len(pairs), flush=True)
    return np.asarray(all_logits)


def exit_prediction(task, layer_logits, strategy, patience=None, tolerance=None,
                    threshold=None):
    streak = 1
    previous = None
    for layer, logits in enumerate(layer_logits, start=1):
        if strategy == "confidence":
            shifted = logits - logits.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / np.log(2)
            exit_now = entropy <= threshold
        else:
            current = int(logits.argmax()) if task == "rte" else float(logits[0])
            stable = (current == previous) if task == "rte" else (
                previous is not None and abs(current - previous) <= tolerance)
            streak = streak + 1 if stable else 1
            previous = current
            exit_now = streak >= patience
        if exit_now or layer == 12:
            prediction = int(logits.argmax()) if task == "rte" else float(np.clip(logits[0], 1, 5))
            return layer, prediction
    raise AssertionError("A camada 12 deve sempre produzir uma saída")


def test_metrics(task, strategy, patience=None, tolerance=None, threshold=None):
    path = DEFAULT_CHECKPOINTS / "evaluation" / task / "test.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["strategy"] != strategy:
                continue
            if strategy == "confidence" and float(row["threshold"]) != threshold:
                continue
            if strategy == "stability" and (int(row["patience"]) != patience or
                                            (task == "sts" and float(row["tolerance"]) != tolerance)):
                continue
            return row
    raise ValueError("Configuração sem métricas de teste")


def configurations(task):
    if task == "rte":
        return [("Confiança · limiar 0,5", "confidence", None, None, 0.5),
                ("Consecutivas · paciência 2", "stability", 2, None, None),
                ("Consecutivas · paciência 3", "stability", 3, None, None)]
    return [("Consecutivas · paciência 2 · tolerância 0,2", "stability", 2, 0.2, None),
            ("Consecutivas · paciência 3 · tolerância 0,2", "stability", 3, 0.2, None)]


def category(layer):
    return "Cedo" if layer <= 4 else "Meio" if layer <= 8 else "Tarde"


def quality_on_candidates(task, candidates):
    gold = np.asarray([row["gold"] for row in candidates])
    predicted = np.asarray([row["prediction"] for row in candidates])
    if task == "sts":
        return float(np.corrcoef(gold, predicted)[0, 1])
    scores = []
    for label in (0, 1):
        tp = np.sum((gold == label) & (predicted == label))
        fp = np.sum((gold != label) & (predicted == label))
        fn = np.sum((gold == label) & (predicted != label))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--candidate-limit", type=int, default=2448)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.candidate_limit < 1 or args.batch_size < 1:
        raise ValueError("Limites devem ser positivos")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = []
    used_pairs = set()
    for task in ("rte", "sts"):
        pairs = test_pairs(task, args.candidate_limit)
        logits = model_outputs(args.checkpoints, task, pairs, args.batch_size)
        for name, strategy, patience, tolerance, threshold in configurations(task):
            metrics = test_metrics(task, strategy, patience, tolerance, threshold)
            candidates = []
            for pair, sequence in zip(pairs, logits):
                layer, predicted = exit_prediction(task, sequence, strategy, patience,
                                                   tolerance, threshold)
                gold = int(pair["gold"]) if task == "rte" else float(pair["gold"])
                error = int(predicted != gold) if task == "rte" else abs(predicted - gold)
                correct = error == 0 if task == "rte" else error <= 0.5
                candidates.append({"task": task.upper(), "configuration": name,
                                   "pair_id": pair["sentence_pair_id"],
                                   "premise": pair["premise"], "hypothesis": pair["hypothesis"],
                                   "exit_group": category(layer), "exit_layer": layer,
                                   "gold": gold, "prediction": predicted,
                                   "absolute_error": error, "correct": correct,
                                   "result": "Acerto" if correct else "Erro",
                                   "test_quality": float(metrics["macro_f1"] if task == "rte" else metrics["pearson"]),
                                   "test_speedup": float(metrics["speedup"])})
            if len(pairs) == 2448:
                actual_quality = quality_on_candidates(task, candidates)
                expected_quality = float(metrics["macro_f1"] if task == "rte" else metrics["pearson"])
                if abs(actual_quality - expected_quality) > 0.002:
                    raise RuntimeError(f"Métrica divergente em {name}: {actual_quality:.4f} vs {expected_quality:.4f}")
            # Um acerto e um erro por configuração, com cobertura das saídas cedo,
            # no meio e tarde ao longo da tabela. Textos curtos ajudam na apresentação.
            preferred = ["Cedo", "Meio", "Tarde"]
            index = configurations(task).index((name, strategy, patience, tolerance, threshold))
            for correct, target in ((True, preferred[index % 3]),
                                    (False, preferred[(index + 1) % 3])):
                choices = [row for row in candidates if row["correct"] == correct]
                if not choices:
                    raise RuntimeError(f"Sem {'acerto' if correct else 'erro'} para {name}")
                chosen = min(choices, key=lambda row: (
                    row["exit_group"] != target,
                    (task, row["pair_id"]) in used_pairs,
                    len(row["premise"]) + len(row["hypothesis"]),
                    row["absolute_error"] if correct else -row["absolute_error"]))
                output.append(chosen)
                used_pairs.add((task, chosen["pair_id"]))
    DEST.mkdir(exist_ok=True)
    csv_path = DEST / "exemplos_qualitativos_100epocas.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    md = ["# Exemplos qualitativos · checkpoints de 100 épocas", "",
          "Pares do teste ASSIN2, recuperados da cópia textual da rodada anterior. "
          "Previsões e camadas recalculadas com os checkpoints atuais, apenas por inferência.", "",
          "RTE: 0 = sem implicação; 1 = implicação. Em STS, ‘acerto’ significa "
          "erro absoluto de até 0,5 ponto. Qualidade e aceleração são métricas "
          "agregadas do teste completo, não de cada par.", ""]
    for task in ("RTE", "STS"):
        md.extend([f"## {task}", "",
                   "| Configuração | Resultado | Saída | Premissa | Hipótese | Esperado | Previsto | Erro | Qualidade teste | Aceleração |",
                   "|---|---|---|---|---|---:|---:|---:|---:|---:|"])
        for row in (item for item in output if item["task"] == task):
            gold = f'{row["gold"]:.2f}' if task == "STS" else str(row["gold"])
            prediction = f'{row["prediction"]:.2f}' if task == "STS" else str(row["prediction"])
            error = f'{row["absolute_error"]:.2f}' if task == "STS" else str(row["absolute_error"])
            cells = [row["configuration"], row["result"],
                     f'{row["exit_group"]} (camada {row["exit_layer"]})',
                     row["premise"], row["hypothesis"], gold, prediction, error,
                     f'{row["test_quality"]:.3f}', f'{row["test_speedup"]:.2f}×']
            md.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
        md.append("")
    (DEST / "exemplos_qualitativos_100epocas.md").write_text("\n".join(md), encoding="utf-8")
    print("Gerados:", csv_path, DEST / "exemplos_qualitativos_100epocas.md")


if __name__ == "__main__":
    main()
