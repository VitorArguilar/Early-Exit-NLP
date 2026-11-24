"""Inferência BERTimbau com parada real nas camadas 3, 6, 9 e 12."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


EXITS = (3, 6, 9, 12)


@dataclass
class ExitResult:
    predictions: torch.Tensor
    layers: torch.Tensor


class EarlyExitBert(nn.Module):
    """Mantém o BERT e a cabeça final intactos; acrescenta três heads lineares."""

    def __init__(self, model_path: str | Path):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(model_path)
        if self.base.config.model_type != "bert" or self.base.config.num_hidden_layers != 12:
            raise ValueError("Esta implementação requer BERT Base com 12 camadas")
        if self.base.bert.pooler is None:
            raise ValueError("O checkpoint precisa conter o pooler do BERT")
        self.task = "rte" if self.base.config.num_labels == 2 else "sts"
        self.heads = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(self.base.config.hidden_size, self.base.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.base.config.hidden_size, self.base.config.num_labels),
            ) for layer in EXITS[:-1]
        })
        for head in self.heads.values():
            head[0].load_state_dict(self.base.bert.pooler.dense.state_dict())
            head[2].load_state_dict(self.base.classifier.state_dict())

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def save_heads(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"task": self.task, "state_dict": self.heads.state_dict()}, path)

    def load_heads(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["task"] != self.task:
            raise ValueError("Heads incompatíveis com a tarefa do modelo")
        self.heads.load_state_dict(payload["state_dict"])

    def _start(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_mask = self.base.bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        hidden = self.base.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        return hidden, extended_mask

    def _logits(self, hidden, layer):
        if layer == 12:
            pooled = self.base.bert.pooler(hidden)
            return self.base.classifier(self.base.dropout(pooled))
        return self.heads[str(layer)](hidden[:, 0])

    def forward_heads(self, input_ids, attention_mask, token_type_ids=None):
        """Treino de heads: computa apenas até a camada 9, sem alterar o encoder."""
        hidden, mask = self._start(input_ids, attention_mask, token_type_ids)
        result = {}
        for index, block in enumerate(self.base.bert.encoder.layer[:9], start=1):
            hidden = block(hidden, attention_mask=mask)[0]
            if index in EXITS[:-1]:
                result[index] = self._logits(hidden, index)
        return result

    @torch.inference_mode()
    def predict(self, input_ids, attention_mask, token_type_ids=None, strategy="fixed", depth=12, threshold=0.3, patience=2, tolerance=0.2):
        if strategy not in {"fixed", "confidence", "stability"}:
            raise ValueError(f"Estratégia desconhecida: {strategy}")
        if strategy == "confidence" and self.task != "rte":
            raise ValueError("Entropia de classes só se aplica a RTE")
        if depth not in EXITS or not 0 <= threshold <= 1 or patience < 2 or tolerance < 0:
            raise ValueError("Parâmetros de saída inválidos")
        batch_size = input_ids.shape[0]
        active = torch.arange(batch_size, device=input_ids.device)
        hidden, mask = self._start(input_ids, attention_mask, token_type_ids)
        predictions = torch.empty(batch_size, self.base.config.num_labels, device=input_ids.device)
        layers = torch.empty(batch_size, dtype=torch.int64, device=input_ids.device)
        previous = torch.zeros(batch_size, device=input_ids.device)
        streak = torch.ones(batch_size, dtype=torch.int64, device=input_ids.device)

        for number, block in enumerate(self.base.bert.encoder.layer, start=1):
            hidden = block(hidden, attention_mask=mask)[0]
            if number not in EXITS:
                continue
            if strategy == "fixed" and number < depth:
                continue
            logits = self._logits(hidden, number)
            if strategy == "fixed":
                should_exit = torch.ones(len(active), dtype=torch.bool, device=active.device)
            elif number == 12:
                should_exit = torch.ones(len(active), dtype=torch.bool, device=active.device)
            elif strategy == "confidence":
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(2.0, device=logits.device))
                should_exit = entropy <= threshold
            else:
                current = logits.argmax(-1) if self.task == "rte" else logits.squeeze(-1)
                stable = (current == previous[active]) if self.task == "rte" else (current - previous[active]).abs() <= tolerance
                if number == 3:
                    stable = torch.zeros_like(stable)
                streak[active] = torch.where(stable, streak[active] + 1, 1)
                previous[active] = current.float()
                should_exit = streak[active] >= patience

            leaving = active[should_exit]
            predictions[leaving] = logits[should_exit]
            layers[leaving] = number
            if number == 12 or strategy == "fixed":
                break
            keep = ~should_exit
            if not bool(keep.any()):
                break
            active = active[keep]
            hidden = hidden[keep]
            mask = mask[keep]
        return ExitResult(predictions, layers)
