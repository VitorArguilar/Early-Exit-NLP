"""Checagens de fidelidade e de interrupção real das camadas."""

import torch
from transformers import AutoTokenizer

from early_exit import EarlyExitBert


def main():
    path = "outputs/epoch_sweep/rte/epoch-7"
    tokenizer = AutoTokenizer.from_pretrained("outputs/epoch_sweep/rte/best_model")
    model = EarlyExitBert(path).eval()
    batch = tokenizer(["O gato dorme.", "Uma pessoa corre."], ["Um animal dorme.", "Ninguém corre."], return_tensors="pt", padding=True)
    with torch.inference_mode():
        original = model.base(**batch).logits
        complete = model.predict(**batch, strategy="fixed", depth=12)
    assert torch.allclose(original, complete.predictions, atol=1e-5), "A saída da camada 12 difere do checkpoint original"
    assert complete.layers.tolist() == [12, 12]

    counts = {index: 0 for index in range(4, 13)}
    hooks = [model.base.bert.encoder.layer[index - 1].register_forward_hook(lambda _, __, ___, index=index: counts.__setitem__(index, counts[index] + 1)) for index in counts]
    try:
        shallow = model.predict(**batch, strategy="fixed", depth=3)
        assert shallow.layers.tolist() == [3, 3]
        assert all(value == 0 for value in counts.values()), f"Camadas profundas executadas: {counts}"
    finally:
        for hook in hooks:
            hook.remove()

    counts = {index: 0 for index in range(4, 13)}
    hooks = [model.base.bert.encoder.layer[index - 1].register_forward_hook(lambda _, __, ___, index=index: counts.__setitem__(index, counts[index] + 1)) for index in counts]
    try:
        confident = model.predict(**batch, strategy="confidence", threshold=1.0)
        assert confident.layers.tolist() == [3, 3]
        assert all(value == 0 for value in counts.values()), f"Camadas profundas executadas: {counts}"
    finally:
        for hook in hooks:
            hook.remove()
    print("OK: camada 12 idêntica ao baseline; saídas fixa e dinâmica na camada 3 interrompem as camadas 4-12")


if __name__ == "__main__":
    main()
