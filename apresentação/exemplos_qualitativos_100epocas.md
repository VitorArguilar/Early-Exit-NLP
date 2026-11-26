# Exemplos qualitativos · checkpoints de 100 épocas

Pares do teste ASSIN2, recuperados da cópia textual da rodada anterior. Previsões e camadas recalculadas com os checkpoints atuais, apenas por inferência.

RTE: 0 = sem implicação; 1 = implicação. Em STS, ‘acerto’ significa erro absoluto de até 0,5 ponto. Qualidade e aceleração são métricas agregadas do teste completo, não de cada par.

## RTE

| Configuração | Resultado | Saída | Premissa | Hipótese | Esperado | Previsto | Erro | Qualidade teste | Aceleração |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Confiança · limiar 0,5 | Acerto | Cedo (camada 1) | Um cara está pensando | O cara está pensando | 1 | 1 | 0 | 0.853 | 1.63× |
| Confiança · limiar 0,5 | Erro | Meio (camada 7) | Um homem está jantando | Um homem está comendo | 1 | 0 | 1 | 0.853 | 1.63× |
| Consecutivas · paciência 2 | Acerto | Meio (camada 5) | O homem está comendo maçã | A pessoa está comendo maçã | 1 | 1 | 0 | 0.689 | 3.68× |
| Consecutivas · paciência 2 | Erro | Cedo (camada 2) | O cara está jejuando | O cara não está comendo | 1 | 0 | 1 | 0.689 | 3.68× |
| Consecutivas · paciência 3 | Acerto | Tarde (camada 11) | Um homem está cantando na praça | Um homem está cantando | 1 | 1 | 0 | 0.736 | 2.15× |
| Consecutivas · paciência 3 | Erro | Cedo (camada 3) | Um homem está cuspindo | Um homem está falando | 0 | 1 | 1 | 0.736 | 2.15× |

## STS

| Configuração | Resultado | Saída | Premissa | Hipótese | Esperado | Previsto | Erro | Qualidade teste | Aceleração |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Consecutivas · paciência 2 · tolerância 0,2 | Acerto | Cedo (camada 4) | Um cara está almoçando | Um cara está comendo | 4.25 | 4.60 | 0.35 | 0.549 | 2.30× |
| Consecutivas · paciência 2 · tolerância 0,2 | Erro | Meio (camada 6) | Algum peixe está nadando | Um animal está na água | 4.00 | 3.31 | 0.69 | 0.549 | 2.30× |
| Consecutivas · paciência 3 · tolerância 0,2 | Acerto | Meio (camada 8) | Um cara está pensando | O cara está pensando | 5.00 | 4.85 | 0.15 | 0.731 | 1.31× |
| Consecutivas · paciência 3 · tolerância 0,2 | Erro | Tarde (camada 10) | O cara está jejuando | O cara não está comendo | 4.00 | 3.23 | 0.77 | 0.731 | 1.31× |
