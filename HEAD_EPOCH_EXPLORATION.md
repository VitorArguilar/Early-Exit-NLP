# Exploração: épocas de treinamento dos heads intermediários

Esta análise é **separada dos resultados padrão de Early Exit**. O BERTimbau ajustado por 7 épocas permaneceu congelado. Para cada tarefa, foi feita uma única trajetória de treinamento dos heads até 15 épocas, com snapshots em 5, 7, 10 e 15. As configurações foram as mesmas dos heads padrão: AdamW, taxa `1e-3`, batch size 16, seed 42 e comprimento máximo 128. Foram usados apenas os splits de treino (6.500 pares) e validação (500); **o teste não foi consultado**. O estado da época 5 reproduziu as métricas dos heads padrão na validação.

Comandos para reproduzir:

```powershell
.\.venv\Scripts\python.exe explore_head_epochs.py --task rte
.\.venv\Scripts\python.exe explore_head_epochs.py --task sts
```

## RTE — macro F1 na validação

| Head | 5 épocas | 7 épocas | 10 épocas | 15 épocas | Melhor F1 |
|---|---:|---:|---:|---:|---:|
| Camada 3 | 0,6172 | 0,5544 | **0,6833** | 0,6396 | 10 |
| Camada 6 | 0,8239 | 0,8306 | 0,8243 | **0,8660** | 15 |
| Camada 9 | 0,9340 | 0,9359 | **0,9380** | 0,9379 | 10 |

Na camada 9, a perda de validação cresceu de **0,1664** (época 5) para **0,2276** (época 15), enquanto a perda de treino caiu de **0,0896** para **0,0790**. O F1 subiu apenas 0,0040. Isso é um sinal de pior calibração/overfitting na camada 9, embora o F1 ainda não tenha caído. Na camada 6, o F1 de validação melhorou 0,0421 até a época 15, sem sinal equivalente de overfitting na perda. A camada 3 mostrou oscilação considerável; época 10 foi a melhor entre as quatro.

## STS — Pearson e MSE na validação

| Head | Pearson 5 | Pearson 7 | Pearson 10 | Pearson 15 | Menor MSE (época) |
|---|---:|---:|---:|---:|---:|
| Camada 3 | 0,3352 | 0,3577 | 0,3840 | **0,4077** | 0,8219 (15) |
| Camada 6 | 0,7944 | 0,8059 | 0,8087 | **0,8155** | 0,3390 (10) |
| Camada 9 | 0,9365 | 0,9373 | 0,9381 | **0,9391** | 0,1190 (10) |

O Pearson da camada 9 ganhou somente **0,0026** entre as épocas 5 e 15. Seu MSE caiu até a época 10 e depois subiu de **0,1190** para **0,1460**, apesar do leve aumento de Pearson. A camada 6 teve comportamento semelhante no MSE (0,3390 na época 10; 0,3571 na 15). Isso indica que treinar até 15 épocas melhorou pouco a ordenação relativa, mas prejudicou a precisão numérica em relação à época 10. A camada 3 ainda se beneficiou de mais treinamento, mas permaneceu bem abaixo das outras.

## Conclusão desta exploração

Mais épocas **não melhoraram todos os heads de modo uniforme**. Para RTE, os melhores pontos observados foram 10/15/10 épocas nas camadas 3/6/9. Para STS, a época 15 teve o maior Pearson nas três camadas, mas a época 10 teve o menor MSE nas camadas 6 e 9. Não há um único número de épocas que domine em todas as métricas e profundidades. Como esta análise usa uma só seed e uma validação de 500 pares, diferenças pequenas exigem cautela.

Nenhum checkpoint exploratório substituiu os heads padrão em `outputs/early_exit`. Antes de avaliar novos heads no teste, os limiares de Early Exit precisariam ser selecionados novamente na validação.

Artefatos: [RTE CSV](outputs/head_epoch_exploration/rte/comparison.csv), [STS CSV](outputs/head_epoch_exploration/sts/comparison.csv), [gráfico RTE](outputs/head_epoch_exploration/rte/quality_and_loss.png) e [gráfico STS](outputs/head_epoch_exploration/sts/quality_and_loss.png). Os snapshots `heads.pt` das quatro épocas ficam nos respectivos diretórios `epoch-5`, `epoch-7`, `epoch-10` e `epoch-15`.
