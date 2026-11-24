# Estudo exploratório: até 30 épocas

Foram executadas quatro trajetórias com **seed 42**, medindo treino e validação nas épocas pares de 2 a 30. O conjunto de teste não foi usado. Os resultados permanecem separados dos experimentos padrão de Early Exit.

## Protocolo

- **Modelo base:** fine-tuning de BERTimbau Base a partir dos pesos pré-treinados, para RTE e STS separadamente; taxa de aprendizado constante de `2e-5`, batch 16 e comprimento máximo 128. É a continuação do protocolo de comparação de épocas anterior. A validação foi calculada a cada época e registrada para comparação a cada duas.
- **Heads intermediários:** camadas 3, 6 e 9 do checkpoint base original de **7 épocas**; encoder congelado, AdamW com taxa de `1e-3`, batch 16. Os vetores CLS foram extraídos uma vez por split e reutilizados nas 30 épocas, pois o encoder está congelado e em modo de avaliação. As métricas foram medidas a cada duas épocas. Os heads deste estudo não usam os novos checkpoints de 24 ou 28 épocas do modelo base.
- **Seleção:** o melhor ponto abaixo refere-se somente à validação. Esta exploração não altera a seleção de hiperparâmetros nem os resultados de teste anteriormente publicados.

## Modelo base

| Tarefa | Métrica | Época 7 do estudo anterior | Melhor época nesta trajetória | Época 30 | Leitura |
|---|---|---:|---:|---:|---|
| RTE | Macro F1 | 0,9560 | **0,9660** (28) | 0,9600 | A queda após 7 épocas não foi persistente; a perda de validação, porém, ficou acima da perda inicial. |
| STS | Pearson | 0,9529 | **0,9721** (24) | 0,9700 | A qualidade voltou a crescer após a queda próxima de 9 épocas; o pico ocorreu bem depois de 7. |

Em RTE, as épocas 24 e 28 tiveram macro F1 praticamente igual (0,965993 e 0,965999). A perda de treino caiu de **0,2139** (época 2) para **0,0177** (época 30), enquanto a perda de validação passou de **0,2064** para **0,2871**. Isso sugere pior calibração ou maior confiança em erros, embora o macro F1 tenha melhorado em alguns pontos tardios. Em STS, o menor valor de perda de validação entre os pontos medidos ocorreu na época 24 (**0,0618**); na época 30 subiu para **0,0939**, ainda abaixo do valor da época 7 (**0,1015**). Portanto, a hipótese de overfitting simples e monotônico após sete épocas não é sustentada por esta trajetória.

## Heads intermediários

| Tarefa | Camada | Métrica na época 10 | Melhor época até 30 | Métrica na época 30 | Leitura |
|---|---:|---:|---:|---:|---|
| RTE | 3 | F1 0,6833 | **0,6999** (28) | 0,6937 | Melhorias tardias com oscilações. |
| RTE | 6 | F1 0,8243 | **0,8740** (14) | 0,8136 | Pico intermediário, seguido de queda; tendência compatível com overfitting ou instabilidade de otimização. |
| RTE | 9 | F1 0,9380 | **0,9420** (12) | 0,9419 | F1 praticamente estável; perda de validação sobe de 0,1784 na época 10 para 0,2219 na 30. |
| STS | 3 | Pearson 0,3840 | **0,4466** (30) | 0,4466 | Continua melhorando até o fim da janela. |
| STS | 6 | Pearson 0,8087 | **0,8297** (30) | 0,8297 | Continua melhorando até o fim da janela. |
| STS | 9 | Pearson 0,9381 | **0,9406** (24) | 0,9393 | Platô de qualidade, com oscilação pequena e aumento da perda no final. |

## Interpretação e limites

As quedas iniciais de qualidade do modelo base eram **vales locais da curva de validação**, especialmente em STS. A evidência de overfitting depende da métrica: a perda pode piorar enquanto F1 ou Pearson se mantêm ou melhoram. Nos heads, não há uma época ideal comum às três camadas; 10 épocas é um padrão comparável para os experimentos anteriores, mas esta rodada mostra que camadas rasas de STS ainda ganham qualidade depois disso.

Essas conclusões se limitam a **uma seed** e a **500 pares de validação**. Selecionar o máximo entre 15 pontos pode favorecer uma flutuação do conjunto de validação. Seriam necessárias novas seeds e uma avaliação final controlada para concluir que checkpoints tardios generalizam melhor no teste. Este estudo não mediu latência nem políticas dinâmicas de Early Exit com os novos checkpoints.

## Artefatos

- [Tabela das 120 medições](final/tabela_30_epocas.csv), [gráfico RTE](final/grafico_30_epocas_rte.svg) e [gráfico STS](final/grafico_30_epocas_sts.svg).
- Resultados brutos: [base RTE](outputs/long_overfit/base/rte/comparison.csv), [base STS](outputs/long_overfit/base/sts/comparison.csv), [heads RTE](outputs/long_overfit/heads/rte/comparison.csv) e [heads STS](outputs/long_overfit/heads/sts/comparison.csv).
- Scripts: [long_base_epochs.py](long_base_epochs.py), [long_head_epochs.py](long_head_epochs.py) e [gerar_estudo_30_epocas.py](final/gerar_estudo_30_epocas.py).

Para reproduzir em PowerShell com a GPU disponível, execute cada tarefa na sequência:

```powershell
.\.venv\Scripts\python.exe long_base_epochs.py --task rte
.\.venv\Scripts\python.exe long_base_epochs.py --task sts
.\.venv\Scripts\python.exe long_head_epochs.py --task rte
.\.venv\Scripts\python.exe long_head_epochs.py --task sts
.\.venv\Scripts\python.exe final\gerar_estudo_30_epocas.py
```
