# Early Exit com heads treinados por 10 épocas: três seeds

Este experimento usa as seeds **42, 123 e 2026** para variar a ordem dos dados no treinamento dos heads intermediários. A inicialização dos heads é sempre copiada dos mesmos pesos do checkpoint final; por isso, a fonte efetiva de variação entre seeds é principalmente o embaralhamento dos lotes. O checkpoint do BERTimbau Base continua sendo o mesmo modelo de 7 épocas para cada tarefa; o encoder e o head final permanecem congelados. Portanto, os desvios abaixo medem variação dos **heads**, não do fine-tuning completo do BERTimbau. Os resultados anteriores de 5 épocas foram preservados em `outputs/early_exit`.

Reprodução:

```powershell
.\.venv\Scripts\python.exe run_early_exit_multiseed.py
```

O script treina os heads por **10 épocas** em cada seed e tarefa. Depois avalia todas as configurações na validação, escolhe parâmetros por run e avalia no teste as quatro profundidades fixas e as estratégias dinâmicas selecionadas. O critério de seleção permanece: perda máxima de 0,01 de macro F1 (RTE) ou 0,02 de Pearson (STS), e speedup mínimo de 1,05× na validação. Hardware: RTX 3060, batch size 16; latência mediana de três passagens por run, sem tokenização ou transferência à GPU.

## Resultados no teste (média ± desvio padrão amostral)

| Tarefa | Estratégia | Qualidade | Latência (ms/par) | Speedup | Camadas médias |
|---|---|---:|---:|---:|---:|
| RTE | Fixa 3 | F1 0,5974 ± 0,0327 | 0,419 ± 0,006 | 3,76× ± 0,03 | 3,00 |
| RTE | Fixa 6 | F1 0,7896 ± 0,0042 | 0,805 ± 0,021 | 1,95× ± 0,01 | 6,00 |
| RTE | Fixa 9 | F1 0,8688 ± 0,0115 | 1,189 ± 0,030 | 1,32× ± 0,004 | 9,00 |
| RTE | Completa 12 | F1 0,8885 ± 0,0000 | 1,572 ± 0,035 | 1,00× | 12,00 |
| RTE | Confiança selecionada | F1 0,8723 ± 0,0025 | 1,253 ± 0,052 | 1,26× ± 0,03 | 8,00 ± 0,36 |
| STS | Fixa 3 | Pearson 0,3862 ± 0,0064 | 0,408 ± 0,002 | 3,74× ± 0,01 | 3,00 |
| STS | Fixa 6 | Pearson 0,7541 ± 0,0006 | 0,781 ± 0,003 | 1,95× ± 0,01 | 6,00 |
| STS | Fixa 9 | Pearson 0,8248 ± 0,0024 | 1,156 ± 0,004 | 1,32× ± 0,005 | 9,00 |
| STS | Completa 12 | Pearson 0,8312 ± 0,0000 | 1,525 ± 0,003 | 1,00× | 12,00 |

Na RTE, a validação selecionou limiares de entropia **0,5 / 0,3 / 0,5** para as três seeds. Nenhuma configuração de estabilidade atingiu simultaneamente qualidade e speedup mínimos nas três runs. No teste, a confiança perdeu em média **0,0162** de F1 ante o modelo completo, excedendo a margem de 0,01 usada na validação; o ganho médio de velocidade foi **1,26×**. A profundidade fixa 9 tem F1 médio próximo (0,8688) e speedup semelhante (1,32×), mas sua variação entre seeds é maior.

Na STS, nenhuma configuração dinâmica de estabilidade foi selecionada. A profundidade fixa 9 manteve Pearson médio **0,8248**, contra **0,8312** do modelo completo, com speedup **1,32×**. O MSE médio da camada 9 foi **0,4897**, ante **0,5136** do modelo completo, mas o desvio entre seeds foi alto (**0,1097**). Assim, a melhora média de MSE não deve ser tratada como robusta com apenas três runs.

## Arquivos para gráficos e análise qualitativa

- `outputs/early_exit_10epochs_3seeds/summary.csv` e `summary.json`: média, desvio e número de runs por estratégia.
- `all_validation.csv` e `all_test.csv`: todas as métricas por seed, estratégia e configuração selecionada, incluindo percentuais de saída por camada.
- `selection_by_seed.csv`: limiar, paciência ou tolerância escolhidos na validação de cada run.
- `seed-<seed>/<tarefa>/evaluation/report.json`, `validation.csv` e `test.csv`: relatórios individuais completos.
- `seed-<seed>/<tarefa>/evaluation/all_test_predictions.csv`: predição de cada par para cada estratégia avaliada no teste, com textos, rótulo, erro, comprimento em tokens e camada de saída.
- `qualitative_examples_combined.csv`: **144 exemplos** amostrados deterministicamente por estratégia e camada de saída, priorizando baixo e alto erro. Os arquivos `qualitative_examples.csv` por run preservam a mesma amostra, e os CSVs completos permitem auditoria.
- [Gráfico RTE](outputs/early_exit_10epochs_3seeds/rte_quality_vs_latency.png) e [gráfico STS](outputs/early_exit_10epochs_3seeds/sts_quality_vs_latency.png): qualidade versus latência com barras de desvio padrão.

Os exemplos qualitativos foram selecionados **depois** da escolha de parâmetros, apenas para interpretação; não foram usados para ajustar as regras de saída. As comparações com o experimento antigo de 5 épocas são exploratórias, pois aquele experimento teve uma só seed. As latências também variam com o estado da GPU e do sistema.
