# Resultados de Early Exit

Experimentos reais no ASSIN2, usando os checkpoints de 7 épocas de RTE e STS. Os heads intermediários foram treinados por 5 épocas com o encoder congelado. Hardware: NVIDIA GeForce RTX 3060, batch size 16. Latência: mediana de três passagens por todos os lotes, sem tokenização nem transferência para GPU. A seleção de parâmetros usou somente os 500 pares de validação; os números abaixo são dos 2.448 pares de teste.

## RTE

| Estratégia | Macro F1 | Acurácia | ms/par | Speedup | Camadas médias |
|---|---:|---:|---:|---:|---:|
| Fixa 3 | 0,5600 | 0,5870 | 0,404 | 3,74× | 3,00 |
| Fixa 6 | 0,7527 | 0,7541 | 0,773 | 1,96× | 6,00 |
| Fixa 9 | 0,8684 | 0,8685 | 1,143 | 1,32× | 9,00 |
| Completa 12 | 0,8885 | 0,8889 | 1,514 | 1,00× | 12,00 |
| Confiança, entropia ≤ 0,6 | 0,8711 | 0,8713 | 1,213 | 1,25× | 8,19 |
| Estabilidade, paciência 3 | 0,8819 | 0,8824 | 1,449 | 1,04× | 10,51 |

Na validação, o limiar 0,6 teve F1 0,9480 contra 0,9560 do modelo completo, dentro da margem de 0,01 estabelecida antes do teste. No teste, a perda foi 0,0174, acima dessa margem. Portanto, confiança é a opção de maior speedup medido, mas sua qualidade não sustentou o limite escolhido. Estabilidade preservou mais F1 no teste (perda de 0,0066), com ganho pequeno de latência. A saída fixa na camada 9 perdeu 0,0201 de F1.

Com confiança 0,6, 39,0% dos pares saíram na camada 6, 48,9% na 9 e 12,2% na 12. Os pares que chegaram à camada 12 tiveram acurácia 0,711, contra 0,889 na camada 6 e 0,897 na 9: a regra reteve exemplos mais difíceis. O comprimento médio foi aproximadamente 27,5 tokens na camada 6, 25,3 na 9 e 26,2 na 12; esses dados não sustentam que pares longos exijam mais profundidade.

## STS

| Estratégia | Pearson | MSE | ms/par | Speedup | Camadas médias |
|---|---:|---:|---:|---:|---:|
| Fixa 3 | 0,3466 | 1,0427 | 0,404 | 3,72× | 3,00 |
| Fixa 6 | 0,7459 | 0,4674 | 0,766 | 1,96× | 6,00 |
| Fixa 9 | 0,8236 | 0,5641 | 1,135 | 1,32× | 9,00 |
| Completa 12 | 0,8312 | 0,5136 | 1,503 | 1,00× | 12,00 |

Na validação, a camada 9 teve Pearson 0,9365 contra 0,9529 da camada 12, dentro da margem de 0,02, e speedup de 1,33×. No teste, a perda de Pearson foi 0,0076, mas o MSE aumentou de 0,5136 para 0,5641. A estabilidade com paciência 2 reduziu qualidade além da margem; com paciência 3, o ganho de latência não alcançou 1,05× na validação. Por isso, nenhuma configuração dinâmica de STS foi escolhida para o teste. Confiança por entropia não se aplica à regressão sem uma distribuição de classes.

## Interpretação e limites

Para STS, a camada fixa 9 oferece o melhor compromisso observado entre Pearson e tempo, considerando também a piora no MSE. Para RTE, o método de confiança acelera a inferência, mas o limite de qualidade definido na validação não se confirmou no teste; estabilidade conserva mais F1 com ganho modesto. A redução no número médio de camadas é maior que o speedup em alguns casos porque decidir e reorganizar os exemplos ativos também custa tempo. Os resultados usam uma única semente e uma GPU; não incluem variação entre repetições independentes de treinamento.

Dados completos e curvas: [RTE](outputs/early_exit/rte/evaluation/report.json), [STS](outputs/early_exit/sts/evaluation/report.json), [qualidade × latência RTE](outputs/early_exit/rte/evaluation/test_quality_vs_latency.png) e [qualidade × latência STS](outputs/early_exit/sts/evaluation/test_quality_vs_latency.png).
