# Seleção de épocas no estudo de 30 épocas

Critério: maior **macro F1** de validação em RTE e maior **Pearson** de validação em STS, entre as épocas pares de 2 a 30. Em empate exato, selecionar a menor época. A perda de validação é informada para análise, mas não é o critério de escolha.

| Tarefa | Componente | Camada | Época escolhida | Qualidade na validação | Perda na validação |
|---|---|---:|---:|---:|---:|
| RTE | Base | 12 | 28 | Macro F1 0,9660 | 0,2936 |
| RTE | Head | 3 | 28 | Macro F1 0,6999 | 0,5681 |
| RTE | Head | 6 | 14 | Macro F1 0,8740 | 0,3130 |
| RTE | Head | 9 | 12 | Macro F1 0,9420 | 0,1839 |
| STS | Base | 12 | 24 | Pearson 0,9721 | 0,0618 |
| STS | Head | 3 | 30 | Pearson 0,4466 | 0,7867 |
| STS | Head | 6 | 30 | Pearson 0,8297 | 0,3365 |
| STS | Head | 9 | 24 | Pearson 0,9406 | 0,1195 |

Os heads foram treinados sobre o **encoder de 7 épocas**, congelado. Portanto, os heads escolhidos acima não formam um sistema já avaliado com os modelos base de 28 ou 24 épocas. Os pesos Git LFS dos heads e dos checkpoints base antigos de 7 épocas foram baixados para este checkout. Para usar épocas diferentes por camada, é necessário carregar os pesos de cada época separadamente.

## Conferência das tabelas

A seleção foi recalculada dos CSVs brutos de `outputs/long_overfit`. O script [selecionar_epocas_30.py](selecionar_epocas_30.py) confirmou os oito valores selecionados em [tabela_30_epocas.csv](tabela_30_epocas.csv), [tabela_todos_os_testes.csv](tabela_todos_os_testes.csv) e [resumo_30_epocas.json](resumo_30_epocas.json). A [tabela_epocas_selecionadas_30.csv](tabela_epocas_selecionadas_30.csv) contém as linhas selecionadas com precisão completa e arquivo de origem.

As tabelas `tabela_principal.csv` e `tabela_comparativa_tres_seeds.csv`, bem como os gráficos de tradeoff e os resultados de teste já apresentados, pertencem ao protocolo anterior: encoder de 7 épocas e heads de 10 épocas, com três seeds. Seus valores não representam os checkpoints desta seleção. Foram preservados como resultados históricos desse protocolo.

**O teste já foi executado para a combinação disponível:** encoder de 7 épocas com os heads selecionados acima. A [tabela_teste_heads_selecionados_30.csv](tabela_teste_heads_selecionados_30.csv) contém qualidade, latência e speedup de cada estratégia. Os relatórios completos estão em `outputs/long_overfit/selected_heads/{rte,sts}/evaluation/report.json`. Os limiares das estratégias dinâmicas foram escolhidos na validação; o teste foi avaliado depois. A latência usou uma repetição, por isso deve ser tratada como estimativa preliminar.

O estudo de 30 épocas não salvou os checkpoints do modelo base nas épocas 28 e 24 em `outputs/long_overfit/base`. Os pesos LFS disponíveis cobrem os modelos base de 7 épocas e os heads de 2 a 30 épocas; o ASSIN2 já foi baixado. Assim, **ainda não há resultados de teste para bases de 28 e 24 épocas**, e as tabelas históricas não devem ser substituídas por essa combinação distinta. Uma avaliação final com as bases selecionadas exigiria reproduzir seus checkpoints, montar as combinações correspondentes, rodar o teste e medir novamente a latência.

A seleção é exploratória: uma seed, 500 pares de validação e máximo entre 15 medições por curva.
