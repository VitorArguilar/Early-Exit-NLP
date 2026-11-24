# Comparação completa dos testes

## Arquivos para análise e apresentação

- [Tabela de todos os registros](tabela_todos_os_testes.csv): 301 linhas, com métricas disponíveis, parâmetros, seed, divisão dos dados e arquivo de origem.
- [Tabela comparativa das três seeds](tabela_comparativa_tres_seeds.csv): nove grupos de estratégia/tarefa, com médias e desvios das métricas, latência, speedup, camadas e percentuais de saída.
- [Early Exit, 10 épocas e três seeds](grafico_todos_early_exit.png): todas as 81 avaliações de validação e 27 de teste. As cores indicam a estratégia; os parâmetros de cada ponto estão na tabela completa.
- Versões para slides, com um gráfico por tarefa e divisão: [RTE validação](grafico_early_exit_rte_validation_slide.png), [RTE teste](grafico_early_exit_rte_test_slide.png), [STS validação](grafico_early_exit_sts_validation_slide.png) e [STS teste](grafico_early_exit_sts_test_slide.png).
- [Early Exit original, cinco épocas](grafico_todos_early_exit_original.png): todas as 27 avaliações de validação e dez avaliações de teste.
- [Épocas do encoder](grafico_todas_epocas_encoder.png): 3, 5, 7 e 9 épocas, para RTE e STS.
- [Épocas dos heads](grafico_todas_epocas_heads.png): 5, 7, 10 e 15 épocas, nas camadas 3, 6 e 9, para RTE e STS.
- [Estudo até 30 épocas em RTE](grafico_30_epocas_rte.png) e [STS](grafico_30_epocas_sts.png): curvas completas do modelo base e dos heads, com métricas a cada duas épocas. A [tabela específica](tabela_30_epocas.csv) contém 120 medições.

As figuras também estão disponíveis em SVG. Execute `\.venv\Scripts\python.exe final\gerar_comparacao_completa.py` para recriar os arquivos a partir dos resultados salvos.

## Como ler a tabela

Cada linha representa uma medição, e `familia` identifica o protocolo experimental. Células vazias significam que a métrica ou o parâmetro não se aplica ou não foi registrado naquela execução; não representam zero. O campo `arquivo_origem` permite conferir cada valor no dado bruto.

| Família | Validação | Teste | Observação |
|---|---:|---:|---|
| Baseline inicial | 1 | 1 | Execução inicial de RTE, com três épocas e protocolo próprio. |
| Comparação de épocas do encoder | 8 | 2 | Teste apenas para a época selecionada em cada tarefa. |
| Exploração de épocas dos heads | 24 | 0 | Experimento separado, apenas validação. |
| Early Exit original | 27 | 10 | Heads com cinco épocas; teste apenas nas configurações escolhidas. |
| Early Exit, três seeds | 81 | 27 | Heads com dez épocas; seeds 42, 123 e 2026. |
| Modelo base, até 30 épocas | 30 | 0 | Uma trajetória por tarefa, medidas de 2 em 2 épocas. |
| Heads, até 30 épocas | 90 | 0 | Camadas 3, 6 e 9, duas tarefas, medidas de 2 em 2 épocas. |

**Total: 301 registros.** O conjunto de teste não foi usado para escolher limiares ou paciência. Os gráficos de épocas dos heads e do encoder devem ser interpretados separadamente dos gráficos de latência de Early Exit, pois correspondem a protocolos distintos. A latência também varia com o ambiente de medição, por isso comparações diretas de tempo devem priorizar estratégias medidas na mesma execução.
