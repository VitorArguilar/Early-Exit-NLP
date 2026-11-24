# Treino até 100 épocas com early stopping

Os resultados e checkpoints anteriores foram movidos para [`backup/`](backup/). A nova rodada está em [`outputs/100epochs/`](outputs/100epochs/). O conjunto de teste foi avaliado após o treino; a seleção de épocas e configurações usa somente validação.

## Protocolo

- Tarefas separadas: RTE (macro F1) e STS (Pearson), seed 42, batch 16, comprimento máximo 128.
- Modelo base: BERTimbau Base iniciado dos pesos pré-treinados; até 100 épocas, taxa de aprendizado constante `2e-5`, avaliação a cada época. Early stopping após **5 épocas consecutivas sem superar a melhor métrica principal de validação**. O melhor checkpoint foi salvo em `float16` para reduzir o espaço em disco; o treino ocorreu em `float32`.
- Heads de saída: **um head linear em cada uma das 12 camadas** do encoder, treinados sobre o melhor checkpoint base da mesma tarefa com o encoder congelado. Cada head usa o vetor CLS da camada correspondente e uma camada linear, tanh e outra linear. Até 100 épocas, AdamW com taxa `1e-3` e batch 16. O head da camada 12 é adicional e separado da cabeça final do modelo base.
- Early stopping dos heads: **5 épocas consecutivas sem melhora da média da métrica principal entre as 12 camadas**. Há um único checkpoint conjunto por tarefa. O histórico registra métricas de treino e validação de cada camada em cada época. O pico individual de uma camada pode ocorrer fora da época do checkpoint conjunto; esses picos individuais não foram salvos como checkpoints separados.

| Tarefa | Componente | Épocas executadas | Melhor época salva | Melhor validação | Critério |
|---|---|---:|---:|---:|---|
| RTE | Base | 11 | 6 | 0,957998 | Macro F1 |
| STS | Base | 31 | 26 | 0,972144 | Pearson |
| RTE | 12 heads | 23 | 18 | 0,841914 | Média dos 12 macro F1 |
| STS | 12 heads | 84 | 79 | 0,796103 | Média dos 12 Pearson |

Os números dos heads são a **média das 12 camadas**, não a qualidade da camada 12 isolada nem um resultado de teste. Consulte [`heads_by_layer.csv`](outputs/100epochs/heads_by_layer.csv) para a qualidade de cada camada no checkpoint salvo e o pico individual observado no histórico.

## Artefatos

- [`summary.csv`](outputs/100epochs/summary.csv): resumo dos quatro treinos.
- `outputs/100epochs/base/{rte,sts}/history.csv`: perda de treino, perda de validação e métrica de validação em cada época.
- `outputs/100epochs/base/{rte,sts}/best_model/`: pesos, configuração e tokenizer do melhor checkpoint base.
- `outputs/100epochs/heads/{rte,sts}/history.csv`: perda e qualidade de treino/validação por época e por camada.
- `outputs/100epochs/heads/{rte,sts}/best_heads.pt`: checkpoint conjunto dos 12 heads.
- [`figures/`](outputs/100epochs/figures/): gráficos de histórico dos modelos base e mapas de calor dos heads.
- `outputs/100epochs/evaluation/{rte,sts}/`: métricas de validação e teste de saída fixa, confiança (RTE) e previsões consecutivas.
- [`apresentação/`](apresentação/): gráficos e tabelas construídos apenas com os checkpoints desta rodada.

## Resultados de teste

| Tarefa | Configuração | Qualidade | Aceleração contra a base |
|---|---|---:|---:|
| RTE | Base, época 6 | Macro F1 0,8959 | 1,00× |
| STS | Base, época 26 | Pearson 0,8383 | 1,00× |
| STS | Saída fixa, camada 9, heads da época 79 | Pearson 0,8385 | 1,31× |

Para selecionar uma saída antecipada, foi exigida na validação perda máxima de 0,01 em macro F1 (RTE) ou 0,02 em Pearson (STS), e aceleração de pelo menos 1,05×. Nenhuma configuração RTE satisfez os dois limites. A configuração STS da camada 9 foi escolhida na validação, antes de consultar o teste. Os demais resultados de teste de todos os candidatos são exploratórios. A latência considera somente inferência na GPU, sem tokenização ou transferência, medida uma vez por configuração; os valores de aceleração são aproximados.

Os scripts são [`train_100_base.py`](train_100_base.py), [`train_100_heads.py`](train_100_heads.py) e [`summarize_100epochs.py`](summarize_100epochs.py). Para repetir o treino base a partir do BERTimbau publicado, use PyTorch 2.6 ou superior com CUDA compatível; a versão atual de Transformers exige isso ao carregar os pesos pré-treinados em `.bin`. Os checkpoints salvos nesta rodada usam `safetensors` e carregam normalmente no ambiente do projeto.
