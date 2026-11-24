# Resultados dos novos treinos

Todos os dados vêm exclusivamente de `outputs/100epochs`. Os históricos mostram treino e validação dos checkpoints novos. A época dos heads é selecionada pela média da métrica de validação nas 12 camadas; o pico individual de cada camada é apenas informativo e não representa outro checkpoint.

As estrelas nos gráficos indicam a configuração selecionada usando apenas validação. A tabela total inclui todos os candidatos avaliados no teste para análise exploratória. A aceleração é a latência da base dividida pela latência da configuração, medidas nesta execução. Confiança só se aplica a RTE, pois STS é regressão.
