# Early Exit no BERTimbau Base / ASSIN2

## Checkpoints e preparação

Os experimentos reutilizam `outputs/epoch_sweep/{rte,sts}/epoch-7`, selecionados pela validação no experimento de épocas. O BERT e a cabeça da camada 12 são carregados sem alterações. Os arquivos `outputs/epoch_sweep/{rte,sts}/comparison.json` registram a seleção da época 7. Não é necessário repetir o fine-tuning do encoder.

```powershell
.\.venv\Scripts\python.exe test_early_exit.py
.\.venv\Scripts\python.exe train_early_exit_heads.py --task rte --epochs 5
.\.venv\Scripts\python.exe train_early_exit_heads.py --task sts --epochs 5
.\.venv\Scripts\python.exe evaluate_early_exit.py --task rte
.\.venv\Scripts\python.exe evaluate_early_exit.py --task sts
```

## Decisões técnicas

- Camadas de saída: 3, 6, 9 e 12. Cada head intermediário tem pooling treinável sobre `[CLS]` e classificador linear; seus pesos são inicializados a partir do pooler e do classificador final do checkpoint. Só esses heads são atualizados; o BERT e a cabeça final permanecem congelados.
- Saída fixa executa exatamente até a camada escolhida. Confiança, disponível apenas para RTE, usa entropia binária normalizada por `log(2)`; sai quando a entropia é menor ou igual ao limiar. Para STS não há probabilidades de classes, portanto essa estratégia não se aplica.
- Estabilidade compara decisões consecutivas nas camadas selecionadas. RTE exige a mesma classe; STS exige diferença absoluta dentro da tolerância. Paciência 2 permite saída a partir da camada 6; paciência 3, a partir da camada 9.
- O código mantém apenas os exemplos ainda ativos no lote após cada saída; os blocos posteriores não são executados para eles. `test_early_exit.py` confirma que a saída completa reproduz os logits originais e que a saída na camada 3 não chama as camadas 4–12.
- Os heads usam apenas o split de treino. Limiar, paciência e tolerância são selecionados na validação. O teste é usado apenas na avaliação final dos candidatos escolhidos, além dos quatro baselines fixos.
- Critério de seleção: menor latência mediana entre candidatos que perdem no máximo 0,01 de macro F1 em RTE ou 0,02 de Pearson em STS e alcançam speedup de pelo menos 1,05×, comparados com a camada 12 na validação. O relatório também mostra todos os candidatos de validação para permitir outras escolhas.
- A latência é a mediana de três passagens por todos os lotes, com batch size 16, na mesma GPU. Tokenização e transferência para GPU ficam fora do tempo cronometrado. O speedup é a latência do modelo completo dividida pela latência da estratégia, ambas medidas na mesma execução.

## Saídas

Os heads ficam em `outputs/early_exit/<tarefa>/heads.pt`, com parâmetros em `heads_config.json`. Os resultados ficam em `outputs/early_exit/<tarefa>/evaluation/`: `validation.csv`, `test.csv`, `report.json`, gráficos PNG de qualidade versus camadas médias e versus latência e, quando uma estratégia dinâmica é selecionada, CSV por par de teste com a camada de saída. `report.json` também resume comprimento médio e qualidade por camada de saída. Os arquivos de resultados são produtos de execuções reais; não há valores simulados.

## Limitações de interpretação

O experimento atualizado com 10 épocas dos heads e três seeds é documentado separadamente em [MULTISEED_10_EPOCHS.md](MULTISEED_10_EPOCHS.md). Os comandos acima com `--epochs 5` reproduzem o experimento original; o padrão atual de `train_early_exit_heads.py` é 10 épocas.

O melhor limiar na validação pode perder mais qualidade no teste. Uma queda na média de camadas não garante speedup igual, porque a seleção dinâmica acrescenta operações e pode deixar lotes pequenos na GPU. Resultados de uma GPU e de uma semente não quantificam a variância entre execuções. O conjunto de teste deve permanecer fora de novas escolhas de limiar ou arquitetura.
