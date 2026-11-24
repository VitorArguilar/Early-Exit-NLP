# BERTimbau Base no ASSIN2

O [estudo exploratório de até 30 épocas](ESTUDO_30_EPOCAS.md) compara as curvas longas do modelo base e dos heads intermediários em RTE e STS.

A [comparação completa dos testes](final/COMPARACAO_COMPLETA.md) reúne tabelas e gráficos de todas as configurações avaliadas.

As seções de motivação, objetivo e trabalhos relacionados para o artigo estão em [ARTIGO_SECOES_INICIAIS.md](ARTIGO_SECOES_INICIAIS.md).

Os experimentos de early exit, com código e resultados, estão descritos em [EARLY_EXIT.md](EARLY_EXIT.md).
As tabelas e a análise dos experimentos executados estão em [RESULTS_EARLY_EXIT.md](RESULTS_EARLY_EXIT.md).
O estudo exploratório separado sobre 5, 7, 10 e 15 épocas dos heads está em [HEAD_EPOCH_EXPLORATION.md](HEAD_EPOCH_EXPLORATION.md).
O experimento com **10 épocas e três seeds**, com dados agregados e exemplos qualitativos, está em [MULTISEED_10_EPOCHS.md](MULTISEED_10_EPOCHS.md).
Os materiais para o slide de experimentos estão na pasta [final](final/ROTEIRO_SLIDE_EXPERIMENTOS.md).

Etapa inicial do trabalho: obter o corpus ASSIN2 e medir o desempenho de BERTimbau Base após fine-tuning nas duas tarefas. Cada linha do corpus representa **um par de sentenças**. A versão publicada contém 9.448 pares: 6.500 de treino, 500 de validação e 2.448 de teste.

## Preparação

Requer Python 3.10+ e espaço para baixar pesos do modelo. Em PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python prepare_data.py
.venv\Scripts\python train.py --task rte
.venv\Scripts\python train.py --task sts
```

O primeiro comando de treino baixa `neuralmind/bert-base-portuguese-cased`; `prepare_data.py` baixa `nilc-nlp/assin2`. Os arquivos do corpus ficam em `data/assin2/`, e os modelos e métricas em `outputs/rte/` e `outputs/sts/`. Use `--help` nos scripts para ajustar hiperparâmetros. GPU é recomendada; CPU funciona, mas pode ser lenta.

Depois de preparar o dataset, execute cada tarefa separadamente:

```powershell
.\.venv\Scripts\python.exe train.py --task rte
.\.venv\Scripts\python.exe train.py --task sts
```

O início do treino registra se CUDA está em uso, o nome da GPU e a quantidade de pares em cada split. O início e o fim de cada época também são registrados. O primeiro treino ainda precisa baixar os pesos do BERTimbau.

## Protocolo

### Comparação de épocas

```powershell
.\.venv\Scripts\python.exe compare_epochs.py --task rte
.\.venv\Scripts\python.exe compare_epochs.py --task sts
```

Cada comando treina uma trajetória até a época 9, salva os pesos nas épocas 3, 5, 7 e 9 e compara as métricas de validação, a perda de treino e a perda de validação. O melhor checkpoint é escolhido pela macro F1 (RTE) ou Pearson (STS), com desempate a favor da época anterior. Somente o escolhido é avaliado no teste. Resultados em `outputs/epoch_sweep/<task>/comparison.csv` e `comparison.json`; o modelo escolhido fica em `best_model`. O agendador de taxa de aprendizado é constante nesse experimento para manter uma trajetória comparável entre as quatro épocas. Os resultados da execução antiga de 3 épocas em `outputs/rte` usaram outro agendador e não devem ser misturados diretamente com essa comparação.

- RTE: classificação `NONE` (0) / `ENTAILMENT` (1), com macro F1 e acurácia.
- STS: regressão em [1, 5], com correlação de Pearson e RMSE; previsões são limitadas a [1, 5] somente ao calcular métricas.
- Seleção do melhor checkpoint pela métrica principal na validação; teste avaliado uma vez ao final. A semente e os hiperparâmetros são salvos com os resultados.
- Os dois fine-tunings usam o mesmo encoder inicial, mas cabeças e checkpoints independentes. Para comparação posterior com early exit, mantenha os mesmos splits e métricas.

## Enunciado

O PDF local exige tópico de PLN aprovado pelos professores, dataset anotado, Transformer, proposta, implementação, análise detalhada, artigo de 8 a 10 páginas no formato SBC e apresentação de exatamente 15 minutos. Exige também declarar no artigo o uso de IA generativa e revisar/validar todo conteúdo gerado. A análise de early exit e os experimentos correspondentes são a próxima etapa, ainda não implementada.

Fontes: [ASSIN2 oficial](https://sites.google.com/view/assin2/english), [dataset publicado](https://huggingface.co/datasets/nilc-nlp/assin2), [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
