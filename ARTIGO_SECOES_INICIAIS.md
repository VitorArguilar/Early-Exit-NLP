# Fundamentação e proposta do trabalho

Texto preparado para adaptação ao artigo no formato SBC. Os resultados numéricos devem ser apresentados nas seções de metodologia e experimentos, a partir dos arquivos produzidos pelo projeto.

## Problema e motivação

Modelos Transformer pré-treinados alcançam bons resultados em tarefas de compreensão de texto, mas a inferência convencional executa todas as camadas para cada exemplo, mesmo quando uma decisão já poderia ser tomada antes. Esse custo importa quando há restrições de latência ou de recursos computacionais. O BERTimbau fornece representações pré-treinadas para português brasileiro [1], e o ASSIN 2 reúne pares de sentenças anotados para inferência textual (RTE) e similaridade semântica (STS) [2]. Juntos, permitem investigar se a profundidade necessária varia entre exemplos e entre uma tarefa de classificação e outra de regressão.

Métodos de *early exit* inserem preditores intermediários no encoder e podem encerrar o processamento após uma camada suficiente para a decisão. DeeBERT [3] e FastBERT [4] usam confiança para orientar essa escolha; PABEE [5] considera a estabilidade das predições ao longo das camadas. A aplicação dessas ideias ao BERTimbau no ASSIN 2 requer medir a economia **real** de inferência e verificar quanto desempenho se perde em cada tarefa. Uma previsão intermediária obtida após executar todas as camadas não representa economia computacional.

## Objetivo e questão de pesquisa

O objetivo é caracterizar o compromisso entre qualidade preditiva e custo de inferência do BERTimbau Base ajustado no ASSIN 2. A questão central é: **quanto é possível reduzir a profundidade e a latência de inferência em RTE e STS sem degradar excessivamente suas métricas de qualidade?**

Para respondê-la, o trabalho compara (i) saídas fixas nas camadas 3, 6, 9 e 12, (ii) saída dinâmica por limiar de entropia e (iii) saída dinâmica por estabilidade das previsões. Em STS, estabilidade é definida por uma tolerância entre valores consecutivos. O encoder do modelo já ajustado é preservado; classificadores ou regressores intermediários são treinados sobre seus estados. Os parâmetros de saída são escolhidos na validação, e o teste fica reservado à avaliação final. As comparações usam o mesmo ambiente de inferência e incluem F1 e acurácia em RTE, Pearson e erro quadrático médio em STS, latência, *speedup*, camadas executadas e distribuição das saídas. Exemplos de predição permitem examinar quais pares exigem maior profundidade.

## Trabalhos relacionados

**BERTimbau.** Souza, Nogueira e Lotufo [1] apresentaram modelos BERT pré-treinados em português brasileiro e sua aplicação a tarefas de PLN. Esse trabalho fornece o encoder de 12 camadas utilizado aqui. Nossa investigação parte de checkpoints já ajustados ao ASSIN 2 e altera a política de inferência, preservando os pesos do encoder.

**ASSIN 2.** Real, Fonseca e Oliveira [2] organizaram uma avaliação conjunta de similaridade semântica e inferência textual em português, com rótulos para ambas as tarefas nos pares de sentenças. Esse conjunto permite estudar a mesma estratégia de saída antecipada em classificação e regressão, mantendo divisões de treino, validação e teste.

**DeeBERT.** Xin e colaboradores [3] adicionaram saídas intermediárias ao BERT e propuseram interromper a inferência quando a previsão atingisse confiança suficiente. A ideia fundamenta nossa política por entropia. Nosso protocolo examina pontos de saída específicos e treina os heads sobre o encoder já ajustado, o que evita modificar o baseline durante essa etapa.

**FastBERT.** Liu e colaboradores [4] combinaram classificadores intermediários, inferência adaptativa e autodestilação. O artigo reforça a necessidade de estudar diferentes limiares de saída. Neste projeto, a destilação não é empregada: isso permite analisar separadamente o efeito dos heads intermediários e da regra de decisão.

**PABEE.** Zhou e colaboradores [5] propuseram sair após uma sequência de predições estáveis, controlada por um parâmetro de paciência. Essa é a referência para nossa política de estabilidade. Como STS produz valores contínuos, a adaptação usa uma tolerância entre previsões; a estabilidade é observada nos pontos de saída selecionados, não em todas as camadas.

Em conjunto, esses estudos sustentam as duas tarefas, o modelo e as políticas dinâmicas. A contribuição experimental deste trabalho é comparar, no mesmo protocolo em português, profundidades fixas, confiança e estabilidade, medindo tanto qualidade quanto tempo efetivamente poupado pela interrupção do encoder.

## Referências

1. SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. **BERTimbau: Pretrained BERT Models for Brazilian Portuguese.** In: *Intelligent Systems (BRACIS 2020)*. Springer, 2020, p. 403–417. DOI: [10.1007/978-3-030-61377-8_28](https://doi.org/10.1007/978-3-030-61377-8_28).
2. REAL, L.; FONSECA, E.; OLIVEIRA, H. G. **Organizing the ASSIN 2 Shared Task.** In: *Proceedings of the ASSIN 2 Shared Task*. CEUR Workshop Proceedings, v. 2583, 2020, p. 1–13. [Texto integral](https://ceur-ws.org/Vol-2583/1_ASSIN-2.pdf).
3. XIN, J. et al. **DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.** In: *Proceedings of ACL*. 2020, p. 2246–2251. DOI: [10.18653/v1/2020.acl-main.204](https://aclanthology.org/2020.acl-main.204/).
4. LIU, W. et al. **FastBERT: a Self-distilling BERT with Adaptive Inference Time.** In: *Proceedings of ACL*. 2020, p. 6035–6044. DOI: [10.18653/v1/2020.acl-main.537](https://aclanthology.org/2020.acl-main.537/).
5. ZHOU, W. et al. **BERT Loses Patience: Fast and Robust Inference with Early Exit.** In: *Advances in Neural Information Processing Systems 33 (NeurIPS)*. 2020. [Artigo](https://proceedings.neurips.cc/paper/2020/hash/d4dd111a4fd973394238aca5c05bebe3-Abstract.html).

> **Nota para submissão:** o enunciado exige declarar o uso de IA generativa e revisar e validar o conteúdo gerado antes de incluí-lo no artigo.
