# Sugestão de fluxo — um slide de experimentos

**Título sugerido:** *Early Exit: quanto desempenho custa ganhar velocidade?*

**Posição na apresentação:** depois de explicar o BERTimbau, os heads nas camadas 3/6/9 e as regras de saída; antes da análise qualitativa e das limitações. Reserve cerca de **75–90 segundos** deste slide dentro da apresentação de 15 minutos.

## Composição do slide

1. **Topo (uma linha):** “ASSIN2 · 3 seeds · heads treinados por 10 épocas · RTX 3060”. Em letra menor: “2.448 pares no teste; limiares escolhidos na validação”.
2. **Centro (maior área):** inserir `grafico_tradeoff.png` ou `grafico_tradeoff.svg`. O painel esquerdo mostra RTE; o direito, STS. Eixo horizontal: speedup em relação ao modelo completo; eixo vertical: qualidade. Barras indicam desvio padrão entre seeds dos heads.
3. **Rodapé (duas mensagens curtas):** “RTE: confiança ≈1,26×, mas F1 cai 0,0162 no teste” e “STS: camada fixa 9 ≈1,32×, Pearson cai 0,0064”. A linha tracejada é uma **referência visual**: qualidade do modelo completo no teste menos a tolerância definida antes do teste (0,01 para F1; 0,02 para Pearson). A seleção real foi feita na validação.

## Fala sugerida

> “Comparamos o modelo completo com saídas antecipadas nas mesmas condições de hardware. Cada ponto resume três treinamentos dos heads com seeds diferentes. Em inferência textual, a regra por confiança acelera cerca de 1,26 vez, mas a perda média de F1 no teste foi 0,016, maior que a margem de 0,01 usada na validação. Em similaridade, a saída fixa na camada 9 é mais consistente: ganha cerca de 1,32 vez e reduz Pearson de 0,831 para 0,825. A regra por estabilidade não atingiu simultaneamente qualidade e ganho mínimo de tempo nessas runs. Assim, a profundidade fixa 9 é a opção mais estável para STS; em RTE, a escolha depende do quanto de F1 podemos perder.”

**Transição para o próximo slide:** “A média esconde quais pares saem cedo e quais chegam às últimas camadas; vamos olhar exemplos concretos.” Os exemplos estão em `outputs/early_exit_10epochs_3seeds/qualitative_examples_combined.csv` e nos CSVs completos de predição por run.

## Arquivos da pasta

- `grafico_tradeoff.png` / `.svg`: figura principal pronta para o slide.
- `tabela_principal.png` / `.svg` / `.csv`: números centrais para slide alternativo ou apêndice.
- `grafico_completo_rte.png` e `grafico_completo_sts.png`: todas as profundidades fixas, úteis no apêndice.
- `tabela_selecao_seeds.csv`: parâmetros escolhidos na validação por seed.
- `gerar_materiais.py`: recria os gráficos e tabelas a partir de `outputs/early_exit_10epochs_3seeds/summary.csv`.

**Nota metodológica para perguntas:** as seeds variaram principalmente o embaralhamento no treino dos heads; o checkpoint de BERTimbau e a cabeça final da camada 12 são os mesmos nas três runs. A latência exclui tokenização e transferência para GPU. O MSE de STS e as tabelas completas estão no relatório `MULTISEED_10_EPOCHS.md`.
