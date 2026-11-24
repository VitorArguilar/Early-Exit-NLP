# Figuras e tabelas para a apresentação

Todos os arquivos PNG têm uma versão SVG de mesmo nome. As tabelas também estão em CSV. Execute `../.venv/bin/python gerar_apresentacao.py` a partir desta pasta para recriar os materiais a partir dos resultados salvos.

## Curvas e escolha de épocas

- `grafico_30_epocas_rte` e `grafico_30_epocas_sts`: qualidade e perda de treino/validação, do modelo base e dos heads, nas épocas pares de 2 a 30. As versões `_normalizado` mostram a evolução relativa de cada curva.
- `tabela_epocas_escolhidas`: melhor época de validação para cada tarefa e componente nesse estudo exploratório.

**Protocolo das curvas:** uma seed (42). Os heads dessa rodada foram treinados sobre um encoder congelado de 7 épocas; suas épocas não são épocas adicionais do encoder. O estudo de 30 épocas não avaliou o teste.

## Comparação dos métodos de Early Exit

- `tabela_total_metricas_solucoes`: médias no teste das nove configurações com resultados publicados, usando três seeds.
- `grafico_saida_fixa` e `tabela_saida_fixa`: qualidade no teste por camada fixa.
- `grafico_confianca` e `tabela_confianca`: macro F1 de validação por limiar em RTE e resultado de teste dos limiares selecionados na validação. Confiança de classes não se aplica ao STS.
- `grafico_previsoes_consecutivas` e `tabela_previsoes_consecutivas`: qualidade de validação por paciência e tolerância. Nenhuma configuração desse método foi selecionada para teste no protocolo histórico de três seeds; as tabelas indicam “não avaliado”.

**Protocolo da comparação:** encoder de 7 épocas, heads de 10 épocas, seeds 42, 123 e 2026. Não misturar a qualidade do teste dessas soluções com os máximos de validação das curvas de 30 épocas. A latência exclui tokenização e transferência para GPU, conforme o script original de avaliação.

Fontes: `../outputs/long_overfit/{base,heads}/`, `../final/tabela_epocas_selecionadas_30.csv` e `../outputs/early_exit_10epochs_3seeds/{summary,all_validation,all_test}.csv`.
