# Gate econômico long / short / flat — V1

Data: 2026-09-02  
Decisão: **não promover ainda para entradas mensais**. Pesquisa sem capital real.

## Regra congelada

Usamos o forecast de variância GARCH(1,1) de 14 dias com correção causal de
viés. Para cada straddle ATM:

- `long` se a variância prevista excede a média das variâncias das IVs de ask;
- `short` se fica abaixo da média das variâncias das IVs de bid;
- `flat` se fica dentro dessa faixa executável.

Não existe buffer ou threshold ajustado nas 22 entradas. A decisão usa o último
forecast disponível antes do book das 12:00 UTC. O resultado posterior usa
bid/ask reais da Tardis, fees já adotadas no piloto e settlement oficial da
Deribit. Todas as posições são normalizadas para 0,1 contrato por perna.

## Cobertura

As 4 observações anteriores ao início útil do DVOL continuam fora. Das 22 datas
elegíveis a partir de 2021-04, 18 puderam ser classificadas. As quatro primeiras
não possuem forecast GARCH causalmente corrigido porque o desenho exige 365 dias
de treino e 30 targets já encerrados. Elas permanecem no artefato como
`forecast_unavailable`; não fizemos backfill com uma regra diferente.

## Resultado

| Política | N | P&L total (BTC) | P&L médio | IC95% da média | Positivas |
|---|---:|---:|---:|---:|---:|
| regra congelada | 18 | **+0,01418** | +0,00079 | -0,00160 a +0,00317 | 12 |
| always long | 18 | -0,02010 | -0,00112 | -0,00350 a +0,00127 | 6 |
| always short | 18 | +0,01335 | +0,00074 | -0,00163 a +0,00312 | 12 |

A regra emitiu 12 `long`, 6 `short` e nenhum `flat`. O lado short teve 6/6
resultados positivos e somou +0,01609 BTC. O lado long teve 6/12 positivos e
somou **-0,00191 BTC**. O ganho total da regra sobre always-short foi somente
+0,00083 BTC, e o IC95% do resultado médio cruza zero.

## Leitura e decisão

O forecast separa bem o regime short nesta amostra, mas ainda não demonstra um
regime long economicamente rentável. Por isso o gate falha o requisito de P&L
positivo em cada lado e não autoriza ampliar a frequência mensal agora. Essa é
uma conclusão mais restrita que “long-vol não funciona”: o teste é trimestral,
pequeno, sem hedge e hold-to-expiry.

Para shorts, o P&L exibido é somente o payoff terminal líquido do crédito, fees
e settlement. Ele não modela margem, liquidação nem a trajetória intradiária;
portanto não substitui o stress de cauda já realizado e não autoriza short-vol
em produção.

Artefato: `artifacts/volatility-regime-gate-v1.json`.

## Holdout vivo

O holdout começa em 2026-09-03, horário de São Paulo. Sua política está em
`config/holdout.volatility-live-v1.json`: novos dados só podem ser anexados ao
monitoramento da especificação V1, nunca usados retrospectivamente para mudar
modelo, correção ou fronteiras long/short/flat.
