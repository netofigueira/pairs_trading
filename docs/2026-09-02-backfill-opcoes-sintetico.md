# Backfill sintético de opções BTC — desenho V1

Data: 2026-09-02  
Estado: infraestrutura e calibração inicial concluídas; backtest de saída ainda
não executado; nada aprovado para dinheiro real.

## Objetivo

Estimar uma faixa plausível de P&L para saídas anteriores ao vencimento quando
não existe histórico contínuo de bid/ask. O resultado será um **envelope de
viabilidade**, não uma reconstrução de fills observados.

## Modelo de preço

A V1 usa Black-76 na convenção inversa documentada pela Deribit. Para forward
`F`, strike `K`, prazo em anos `T` e volatilidade anualizada `sigma`:

```text
d1 = [ln(F/K) + 0,5 sigma² T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
call_BTC = N(d1) - (K/F) N(d2)
put_BTC  = (K/F) N(-d2) - N(-d1)
```

O prêmio resultante é denominado em BTC. A implementação está em
`src/quant_pairs/inverse_options.py`, incluindo valor intrínseco, inversão de
IV por bisseção e geração de quote sintético explicitamente rotulado.

Referência: [Deribit — Inverse Options](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options).

## Forward: aprendizado do controle de um dia

Tratar o mid do BTC-PERPETUAL como o forward do vencimento produziu IVs
incompatíveis para call e put de mesmo strike em 2024-01-01: aproximadamente
74% contra 60%. Isso era erro do proxy, não skew entre as pernas.

A calibração foi corrigida para inferir o forward datado pela paridade inversa:

```text
call - put = 1 - K/F
F = K / [1 - (call - put)]
```

Com os mids pareados, ambas as pernas de 2024-01-01 passaram a implicar 67,04%
de IV. O perp continua útil para selecionar o strike ATM. No backfill, onde a
paridade observada não existirá, a V1 deverá aproximar o forward com perp mais
um cenário explícito de basis; não esconderá o perp como se fosse future
datado.

## Calibração observada de spread

O comando abaixo percorreu os books Tardis já existentes, selecionando o mesmo
straddle ATM e vencimento-alvo de 14 DTE do piloto trimestral:

```bash
.venv/bin/python scripts/calibrate_tardis_option_spreads.py
```

Artefato: `artifacts/tardis-option-spread-calibration-v1.json` (cerca de 38 KB).

Resultados sobre 52 pernas em 26 datas, de 2020-04-01 a 2026-07-01, sem falha
de inversão de IV:

| Cenário | Spread total / prêmio | Half-spread / prêmio | Largura bid-IV--ask-IV |
|---|---:|---:|---:|
| P50 | 3,24% | 1,62% | 1,46 pontos de vol |
| P75 | 3,89% | 1,95% | 2,13 pontos de vol |
| P90 | 5,49% | 2,75% | 3,92 pontos de vol |
| P95 | 7,22% | 3,61% | 4,04 pontos de vol |

O spread mínimo foi 1,24% e o máximo, 10,75% do prêmio. O backfill aplicará o
percentual ao **prêmio da opção**, nunca ao spot do BTC. Para fechar uma posição
vendida, o fill será o ask sintético; para fechar uma comprada, o bid sintético.

## Limites e proteção contra falsa precisão

- Os 26 snapshots são trimestrais, às 12:00 UTC, ATM e entre 7--30 DTE. Não
  representam diretamente spreads intradiários, opções OTM ou os últimos dias
  antes do vencimento.
- O forward por paridade herda ruído dos dois mids exibidos.
- Top-of-book não garante tamanho suficiente. Os percentis de spread não
  substituem o gate de quantidade.
- O mark ou o mid teórico nunca será tratado como fill.
- Cada linha sintética terá `source = synthetic_model` e ficará separada de
  quotes observados.
- A estratégia deverá sobreviver a uma grade de P50/P75/P90/P95 de spread e
  choques de +5/+10/+15 pontos de IV. Resultado apenas no cenário central será
  rejeitado.

## Próxima implementação

1. alinhar a série diária de BTC e DVOL sem usar informação futura;
2. construir cenários explícitos de basis do forward;
3. marcar diariamente call e put do contrato originalmente vendido;
4. simular recompras no ask sintético, fees e slippage;
5. comparar regras pré-declaradas de lucro, stop, DTE mínimo e choque de IV;
6. reportar o envelope completo, inclusive cauda e pior cenário, sem otimizar a
   regra sobre toda a amostra.
