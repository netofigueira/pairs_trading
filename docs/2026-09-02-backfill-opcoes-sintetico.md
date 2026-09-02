# Backfill sintético de opções BTC — desenho V1

Data: 2026-09-02  
Estado: envelope diário V1 executado; resultado exploratório; nada aprovado
para dinheiro real.

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

## Envelope diário executado

O runner marca os contratos originais uma vez por dia. O close de cada candle
só fica disponível em `timestamp + 1 dia`, impedindo o uso antecipado do preço.
A IV parte da IV de entrada observada e acompanha a variação do último DVOL
disponível. O forward diário usa o close do perp e o yield de basis implícito na
entrada, com choque adicional de -50/0/+50 bps.

```bash
.venv/bin/python scripts/run_synthetic_option_backfill.py
```

Foram avaliadas três regras declaradas em configuração antes da leitura do
resultado, cruzadas com quatro percentis de spread, quatro choques de IV e três
choques de basis: 48 cenários por regra. A posição é de 0,1 contrato por perna.

Cobertura: 22 de 26 entradas. As quatro anteriores a 2021-03 foram recusadas
porque ainda não existia DVOL público; nenhuma IV foi preenchida olhando o
futuro.

### Cenário central — spread P50, IV sem choque, basis neutra

| Regra | Positivas | P&L total | Retorno médio/crédito | Mediana | Pior trade |
|---|---:|---:|---:|---:|---:|
| TP25, stop 2x, saída 3 DTE | 17/22 | +0,01056 BTC | +1,65% | +26,93% | -194,86% |
| TP50, stop 2x, saída 3 DTE | 16/22 | +0,01925 BTC | +4,45% | +21,14% | -194,86% |
| TP50, stop 1,5x, saída 7 DTE | 15/22 | +0,00586 BTC | +0,95% | +12,66% | -67,24% |

O baseline vendido até o vencimento nas mesmas 22 datas teve 14 positivas,
+0,01747 BTC, retorno médio de +2,19%, mediana de +7,80% e pior trade de
-152,62% do crédito. Assim, TP50/stop 2x/3 DTE melhora modestamente o agregado
e a média centrais, mas não melhora a cauda nesta resolução diária.

### Cenário adverso — spread P95, +15 pontos de IV, +50 bps de basis

| Regra | Positivas | P&L total | Retorno médio/crédito | Mediana | Pior trade |
|---|---:|---:|---:|---:|---:|
| TP25, stop 2x, saída 3 DTE | 14/22 | -0,00423 BTC | -8,33% | +5,81% | -194,36% |
| TP50, stop 2x, saída 3 DTE | 13/22 | -0,00012 BTC | -7,12% | +3,74% | -194,36% |
| TP50, stop 1,5x, saída 7 DTE | 6/22 | -0,02978 BTC | -21,59% | -14,06% | -83,49% |

Artefato completo: `artifacts/synthetic-option-backfill-v1.json` (cerca de
1,9 MB), com 3.168 resultados individuais e 144 agregações.

## Decisão da rodada

O envelope sustenta a hipótese de que existe prêmio a capturar, mas **não
aprova uma regra de saída**. TP50/stop 2x/3 DTE quase preserva o P&L agregado no
pior cenário, porém sua média fica negativa e o stop diário permite gaps muito
além de 2x. A regra mais apertada reduz a pior perda, mas destrói resultado sob
estresse.

O próximo gate não é procurar mais combinações de take-profit. É incorporar
margem/liquidação e medir barreiras intradiárias ou, na ausência delas, aplicar
um choque explícito de gap. Só depois faz sentido separar formação e validação
temporal para escolher uma regra sem overfit.
