# Fase 2 — custo de execução do book short-vol hedgeado (envelope)

Data: 2026-09-02
Estado: experimento executado; envelope sobre marcação sintética com entradas em
books Tardis reais; nada aprovado para dinheiro real.
Plano: [docs/2026-09-02-plano-pipeline-vol.md](2026-09-02-plano-pipeline-vol.md)

## Pergunta

Quanto do prêmio sobrevive sob cada política de fill na entrada? Comparamos
quatro cenários sobre as mesmas 22 entradas trimestrais, todas hold-to-expiry
com delta-hedge diário (motor da Fase 1, generalizado para cestas de pernas):

- `atm_cross` — straddle ATM vendido no bid exibido (baseline da Fase 1);
- `atm_post_mid` — straddle ATM preenchido no mid (envelope maker; probabilidade
  de fill **não** modelada, é um teto);
- `strangle25_cross` — strangle 25-delta vendido no bid;
- `strangle25_post_mid` — strangle 25-delta no mid.

Os strikes do strangle vêm do book Tardis real de cada entrada, mesmo
vencimento do ATM, IVs de mid invertidas contra o forward de paridade, delta
forward Black-76 alvo de ±0,25 (`select_strangle_by_delta`).

```bash
.venv/bin/python scripts/run_execution_cost_scenarios.py
```

Artefato: `artifacts/execution-cost-scenarios-v1.json`.

## Resultado (22 trades, 0,1 contrato por perna, P&L hedgeado em BTC)

| Cenário | Média/trade | Std | t-stat | Positivos | Diff pareado vs baseline |
|---|---:|---:|---:|---:|---:|
| atm_cross | +0,00110 | 0,00272 | 1,90 | 17/22 | — |
| atm_post_mid | +0,00123 | 0,00275 | **2,09** | 17/22 | +0,000125 ± 0,000047 |
| strangle25_cross | +0,00077 | 0,00214 | 1,69 | 18/22 | -0,000332 ± 0,001364 |
| strangle25_post_mid | +0,00087 | 0,00216 | 1,89 | 18/22 | -0,000234 ± 0,001358 |

## Leitura

1. **Postar em vez de cruzar vale ~11% do edge, de graça em risco.** O ganho
   pareado de +0,000125 BTC/trade tem desvio de 0,000047 (é quase
   determinístico: metade do spread na entrada). Só essa mudança leva o t-stat
   de 1,90 a 2,09. E este envelope só tem custo de entrada; numa política com
   saída antecipada o efeito dobra.
2. **O strangle 25-delta não domina o ATM nesta amostra.** Tem menos variância
   (0,00214 vs 0,00272) e mais trades positivos (18/22), mas média menor; o
   retorno por unidade de risco fica parecido (~0,36 vs ~0,40). Sem
   normalização por vega, a comparação é indicativa. O strangle continua
   candidato para quando o sizing (Fase 3) penalizar cauda, não para aumentar
   média.
3. **Decisão para a Fase 4:** a expressão de referência segue sendo o straddle
   ATM delta-hedgeado, com execução **postada** (maker) como política-alvo e o
   cruzamento como piso conservador nos backtests.

## Limitações declaradas

- `post_mid` assume fill integral no mid: teto de maker sem modelo de fila,
  cancelamento ou adverse selection. A verdade operacional fica entre `cross` e
  `post_mid`.
- Fees maker e taker de opção assumidas iguais (fórmula de cap da Deribit).
- IVs do strangle invertidas contra o forward de paridade do ATM do mesmo
  vencimento; strikes OTM herdam esse proxy.
- Marcas diárias `synthetic_model`; sem margem, liquidação ou portfólio.

## Próximo passo (Fase 3 do plano)

Sizing, margem e ruína: bootstrap do book **hedgeado** (não mais do straddle nu),
fração de Kelly com desconto de incerteza e barreira de liquidação da Deribit.
