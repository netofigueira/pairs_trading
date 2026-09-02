# Handoff — pesquisa de volatilidade cripto

Data: 2026-09-01  
Commit-base: `df557f1 Add observed-delta hedge to Tardis gate`
Estado: P0 concluída; P1 de infraestrutura em andamento; nada aprovado para dinheiro real.

## Onde paramos

O pairs trading intraday V1 foi rejeitado após incluir funding, matching e
custos. O projeto migrou para a trilha independente de volatilidade em opções
BTC/Deribit.

### P0 — concluída

- Dados públicos Deribit: 1.988 barras diárias de DVOL e 2.070 barras diárias
  de BTC-PERPETUAL.
- Diagnóstico sem leakage: DVOL em `t` contra RV anualizada dos 30 retornos
  posteriores.
- 66 janelas não sobrepostas: IV média 62,31%, RV futura 51,68%, diferença de
  10,64 pontos de vol; IV excedeu RV em 75,76%.
- Interpretação: hipótese de prêmio de variância para investigar; **não** é P&L
  de opção, nem aprovação de short-vol.

Comandos:

```bash
.venv/bin/python scripts/collect_deribit_p0.py --start 2021-01-01T00:00:00Z
.venv/bin/python scripts/diagnose_deribit_p0.py
```

### P1 — infraestrutura executável

- Volar sandbox: 27 snapshots BTC / 13 dias; 2.412 quotes executáveis; não
  basta para backtest histórico.
- Deribit público não resolve P1 histórica: `get_mark_price_history` rejeita
  contratos expirados (`instrument is not active`), o que impediria reconstruir
  a cadeia sem viés de sobrevivência.
- Tardis libera CSVs reais no primeiro dia de cada mês sem chave. O `GET`
  funciona, embora `HEAD` retorne 404 no CDN.
- Já baixado localmente (ignorado pelo Git):
  - opções Deribit de 2024-01-01: 133 MB;
  - quotes BTC-PERPETUAL de 2024-01-01: 8,2 MB.
  - options_chain Deribit de 2024-01-01: 1,87 GB.
- O reconstrutor de quote incremental funciona e exige bid, ask e tamanho nos
  dois lados, sem cruzamento e com frescor máximo configurável por campo. Uma
  atualização recente do ask não mascara um bid antigo.
- Evidência no snapshot de opções de 2024-01-01: 1.316 books completos; 948
  com atualização nos últimos 5 minutos do dia. Spread relativo mediano de
  717,5 bps, reforçando que mark não deve ser usado como fill.
- O runner intraday ask/bid e delta-hedged foi validado em 2024-01-01,
  12:00--20:00 UTC. Ele selecionou o straddle BTC-12JAN24-43000, manteve os
  mesmos contratos na saída e leu deltas de exchange de +0,53343/-0,46657.
  A exposição de +0,06686 BTC foi coberta com -286 contratos do perp; residual
  de -0,00010 BTC.
- Resultado por straddle: P&L mid conjunto de +0,003181 BTC, spread conjunto
  de 0,005001 BTC e fees de opções+perp de 0,001266 BTC; líquido **antes de
  funding** de -0,003086 BTC. O líquido final permanece nulo no artefato até
  integrar funding. É uma observação de plumbing, sem significado estatístico.
- O corte as-of usa `local_timestamp` (horário de captura), impedindo que uma
  mensagem da exchange recebida depois da decisão entre retrospectivamente.

Comandos:

```bash
.venv/bin/python scripts/collect_tardis_monthly_samples.py --start 2024-01 --end 2024-12
.venv/bin/python scripts/inspect_tardis_quotes.py \
  data/market/tardis/deribit/quotes/2024-01-01/OPTIONS.csv.gz \
  --max-age-seconds 300
.venv/bin/python scripts/run_tardis_intraday.py \
  --date 2024-01-01 --entry-time 12:00:00 --exit-time 20:00:00 \
  --with-options-chain
```

## Próximo passo exato

Completar a contabilidade e tornar a amostra mensal eficiente:

1. Integrar funding histórico do BTC-PERPETUAL no intervalo; até isso, não
   preencher `net_delta_hedged_pnl_btc`.
2. Extrair/cachear somente Greeks necessários do `options_chain` de 1,87 GB,
   evitando redescompactar metade do dia a cada cenário.
3. Rodar múltiplas janelas intraday pré-declaradas nos primeiros dias mensais
   gratuitos e medir cobertura/falhas, sem tratar as observações como backtest
   de carry de 7--30 dias.

Este runner é apenas validação de dados e execução intraday. As amostras
mensais não permitem avaliar carregar opções por 7--30 dias. Um backtest
contínuo requer quotes históricos pagos ou coleta própria acumulada.

## Arquivos relevantes

- `src/quant_pairs/dvol.py`: P0 DVOL × RV sem leakage.
- `src/quant_pairs/tardis.py`: downloader dos CSVs Tardis.
- `src/quant_pairs/tardis_quotes.py`: reconstrução top-of-book por updates.
- `src/quant_pairs/tardis_options.py`: parser de contrato e seleção ATM.
- `src/quant_pairs/tardis_intraday.py`: round-trip intraday executável com
  delta observado, hedge inverso no perp e líquido final bloqueado sem funding.
- `scripts/run_tardis_intraday.py`: CLI do gate intraday.
- `scripts/inspect_tardis_quotes.py`: auditor de livros executáveis.
- `docs/2026-08-31-roadmap-pesquisa-quant-cripto.md`: diário completo e
  decisões metodológicas.

## Regras de continuidade

- Não usar `mark_price`/`mark_iv` como preço de execução.
- Nunca usar `modeled_surface` da Volar em P&L executável.
- Não chamar dados de trades-only de bid/ask; servem para regime/sanity check.
- Não rodar o holdout de pairs já reservado: V1 foi rejeitada antes disso.
- Antes de qualquer capital real: P&L fora da amostra, custos, hedge, funding,
  cauda e paper trading.
