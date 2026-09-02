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

### P1.5 — carry até o vencimento com dados 100% gratuitos

- Validado que a API pública da Deribit fornece `get_delivery_prices` (2.595
  registros diários, desde ~2019) e `get_funding_rate_history` (funding horário
  para datas antigas). Isso fecha payoff e funding sem provedor pago.
- `funding.py`: histórico horário paginado em janelas de 30 dias, cache CSV e
  `funding_pnl_btc` para perp inverso (long paga funding positivo; cobertura
  horária incompleta é rejeitada, não silenciada).
- `settlement.py`: delivery prices paginados com cache, payoff inverso
  (`max(0, S−K)/S`) e fee de liquidação (0,015% do subjacente, cap de 12,5% do
  valor da opção; OTM não paga).
- O runner intraday agora aceita `funding` e preenche
  `net_delta_hedged_pnl_btc` (status `delta_hedged_intraday_with_funding`).
- `tardis_carry.py` + `scripts/run_tardis_carry.py`: entrada no ask real da
  Tardis, hold até o vencimento, saída pelo preço oficial de liquidação, com
  duas variantes: sem hedge e hedge estático pelo delta observado na entrada,
  carregado até o settle com funding horário. A saída do hedge usa o
  delivery price como fill (aproximação declarada: não há book do perp no
  vencimento nas amostras gratuitas; `--hedge-exit-slippage-bps` desloca o
  fill contra a posição para estressar o spread ausente).
- Validação de funding exige cadência horária exata, não só contagem de linhas.
- Resultado real de 2024-01-01 (straddle BTC-12JAN24-43000, 10,8 dias):
  prêmio de entrada 0,093 BTC, payoff 0,06405 BTC, hedge -286 contratos,
  funding +0,000262 BTC; líquido sem hedge -0,029699 BTC e com hedge estático
  **-0,034211 BTC**. Uma observação; sem significado estatístico.

Comandos:

```bash
.venv/bin/python scripts/run_tardis_carry.py --date 2024-01-01 --with-options-chain
.venv/bin/python scripts/run_tardis_intraday.py --date 2024-01-01 \
  --entry-time 12:00:00 --exit-time 20:00:00 --with-options-chain --with-funding
```

## Próximo passo exato

1. Baixar os primeiros dias mensais de 2019-05 até hoje
   (`collect_tardis_monthly_samples.py`) e rodar `run_tardis_carry.py` em todos,
   nas duas variantes, medindo cobertura e falhas por mês.
2. Extrair/cachear somente Greeks necessários do `options_chain` (1,87 GB/dia),
   evitando redescompactar metade do dia a cada cenário.
3. Analisar a amostra completa (~75-80 meses) como gate de viabilidade,
   declarando: só entradas no dia 1, hedge estático mistura gamma do caminho
   com prêmio de vol, saída do hedge aproximada pelo delivery price, amostra
   pequena e viés de calendário.

O carry até o vencimento contorna a falta de quotes contínuos de opções: só a
entrada exige book executável (Tardis gratuito no dia 1 de cada mês); payoff,
hedge e funding vêm de APIs públicas. Rebalancear o hedge no meio do caminho
continua exigindo dados pagos ou coleta própria.

## Arquivos relevantes

- `src/quant_pairs/dvol.py`: P0 DVOL × RV sem leakage.
- `src/quant_pairs/tardis.py`: downloader dos CSVs Tardis.
- `src/quant_pairs/tardis_quotes.py`: reconstrução top-of-book por updates.
- `src/quant_pairs/tardis_options.py`: parser de contrato e seleção com regra
  pré-declarada: primeiro o vencimento mais próximo do `target_dte` (default
  14), depois o strike ATM dentro dele.
- `src/quant_pairs/tardis_intraday.py`: round-trip intraday executável com
  delta observado, hedge inverso no perp e líquido final bloqueado sem funding.
- `src/quant_pairs/funding.py`: funding público horário + P&L de funding.
- `src/quant_pairs/settlement.py`: delivery prices oficiais, payoff e fee.
- `src/quant_pairs/tardis_carry.py`: carry até o vencimento (sem hedge e
  hedge estático).
- `scripts/run_tardis_carry.py`: CLI do gate de carry mensal.
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
