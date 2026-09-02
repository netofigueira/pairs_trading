# Handoff — pesquisa de volatilidade cripto

Data: 2026-09-02
Commit-base: `69727b7 Add volatility research dashboard`
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

O piloto trimestral long-only foi concluído antes da aquisição massiva de
Greeks. Foram coletadas 26 datas de 2020-04 a 2026-07 (6,7 GB de quotes), com
22 fills de um contrato. Apenas 7 foram positivos; retorno mediano sobre prêmio
de -9,91% e P&L total de -0,177873 BTC. Uma sensibilidade de 0,1 contrato
recuperou as quatro falhas de tamanho; nas 26 observações, 8 foram positivas e
o retorno mediano foi -12,68%. A compra sistemática de straddle foi reprovada
como estratégia standalone neste gate.

O relatório compacto `artifacts/volatility-research-v1.json` reúne 66 janelas
independentes IV--RV, 1.958 pontos diários de contexto e 26 observações do
carry em 422 KB. A rota `/volatility` do dashboard renderiza KPIs, scatter de
calibração, VRP em variância, barras do carry e cobertura sem consultar os
arquivos brutos. A API é `/api/v1/volatility/research`; ambos funcionam sem
conexão com o TimescaleDB.

Próximas ações:

1. Não baixar `options_chain` para todos os trimestres enquanto não houver uma
   pergunta adicional que justifique ~150 GB de tráfego.
2. Atribuir os 26 resultados por regime DVOL, magnitude do movimento e DTE com
   os dados públicos já existentes, sem escolher regra depois de ver o P&L.
3. Não inverter mecanicamente o resultado para short-vol: qualquer experimento
   vendido precisa modelar margem, liquidação e cauda antes de ser declarado.

### P1.6 — base do backfill teórico de saídas

- Implementado Black-76 inverso, em BTC, com valor intrínseco e inversão de IV
  por bisseção.
- O controle de 2024-01-01 rejeitou o perp como substituto direto do forward:
  ele gerava IV de aproximadamente 74% na call e 60% na put. Pela paridade
  call--put, ambas passaram a 67,04%.
- Calibração de 52 pernas ATM nas 26 datas trimestrais, sem falhas de inversão:
  half-spread mediano de 1,62% do prêmio; P75 de 1,95%; P90 de 2,75%; P95 de
  3,61%. Largura bid-IV--ask-IV P95 de 4,04 pontos de vol.
- O artefato compacto versionado é
  `artifacts/tardis-option-spread-calibration-v1.json`.
- O próximo passo é o painel diário de marcação sintética com DVOL, cenários de
  basis e recompra de posições vendidas no ask sintético.
- Envelope executado em 22 datas com 48 cenários por regra. No centro,
  TP50/stop 2x/3 DTE fez +0,01925 BTC contra +0,01747 BTC do hold comparável,
  mas teve pior trade de -194,86% do crédito. Em P95 de spread, +15 pontos de
  IV e +50 bps de basis, ficou em -0,00012 BTC e retorno médio de -7,12%.
- Nenhuma regra foi aprovada. Próximo gate: margem, liquidação e gap
  intradiário antes de qualquer nova otimização de saída.
- O stress posterior força um choque de +/-10%, +/-15% ou +/-20% uma vez em
  cada trade e no pior dia ex post. Após corrigir cancelamento numérico no
  Black-76, a cobertura permaneceu 22 datas. Choques de -10% e +10% tornaram o
  agregado central negativo. Isso reprova o stop diário como proteção, mas não
  estima retorno esperado: o evento foi imposto em 100% dos trades e não pela
  frequência histórica.

Desenho e limitações: `docs/2026-09-02-backfill-opcoes-sintetico.md`.

O carry até o vencimento contorna a falta de quotes contínuos de opções: só a
entrada exige book executável (Tardis gratuito no dia 1 de cada mês); payoff,
hedge e funding vêm de APIs públicas. Rebalancear o hedge no meio do caminho
continua exigindo dados pagos ou coleta própria.

## Arquivos relevantes

- `src/quant_pairs/dvol.py`: P0 DVOL × RV sem leakage.
- `src/quant_pairs/inverse_options.py`: Black-76 inverso, IV implícita e quote
  sintético rotulado.
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
- `src/quant_pairs/volatility_report.py`: payload compacto para visualização.
- `scripts/build_volatility_report.py`: regenera o artefato do dashboard.
- `scripts/calibrate_tardis_option_spreads.py`: calibra spread e largura de IV
  nos ATM trimestrais observados.
- `src/quant_pairs/synthetic_option_backfill.py`: marcação diária sem leakage e
  avaliação das regras de recompra.
- `scripts/run_synthetic_option_backfill.py`: executa o envelope de 144
  combinações agregadas.
- `src/quant_pairs/static/volatility.html`: página autocontida de pesquisa.
- `docs/2026-09-01-piloto-carry-trimestral.md`: desenho, resultado e decisão do
  gate trimestral long-only.
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
