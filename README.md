# Pairs Trading

## Estado do projeto

O código é um protótipo de pesquisa para pairs trading por cointegração em ações brasileiras. Ele **não está pronto para operar**: antes de qualquer conexão com corretora/exchange, a metodologia e o backtest precisam da correção descrita em [docs/2026-08-31-auditoria-metodologia.md](docs/2026-08-31-auditoria-metodologia.md).

A trilha de P&D para a plataforma quant cripto está em [docs/2026-08-31-roadmap-pesquisa-quant-cripto.md](docs/2026-08-31-roadmap-pesquisa-quant-cripto.md). O plano em fases do pipeline de volatilidade está em [docs/2026-09-02-plano-pipeline-vol.md](docs/2026-09-02-plano-pipeline-vol.md); a Fase 1 (carry delta-hedgeado) está executada em [docs/2026-09-02-carry-delta-hedgeado.md](docs/2026-09-02-carry-delta-hedgeado.md) e é reproduzível via `scripts/run_delta_hedged_carry.py`.

## Ambiente de pesquisa

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q tests
```

Os dados públicos coletados ficam em `data/` (ignorado pelo Git). O primeiro adaptador é Binance USDⓈ-M: ele não usa API key e só baixa candles/funding para pesquisa e paper trading.

## Banco de dados

O armazenamento operacional é o TimescaleDB (PostgreSQL), com a estrutura versionada em `db/migrations/`. Ele não é exposto à internet: na VM a porta fica limitada ao próprio host.

Para desenvolver contra o banco, defina uma URL de conexão (não a versione):

```bash
export QUANT_PAIRS_DATABASE_URL='postgresql://quantpairs:SENHA@127.0.0.1:5433/quantpairs'
.venv/bin/python scripts/migrate_database.py
```

O coletor pode gravar no banco, além do data lake local:

```bash
.venv/bin/python scripts/collect_binance_usdm.py --symbol BTCUSDT --interval 1h --days 30 --database-url "$QUANT_PAIRS_DATABASE_URL"
```

O desenho da futura interface está em [docs/2026-08-31-plataforma-operacional.md](docs/2026-08-31-plataforma-operacional.md). A interface é inicialmente de pesquisa e paper trading; ela não envia ordens.

O painel privado possui uma página de pesquisa de volatilidade em
`/volatility`. Ela lê somente um artefato compacto e versionado, regenerado por:

```bash
.venv/bin/python scripts/build_volatility_report.py
.venv/bin/uvicorn quant_pairs.dashboard_api:app --reload
```

## Experimento intraday declarado

O experimento inicial em candles fechados de 1h está em
[`config/experiment.crypto-usdm-intraday-v1.json`](config/experiment.crypto-usdm-intraday-v1.json):
formação de 30, 60 e 90 dias, trade OOS semanal sem sobreposição e os últimos
30 dias reservados como holdout final. O script compara as variantes somente no
histórico anterior ao holdout; ele não executa nem revela métricas do holdout.

```bash
.venv/bin/python scripts/walk_forward_timescaledb.py
```

O experimento de escala de volatilidade preserva a seleção/hedge de 90 dias e
compara z-score fixo, EWMA de 72h e desvio rolling de 72h, sem tocar no holdout:

```bash
.venv/bin/python scripts/walk_forward_timescaledb.py \
  --experiment config/experiment.crypto-usdm-volatility-v1.json
```

Exemplo de coleta pública:

```bash
.venv/bin/python scripts/collect_binance_usdm.py --symbol BTCUSDT --interval 1h --days 30
```

## Pesquisa de volatilidade em opções

A trilha V0 coleta somente dados públicos da Deribit: cadeia de opções com
bid/ask e IV marcada, além do índice DVOL. Não há autenticação, conta ou envio
de ordens nesse adaptador. Os snapshots locais permitem pesquisar IV versus
volatilidade realizada sem supor que o preço de marca seja executável.

```bash
.venv/bin/python scripts/collect_deribit_options.py --currency BTC --currency ETH --dvol-days 30
```

A P0 pública usa o DVOL diário histórico e o BTC-PERPETUAL diário para medir
IV em `t` contra RV realizada nos 30 dias seguintes. É diagnóstico de
calibração, não backtest de P&L de opções:

```bash
.venv/bin/python scripts/collect_deribit_p0.py --start 2021-01-01T00:00:00Z
.venv/bin/python scripts/diagnose_deribit_p0.py
```

O gate P1 audita se cada round-trip usa o mesmo contrato e lados executáveis
(`ask` na compra e `bid` na venda); ele não calcula P&L a partir de marks:

```bash
.venv/bin/python scripts/audit_option_quote_coverage.py
```

Para o gate gratuito de plumbing P1, a Tardis disponibiliza arquivos reais no
primeiro dia de cada mês. O coletor baixa apenas quotes; `options_chain` é
opcional e muito maior:

```bash
.venv/bin/python scripts/collect_tardis_monthly_samples.py --start 2024-01 --end 2024-12
```

O desenho e os gates de validação estão no [roadmap cripto](docs/2026-08-31-roadmap-pesquisa-quant-cripto.md).

A base do backfill de saídas usa Black-76 inverso e calibra cenários de spread
nos books trimestrais observados. Quotes modelados permanecem rotulados e não
são confundidos com execução real:

```bash
.venv/bin/python scripts/calibrate_tardis_option_spreads.py
```

Metodologia, percentis e limitações: [backfill sintético V1](docs/2026-09-02-backfill-opcoes-sintetico.md).

O envelope diário de saída antecipada é regenerado por:

```bash
.venv/bin/python scripts/run_synthetic_option_backfill.py
```

Ele usa entrada vendida no bid observado e recompra no ask sintético. O
resultado é diagnóstico de viabilidade e não inclui margem ou liquidação.

A distribuição condicional de perda reamostra conjuntamente retorno do BTC e
mudança do DVOL em blocos, sem adicionar gaps à trajetória histórica:

```bash
.venv/bin/python scripts/run_short_straddle_bootstrap.py
```

O resultado não é probabilidade de ruína, pois ainda não inclui capital,
sizing, margem ou liquidação. Veja a [metodologia do bootstrap](docs/2026-09-02-bootstrap-distribuicao-perda.md).

O monitor de forecast compara RV rolling, EWMA e GARCH(1,1) nos horizontes de
14 e 30 dias. Ele publica o último snapshot em `/volatility`, sem convertê-lo
automaticamente em ordem:

```bash
.venv/bin/python scripts/forecast_btc_volatility.py
```

Metodologia e rotina diária: [forecast de volatilidade V1](docs/2026-09-02-forecast-volatilidade.md).
Comparação walk-forward entre refit diário e mensal:
[cadência de refit do GARCH](docs/2026-09-02-walk-forward-refit-garch.md).

Uma cross-section intraday pode validar seleção ATM, continuidade do mesmo
contrato e fills ask/bid. Com `options_chain`, o runner usa deltas observados
no mesmo instante para neutralizar a entrada com BTC-PERPETUAL. O líquido final
permanece nulo até integrar funding; o campo `before_funding` não é P&L final:

```bash
.venv/bin/python scripts/run_tardis_intraday.py \
  --date 2024-01-01 --entry-time 12:00:00 --exit-time 20:00:00 \
  --with-options-chain
```

Arquivos Parquet históricos de terceiros passam por um gate de qualidade antes
de qualquer backtest; quotes sem os dois lados, cruzadas, expiradas ou modeladas
não são tratadas como execução:

```bash
.venv/bin/python scripts/inspect_volar_chain.py /caminho/chain.parquet
```

Para uma chave Volar guardada em `.env` como `VOLAR_API_KEY`, a coleta histórica
sandbox continua somente em leitura e não imprime o segredo:

```bash
.venv/bin/python scripts/collect_volar_chain.py --at 2026-08-25T12:00:00Z
```
