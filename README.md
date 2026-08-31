# Pairs Trading

## Estado do projeto

O código é um protótipo de pesquisa para pairs trading por cointegração em ações brasileiras. Ele **não está pronto para operar**: antes de qualquer conexão com corretora/exchange, a metodologia e o backtest precisam da correção descrita em [docs/2026-08-31-auditoria-metodologia.md](docs/2026-08-31-auditoria-metodologia.md).

A trilha de P&D para a plataforma quant cripto está em [docs/2026-08-31-roadmap-pesquisa-quant-cripto.md](docs/2026-08-31-roadmap-pesquisa-quant-cripto.md).

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

Exemplo de coleta pública:

```bash
.venv/bin/python scripts/collect_binance_usdm.py --symbol BTCUSDT --interval 1h --days 30
```
