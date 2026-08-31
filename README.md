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

Exemplo de coleta pública:

```bash
.venv/bin/python scripts/collect_binance_usdm.py --symbol BTCUSDT --interval 1h --days 30
```
