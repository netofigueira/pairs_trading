BEGIN;

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS research;
CREATE SCHEMA IF NOT EXISTS execution;

CREATE TABLE IF NOT EXISTS market.instrument (
    venue TEXT NOT NULL,
    market_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    base_asset TEXT,
    quote_asset TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, market_type, symbol)
);

CREATE TABLE IF NOT EXISTS market.candle (
    venue TEXT NOT NULL,
    market_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    open NUMERIC NOT NULL CHECK (open > 0),
    high NUMERIC NOT NULL CHECK (high > 0),
    low NUMERIC NOT NULL CHECK (low > 0),
    close NUMERIC NOT NULL CHECK (close > 0),
    volume NUMERIC NOT NULL CHECK (volume >= 0),
    quote_volume NUMERIC,
    trade_count BIGINT,
    taker_buy_base_volume NUMERIC,
    taker_buy_quote_volume NUMERIC,
    source_ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, market_type, symbol, interval, open_time),
    FOREIGN KEY (venue, market_type, symbol)
        REFERENCES market.instrument (venue, market_type, symbol)
);

SELECT create_hypertable('market.candle', 'open_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS candle_lookup_idx
    ON market.candle (venue, market_type, symbol, interval, open_time DESC);

CREATE TABLE IF NOT EXISTS market.funding_rate (
    venue TEXT NOT NULL,
    market_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    funding_time TIMESTAMPTZ NOT NULL,
    funding_rate NUMERIC NOT NULL,
    mark_price NUMERIC,
    source_ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, market_type, symbol, funding_time),
    FOREIGN KEY (venue, market_type, symbol)
        REFERENCES market.instrument (venue, market_type, symbol)
);

SELECT create_hypertable('market.funding_rate', 'funding_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS funding_lookup_idx
    ON market.funding_rate (venue, market_type, symbol, funding_time DESC);

CREATE TABLE IF NOT EXISTS research.formation_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_version TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    config JSONB NOT NULL,
    data_start TIMESTAMPTZ NOT NULL,
    data_end TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS research.candidate (
    formation_run_id UUID NOT NULL REFERENCES research.formation_run (id),
    dependent_symbol TEXT NOT NULL,
    independent_symbol TEXT NOT NULL,
    hedge_alpha DOUBLE PRECISION NOT NULL,
    hedge_beta DOUBLE PRECISION NOT NULL,
    coint_t_stat DOUBLE PRECISION NOT NULL,
    coint_pvalue DOUBLE PRECISION NOT NULL,
    half_life_bars DOUBLE PRECISION,
    accepted BOOLEAN NOT NULL,
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (formation_run_id, dependent_symbol, independent_symbol)
);

CREATE TABLE IF NOT EXISTS research.backtest_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    formation_run_id UUID REFERENCES research.formation_run (id),
    code_version TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    config JSONB NOT NULL,
    data_start TIMESTAMPTZ NOT NULL,
    data_end TIMESTAMPTZ NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS research.backtest_trade (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_run_id UUID NOT NULL REFERENCES research.backtest_run (id),
    dependent_symbol TEXT NOT NULL,
    independent_symbol TEXT NOT NULL,
    direction SMALLINT NOT NULL CHECK (direction IN (-1, 1)),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL,
    exit_reason TEXT NOT NULL,
    gross_pnl NUMERIC NOT NULL,
    fee_pnl NUMERIC NOT NULL,
    slippage_pnl NUMERIC NOT NULL,
    funding_pnl NUMERIC NOT NULL,
    net_pnl NUMERIC NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS execution.order_intent (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    environment TEXT NOT NULL CHECK (environment IN ('paper', 'live')),
    status TEXT NOT NULL,
    payload JSONB NOT NULL,
    CHECK (environment = 'paper')
);

COMMIT;
