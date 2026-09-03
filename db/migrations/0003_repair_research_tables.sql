-- Repair for databases where 0001 was recorded before its research section
-- existed.  Keep this separate from 0001: migration files are immutable once
-- deployed and public.schema_migration prevents re-running them.
BEGIN;

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

COMMIT;
