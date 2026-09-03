-- Immutable operational record for the sealed volatility paper-trading holdout.
BEGIN;

CREATE TABLE IF NOT EXISTS research.paper_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    as_of TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    holdout_id TEXT NOT NULL,
    code_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'blocked', 'failed')),
    config JSONB NOT NULL,
    input_quality JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (holdout_id, as_of, code_version)
);

CREATE TABLE IF NOT EXISTS research.paper_decision (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES research.paper_run (id),
    decided_at TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('short', 'flat')),
    status TEXT NOT NULL CHECK (status IN ('decided', 'blocked')),
    reason TEXT NOT NULL,
    forecast_rv DOUBLE PRECISION,
    bid_iv DOUBLE PRECISION,
    ask_iv DOUBLE PRECISION,
    forecast_at TIMESTAMPTZ,
    quotes_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, decided_at)
);

CREATE TABLE IF NOT EXISTS research.paper_position (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES research.paper_decision (id),
    instrument_name TEXT NOT NULL,
    leg_type TEXT NOT NULL CHECK (leg_type IN ('call', 'put', 'hedge')),
    side TEXT NOT NULL CHECK (side IN ('short', 'long')),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    contracts DOUBLE PRECISION NOT NULL CHECK (contracts > 0),
    entry_price_btc DOUBLE PRECISION NOT NULL CHECK (entry_price_btc >= 0),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed', 'cancelled')),
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS research.paper_mark (
    position_id UUID NOT NULL REFERENCES research.paper_position (id),
    marked_at TIMESTAMPTZ NOT NULL,
    option_mark_btc DOUBLE PRECISION,
    hedge_mark_usd DOUBLE PRECISION,
    unrealized_pnl_btc DOUBLE PRECISION NOT NULL,
    realized_pnl_btc DOUBLE PRECISION NOT NULL DEFAULT 0,
    margin_estimate_btc DOUBLE PRECISION,
    source TEXT NOT NULL,
    PRIMARY KEY (position_id, marked_at)
);

CREATE INDEX IF NOT EXISTS paper_run_latest_idx
    ON research.paper_run (as_of DESC);
CREATE INDEX IF NOT EXISTS paper_position_open_idx
    ON research.paper_position (status, opened_at DESC);
CREATE INDEX IF NOT EXISTS paper_mark_latest_idx
    ON research.paper_mark (marked_at DESC);

COMMIT;
