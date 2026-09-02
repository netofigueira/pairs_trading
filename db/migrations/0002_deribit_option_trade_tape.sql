BEGIN;

-- Immutable, normalized public trade tape.  The time column is part of the
-- key because TimescaleDB unique indexes on hypertables must include it.
CREATE TABLE IF NOT EXISTS market.option_trade (
    venue TEXT NOT NULL DEFAULT 'deribit',
    currency TEXT NOT NULL,
    traded_at TIMESTAMPTZ NOT NULL,
    trade_id TEXT NOT NULL,
    trade_seq BIGINT,
    instrument_name TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL CHECK (price >= 0),
    mark_price DOUBLE PRECISION,
    iv DOUBLE PRECISION,
    index_price DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    contracts DOUBLE PRECISION,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    tick_direction SMALLINT,
    liquidation TEXT,
    source TEXT NOT NULL,
    source_ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, currency, trade_id, traded_at)
);

SELECT create_hypertable(
    'market.option_trade',
    'traded_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS option_trade_instrument_time_idx
    ON market.option_trade (instrument_name, traded_at DESC);
CREATE INDEX IF NOT EXISTS option_trade_currency_time_idx
    ON market.option_trade (currency, traded_at DESC);

-- File-level provenance makes the local gzip cache replayable and each load
-- idempotent, without treating the local machine as a source of truth.
CREATE TABLE IF NOT EXISTS market.tape_ingestion_file (
    source_path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    source_first_at TIMESTAMPTZ,
    source_last_at TIMESTAMPTZ,
    source_rows BIGINT NOT NULL CHECK (source_rows >= 0),
    loaded_rows BIGINT NOT NULL CHECK (loaded_rows >= 0),
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
