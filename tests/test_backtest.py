import pandas as pd

from quant_pairs.backtest import BacktestConfig, run_pair_backtest
from quant_pairs.cointegration import FormationModel


def model() -> FormationModel:
    return FormationModel(
        dependent="Y",
        independent="X",
        alpha=0.0,
        beta=1.0,
        spread_mean=0.0,
        spread_std=1.0,
        coint_t_stat=-4.0,
        coint_pvalue=0.01,
        critical_values=(-3.9, -3.3, -3.0),
        formation_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        formation_end=pd.Timestamp("2024-01-01T01:00:00Z"),
        observations=100,
    )


def prices() -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2024-01-02", periods=6, freq="h", tz="UTC")
    # log(y/x) crosses +2 at bar 1, then mean-reverts at bar 3.
    y = pd.Series([1.0, 10.0, 11.0, 1.0, 1.0, 1.0], index=index, name="Y")
    x = pd.Series([1.0] * len(index), index=index, name="X")
    return y, x


def test_entry_and_exit_fill_on_next_bar_with_costs() -> None:
    y, x = prices()
    result = run_pair_backtest(
        model(),
        y,
        x,
        BacktestConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, max_holding_bars=10, taker_fee_bps=10),
    )

    trade = result.trades.iloc[0]
    assert trade.entry_time == y.index[2]  # crossing at bar 1, fill at bar 2
    assert trade.exit_time == y.index[4]  # mean reversion at bar 3, fill at bar 4
    assert trade.exit_reason == "mean_reversion"
    assert trade.trading_cost > 0
    assert trade.net_pnl < trade.gross_pnl


def test_positive_funding_credits_the_short_leg() -> None:
    y, x = prices()
    funding = pd.DataFrame(
        {
            "funding_time": [y.index[3]],
            "funding_rate": [0.01],
            "mark_price": [10.0],
        }
    )
    result = run_pair_backtest(
        model(),
        y,
        x,
        BacktestConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, max_holding_bars=10),
        dependent_funding=funding,
    )

    assert result.trades.iloc[0].funding_pnl > 0
