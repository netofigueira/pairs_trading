"""Deribit-style inverse BTC option pricing and conservative synthetic quotes.

Prices are denominated in BTC.  The model uses the expiry forward, rather
than spot, and follows the Black-76 form documented for Deribit inverse
options.  Synthetic quotes are a research aid; they are never labelled as
observed or executable market data.
"""

from __future__ import annotations

import math
from typing import Literal

OptionType = Literal["call", "put"]


def inverse_option_price(
    option_type: OptionType,
    *,
    forward: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> float:
    """Return a Black-76 inverse-option value in BTC per contract.

    ``volatility`` is annualized as a decimal (for example, 0.60 is 60%).
    At expiry or zero volatility, the function returns intrinsic value under
    the supplied forward.
    """

    _validate_inputs(option_type, forward, strike, time_years, volatility)
    if time_years == 0 or volatility == 0:
        return inverse_intrinsic_value(option_type, forward=forward, strike=strike)

    standard_deviation = volatility * math.sqrt(time_years)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * time_years) / (
        standard_deviation
    )
    d2 = d1 - standard_deviation
    if option_type == "call":
        value = _normal_cdf(d1) - strike / forward * _normal_cdf(d2)
    else:
        value = strike / forward * _normal_cdf(-d2) - _normal_cdf(-d1)
    # Deep OTM values can suffer cancellation at machine precision.  An
    # option value cannot fall below inverse intrinsic value.
    intrinsic = inverse_intrinsic_value(option_type, forward=forward, strike=strike)
    return max(value, intrinsic)


def inverse_intrinsic_value(
    option_type: OptionType, *, forward: float, strike: float
) -> float:
    """Return inverse intrinsic value in BTC under ``forward``."""

    _validate_inputs(option_type, forward, strike, 0.0, 0.0)
    if option_type == "call":
        return max(1.0 - strike / forward, 0.0)
    return max(strike / forward - 1.0, 0.0)


def inverse_forward_from_parity(
    *, call_price_btc: float, put_price_btc: float, strike: float
) -> float:
    """Infer the dated forward from inverse put-call parity at one strike.

    For Deribit inverse prices, ``call - put = 1 - strike / forward``.
    Paired mids therefore provide a better calibration forward than treating
    the perpetual contract as if it expired with the option.
    """

    if call_price_btc < 0 or put_price_btc < 0:
        raise ValueError("option prices cannot be negative")
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be finite and positive")
    denominator = 1.0 - (call_price_btc - put_price_btc)
    if denominator <= 0:
        raise ValueError("option prices imply a non-positive forward denominator")
    return strike / denominator


def implied_volatility(
    option_type: OptionType,
    *,
    price_btc: float,
    forward: float,
    strike: float,
    time_years: float,
    max_volatility: float = 10.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Invert an observed BTC option price to annualized implied volatility.

    Bisection is deliberately used instead of a fragile Newton step.  Prices
    outside model bounds fail loudly, which also exposes a poor forward proxy.
    """

    _validate_inputs(option_type, forward, strike, time_years, 0.0)
    if price_btc < 0:
        raise ValueError("price_btc cannot be negative")
    if time_years <= 0:
        raise ValueError("time_years must be positive when solving implied volatility")
    if max_volatility <= 0 or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("solver controls must be positive")

    intrinsic = inverse_intrinsic_value(option_type, forward=forward, strike=strike)
    if price_btc < intrinsic - tolerance:
        raise ValueError("price_btc is below inverse intrinsic value")
    if abs(price_btc - intrinsic) <= tolerance:
        return 0.0

    ceiling = inverse_option_price(
        option_type,
        forward=forward,
        strike=strike,
        time_years=time_years,
        volatility=max_volatility,
    )
    if price_btc > ceiling + tolerance:
        raise ValueError("price_btc exceeds the configured model volatility bound")

    low, high = 0.0, max_volatility
    for _ in range(max_iterations):
        middle = (low + high) / 2
        value = inverse_option_price(
            option_type,
            forward=forward,
            strike=strike,
            time_years=time_years,
            volatility=middle,
        )
        if abs(value - price_btc) <= tolerance:
            return middle
        if value < price_btc:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def synthetic_quote(
    theoretical_mid_btc: float,
    *,
    relative_half_spread: float,
    minimum_half_spread_btc: float = 0.0,
) -> dict[str, float]:
    """Create a labelled synthetic bid/ask around a theoretical BTC mid.

    The relative spread is applied to the option premium, never to BTC spot.
    The absolute floor captures tick size or a deliberately pessimistic
    liquidity allowance.
    """

    if theoretical_mid_btc < 0:
        raise ValueError("theoretical_mid_btc cannot be negative")
    if relative_half_spread < 0 or minimum_half_spread_btc < 0:
        raise ValueError("synthetic spread assumptions cannot be negative")
    half_spread = max(
        theoretical_mid_btc * relative_half_spread, minimum_half_spread_btc
    )
    return {
        "source": "synthetic_model",
        "mid_btc": theoretical_mid_btc,
        "bid_btc": max(theoretical_mid_btc - half_spread, 0.0),
        "ask_btc": theoretical_mid_btc + half_spread,
        "half_spread_btc": half_spread,
        "relative_half_spread": relative_half_spread,
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _validate_inputs(
    option_type: str,
    forward: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError("forward must be finite and positive")
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be finite and positive")
    if not math.isfinite(time_years) or time_years < 0:
        raise ValueError("time_years must be finite and non-negative")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and non-negative")
