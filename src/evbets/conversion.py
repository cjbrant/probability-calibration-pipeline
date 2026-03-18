from __future__ import annotations

from typing import Tuple


def american_to_decimal(american: float) -> float:
    if american == 0:
        raise ValueError("American odds cannot be zero.")
    if american > 0:
        return 1 + (american / 100.0)
    return 1 + (100.0 / abs(american))


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    if decimal_odds >= 2:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / decimal_odds


def no_vig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    total = p1 + p2
    if total == 0:
        raise ValueError("Total implied probability cannot be zero.")
    return p1 / total, p2 / total


__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "implied_prob",
    "no_vig_two_way",
]
