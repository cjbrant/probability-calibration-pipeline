from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import orjson
import polars as pl

from .consensus import compute_sharp_consensus
from .conversion import (
    american_to_decimal,
    decimal_to_american,
    implied_prob,
    no_vig_two_way,
)


def _convert_to_decimal(price: float, odds_format: str) -> float:
    fmt = odds_format.lower()
    if fmt == "american":
        return american_to_decimal(float(price))
    if fmt == "decimal":
        value = float(price)
        if value <= 1:
            raise ValueError("Decimal odds must be greater than 1.")
        return value
    raise ValueError(f"Unsupported odds format '{odds_format}'.")


def _convert_prices(value: float, odds_format: str) -> tuple[float, float]:
    decimal_value = _convert_to_decimal(value, odds_format)
    if odds_format.lower() == "american":
        american_value = float(value)
    else:
        american_value = decimal_to_american(decimal_value)
    return decimal_value, american_value


def read_snapshot_events(snapshot_path: str | Path) -> List[dict]:
    path = Path(snapshot_path)
    data = orjson.loads(path.read_bytes())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"{path} does not contain a JSON array of events.")


def flatten_rows(
    events: Iterable[dict],
    *,
    market: str,
    odds_format: str,
) -> pl.DataFrame:
    rows: List[dict] = []
    for event in events or []:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        for bookmaker in event.get("bookmakers") or []:
            book_key = bookmaker.get("key")
            if not book_key:
                continue
            for market_data in bookmaker.get("markets") or []:
                if market_data.get("key") != market:
                    continue
                outcomes = market_data.get("outcomes") or []
                if len(outcomes) != 2:
                    continue
                decimal_odds: List[float] = []
                american_odds: List[float] = []
                try:
                    for outcome in outcomes:
                        decimal_value, american_value = _convert_prices(
                            outcome["price"], odds_format
                        )
                        decimal_odds.append(decimal_value)
                        american_odds.append(american_value)
                except (KeyError, ValueError, TypeError):
                    continue

                try:
                    implied = [implied_prob(value) for value in decimal_odds]
                    no_vig = no_vig_two_way(implied[0], implied[1])
                except ValueError:
                    continue

                for idx, outcome_data in enumerate(outcomes):
                    name = outcome_data.get("name")
                    if not name:
                        continue
                    rows.append(
                        {
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "book": book_key,
                            "market": market,
                            "outcome": name,
                            "odds_decimal": decimal_odds[idx],
                            "odds_american": american_odds[idx],
                            "p_novig": no_vig[idx],
                        }
                    )
    return pl.DataFrame(rows)


def flatten_snapshot(
    events: Iterable[dict],
    *,
    market: str,
    odds_format: str,
    sharp_books: Sequence[str],
) -> pl.DataFrame:
    rows = flatten_rows(events, market=market, odds_format=odds_format)
    if rows.is_empty():
        raise ValueError("No odds rows available after filtering snapshot.")
    consensus = compute_sharp_consensus(rows, sharp_books)
    return consensus.select(
        [
            "event_id",
            "outcome",
            "p_prior",
            "commence_time",
            "home_team",
            "away_team",
        ]
    )


def build_training(
    snapshot_files: List[str],
    *,
    results_csv: str,
    market: str,
    odds_format: str,
    sharp_books: Sequence[str],
) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []
    for snapshot_path in snapshot_files:
        data = read_snapshot_events(snapshot_path)
        frame = flatten_snapshot(
            data,
            market=market,
            odds_format=odds_format,
            sharp_books=sharp_books,
        ).select(["event_id", "outcome", "p_prior"])
        frames.append(frame)

    if not frames:
        raise ValueError("No snapshot files were provided.")

    all_priors = pl.concat(frames, how="vertical_relaxed")

    results = pl.read_csv(results_csv).select(["event_id", "winner_outcome"])
    dataset = (
        all_priors.join(results, on="event_id", how="inner")
        .with_columns(
            (pl.col("outcome") == pl.col("winner_outcome"))
            .cast(pl.Int64)
            .alias("y")
        )
        .select(["event_id", "outcome", "p_prior", "y"])
    )
    if dataset.is_empty():
        raise ValueError("Joined dataset is empty; ensure results cover the snapshots.")
    return dataset


__all__ = ["read_snapshot_events", "flatten_rows", "flatten_snapshot", "build_training"]
