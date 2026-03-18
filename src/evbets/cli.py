from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import orjson
import polars as pl
import typer

from . import VERSION
from .calibration import (
    BBQCalibrator,
    BetaCalibratorLib,
    evaluate_probs,
    load_calibrator,
)
from .config import AppConfig, load_config
from .conversion import american_to_decimal, implied_prob, no_vig_two_way
from .consensus import compute_sharp_consensus
from .dataset import build_training, flatten_rows, flatten_snapshot, read_snapshot_events
from .odds_fetch import fetch_odds
from .paths import get_data_dir, get_reports_dir

app = typer.Typer(help="EV betting toolkit", no_args_is_help=True)


def _read_events(path: Path) -> List[Dict[str, Any]]:
    try:
        return read_snapshot_events(path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to read odds file {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _ensure_books(option: str, books: Sequence[str]) -> List[str]:
    filtered = [book for book in books if book]
    if not filtered:
        typer.echo(f"Provide at least one entry for {option}.", err=True)
        raise typer.Exit(code=1)
    return filtered


def _extract_possible_outcomes(event: Dict[str, Any], market: str) -> List[str] | None:
    for bookmaker in event.get("bookmakers") or []:
        for market_data in bookmaker.get("markets") or []:
            if market_data.get("key") != market:
                continue
            outcomes = market_data.get("outcomes") or []
            if len(outcomes) != 2:
                continue
            names = [outcome.get("name") for outcome in outcomes]
            if all(names):
                return [str(name) for name in names]
    return None


def _load_config_for_defaults() -> tuple[AppConfig | None, str | None]:
    try:
        return load_config(), None
    except RuntimeError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to load config: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _prepare_probs(priors: pl.DataFrame, calibrated_csv: Path | None) -> pl.DataFrame:
    frame = priors.select(["event_id", "outcome", "p_prior"])
    if calibrated_csv is None:
        return frame.with_columns(pl.col("p_prior").alias("p_use"))

    try:
        calibrated = pl.read_csv(calibrated_csv)
    except Exception as exc:
        typer.echo(f"Failed to read calibrated probabilities: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if "p_true_calibrated" not in calibrated.columns:
        typer.echo("Calibrated CSV missing 'p_true_calibrated' column.", err=True)
        raise typer.Exit(code=1)

    calibrated = calibrated.rename({"p_true_calibrated": "p_cal"})
    merged = frame.join(calibrated, on=["event_id", "outcome"], how="left")
    return (
        merged.with_columns(pl.coalesce([pl.col("p_cal"), pl.col("p_prior")]).alias("p_use"))
        .drop("p_cal")
    )


def _build_ev_table(
    offers: pl.DataFrame,
    probs: pl.DataFrame,
    target_books: Sequence[str],
    ev_threshold: float,
) -> pl.DataFrame:
    if offers.is_empty():
        return pl.DataFrame()

    target_lower = {book.lower() for book in target_books}
    filtered = (
        offers.with_columns(pl.col("book").str.to_lowercase().alias("book_lower"))
        .filter(pl.col("book_lower").is_in(list(target_lower)))
        .drop("book_lower")
    )
    if filtered.is_empty():
        return pl.DataFrame()

    prob_subset = probs.select(["event_id", "outcome", "p_use"])
    joined = filtered.join(prob_subset, on=["event_id", "outcome"], how="inner")
    if joined.is_empty():
        return pl.DataFrame()

    enriched = joined.with_columns(
        [
            (pl.col("away_team") + pl.lit(" @ ") + pl.col("home_team")).alias("match"),
            pl.col("commence_time").alias("start"),
            (pl.col("p_use") * (pl.col("odds_decimal") - 1.0) - (1.0 - pl.col("p_use"))).alias("ev"),
        ]
    )

    positive = enriched.filter(pl.col("ev") >= ev_threshold)
    if positive.is_empty():
        return positive.select(
            [
                "match",
                "start",
                "book",
                "outcome",
                "odds_american",
                "odds_decimal",
                "p_use",
                "ev",
                "event_id",
            ]
        )

    return (
        positive.select(
            [
                "match",
                "start",
                "book",
                "outcome",
                "odds_american",
                "odds_decimal",
                "p_use",
                "ev",
                "event_id",
            ]
        )
        .sort("ev", descending=True)
        .with_columns(pl.col("ev").round(4), pl.col("p_use").round(4))
    )


def _betting_summary(ev_df: pl.DataFrame, results_df: pl.DataFrame, stake: float = 100.0) -> dict:
    summary = {"n_bets": int(ev_df.height), "precision": None, "roi": None}
    if ev_df.is_empty():
        return summary

    joined = ev_df.join(results_df, on="event_id", how="left")
    resolved = joined.filter(pl.col("winner_outcome").is_not_null())
    if resolved.is_empty():
        return summary

    resolved = resolved.with_columns(
        (pl.col("outcome") == pl.col("winner_outcome")).alias("won"),
    )
    win_rate = (
        resolved.select(pl.col("won").cast(pl.Float64).mean().alias("precision"))["precision"][0]
    )
    profits = resolved.with_columns(
        pl.when(pl.col("won"))
        .then((pl.col("odds_decimal") - 1.0) * stake)
        .otherwise(-stake)
        .alias("profit")
    )
    roi = profits["profit"].sum() / (resolved.height * stake)
    summary["precision"] = float(win_rate) if win_rate is not None else None
    summary["roi"] = float(roi)
    return summary


def _ev_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("ev") < 0.02)
        .then("0.00-0.02")
        .when(pl.col("ev") < 0.04)
        .then("0.02-0.04")
        .when(pl.col("ev") < 0.08)
        .then("0.04-0.08")
        .otherwise("0.08+")
        .alias("ev_bucket")
    )


def _bucket_report(resolved: pl.DataFrame, stake: float) -> pl.DataFrame:
    if resolved.is_empty():
        return pl.DataFrame(
            {
                "ev_bucket": pl.Series([], dtype=pl.Utf8),
                "n_bets": pl.Series([], dtype=pl.Int64),
                "expected_ev": pl.Series([], dtype=pl.Float64),
                "profit": pl.Series([], dtype=pl.Float64),
                "stake": pl.Series([], dtype=pl.Float64),
                "realized_roi": pl.Series([], dtype=pl.Float64),
            }
        )

    with_buckets = resolved.with_columns(_ev_bucket_expr())
    return (
        with_buckets.group_by("ev_bucket")
        .agg(
            pl.count().alias("n_bets"),
            pl.col("ev").mean().alias("expected_ev"),
            pl.col("profit").sum().alias("profit"),
        )
        .with_columns(
            (pl.col("n_bets") * pl.lit(stake)).alias("stake"),
            pl.when(pl.col("n_bets") > 0)
            .then(pl.col("profit") / (pl.col("n_bets") * pl.lit(stake)))
            .otherwise(0.0)
            .alias("realized_roi"),
        )
        .sort("ev_bucket")
    )


def _book_report(resolved: pl.DataFrame, stake: float) -> pl.DataFrame:
    if resolved.is_empty():
        return pl.DataFrame(
            {
                "book": pl.Series([], dtype=pl.Utf8),
                "n_bets": pl.Series([], dtype=pl.Int64),
                "profit": pl.Series([], dtype=pl.Float64),
                "stake": pl.Series([], dtype=pl.Float64),
                "roi": pl.Series([], dtype=pl.Float64),
                "hit_rate": pl.Series([], dtype=pl.Float64),
                "avg_ev": pl.Series([], dtype=pl.Float64),
            }
        )

    return (
        resolved.group_by("book")
        .agg(
            pl.count().alias("n_bets"),
            pl.col("profit").sum().alias("profit"),
            pl.col("won").cast(pl.Float64).mean().alias("hit_rate"),
            pl.col("ev").mean().alias("avg_ev"),
        )
        .with_columns(
            (pl.col("n_bets") * pl.lit(stake)).alias("stake"),
            pl.when(pl.col("n_bets") > 0)
            .then(pl.col("profit") / (pl.col("n_bets") * pl.lit(stake)))
            .otherwise(0.0)
            .alias("roi"),
        )
        .select(["book", "n_bets", "profit", "stake", "roi", "hit_rate", "avg_ev"])
        .sort("book")
    )


def _prepare_results_frame(results_path: Path) -> pl.DataFrame:
    try:
        df = pl.read_csv(results_path)
    except Exception as exc:
        raise ValueError(f"Failed to read results CSV: {exc}") from exc

    if "event_id" not in df.columns or "winner_outcome" not in df.columns:
        raise ValueError("Results CSV must include 'event_id' and 'winner_outcome' columns.")

    return (
        df.select(["event_id", "winner_outcome"])
        .with_columns(
            pl.col("event_id").cast(pl.Utf8),
            pl.when(pl.col("winner_outcome").is_null())
            .then(None)
            .otherwise(pl.col("winner_outcome").cast(pl.Utf8).str.strip_chars())
            .alias("winner_outcome"),
        )
        .with_columns(
            pl.when(pl.col("winner_outcome") == "").then(None).otherwise(pl.col("winner_outcome"))
        )
        .unique("event_id")
    )


def _select_ev_bets(
    offers: pl.DataFrame,
    priors: pl.DataFrame,
    target_books: Sequence[str],
    ev_threshold: float,
    stake: float,
    snapshot_label: str,
) -> pl.DataFrame:
    if offers.is_empty():
        return pl.DataFrame()

    target_lower = {book.lower() for book in target_books}
    filtered = (
        offers.with_columns(pl.col("book").str.to_lowercase().alias("book_lower"))
        .filter(pl.col("book_lower").is_in(list(target_lower)))
        .drop("book_lower")
    )
    if filtered.is_empty():
        return pl.DataFrame()

    prob_subset = priors.select(
        [
            "event_id",
            "outcome",
            "p_prior",
            "p_star",
            "commence_time",
            "home_team",
            "away_team",
        ]
    )
    joined = filtered.join(prob_subset, on=["event_id", "outcome"], how="inner")
    if joined.is_empty():
        return pl.DataFrame()

    enriched = joined.with_columns(
        [
            pl.col("event_id").cast(pl.Utf8),
            (pl.col("away_team") + pl.lit(" @ ") + pl.col("home_team")).alias("match"),
            pl.col("commence_time").alias("start"),
            (
                pl.col("p_star") * (pl.col("odds_decimal") - 1.0)
                - (1.0 - pl.col("p_star"))
            ).alias("ev"),
            pl.lit(snapshot_label).alias("snapshot"),
            pl.lit(ev_threshold).alias("ev_threshold"),
            pl.lit(stake).alias("stake"),
        ]
    )

    positive = enriched.filter(pl.col("ev") >= ev_threshold)
    if positive.is_empty():
        return pl.DataFrame()

    return positive.select(
        [
            "snapshot",
            "event_id",
            "match",
            "start",
            "book",
            "outcome",
            "odds_american",
            "odds_decimal",
            "p_prior",
            "p_star",
            "ev",
            "stake",
            "ev_threshold",
            "home_team",
            "away_team",
        ]
    ).with_columns((pl.col("ev") * pl.col("stake")).alias("expected_profit"))


# TODO: consolidate this flattening with dataset.flatten_rows to avoid duplication.
def _flatten_odds(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        bookmakers = event.get("bookmakers") or []

        for bookmaker in bookmakers:
            book_key = bookmaker.get("key")
            if not book_key:
                continue
            markets = bookmaker.get("markets") or []
            for market in markets:
                market_key = market.get("key", "unknown")
                outcomes = market.get("outcomes") or []
                if len(outcomes) != 2:
                    continue

                try:
                    decimal_odds = [
                        american_to_decimal(float(outcome["price"])) for outcome in outcomes
                    ]
                except (TypeError, ValueError, KeyError):
                    continue

                try:
                    implied = [implied_prob(value) for value in decimal_odds]
                    no_vig = no_vig_two_way(implied[0], implied[1])
                except ValueError:
                    continue

                if not all(outcome.get("name") for outcome in outcomes):
                    continue

                for outcome_data, decimal_value, p_novig in zip(outcomes, decimal_odds, no_vig):
                    rows.append(
                        {
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "book": book_key,
                            "market": market_key,
                            "outcome": outcome_data["name"],
                            "decimal_odds": decimal_value,
                            "p_novig": p_novig,
                        }
                    )

    return rows


@app.command("fetch-snapshot")
def fetch_snapshot(
    sport: str | None = typer.Option(None, "--sport", help="Sport key (defaults to config)."),
    region: str | None = typer.Option(None, "--region", help="Region code (defaults to config)."),
    markets: str | None = typer.Option(None, "--markets", help="Markets (defaults to config)."),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Destination JSON path."),
) -> None:
    """Fetch raw odds snapshot and write to JSON."""
    try:
        config: AppConfig = load_config()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to load config: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    sport_val = sport or config.sport
    region_val = region or config.region
    markets_val = markets or config.markets
    odds_fmt_val = odds_format or config.odds_format

    async def _run() -> List[Dict[str, Any]]:
        return await fetch_odds(
            api_key=config.api_key,
            sport=sport_val,
            region=region_val,
            markets=markets_val,
            odds_format=odds_fmt_val,
        )

    try:
        events = asyncio.run(_run())
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to fetch odds: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if out is not None:
        dest = out
    else:
        snapshots_dir = get_data_dir() / "snapshots"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = snapshots_dir / f"{sport_val}_{timestamp}.json"

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.write_bytes(orjson.dumps(events, option=orjson.OPT_INDENT_2))
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to write snapshot: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote snapshot with {len(events)} events to {dest}")


@app.command("fetch-consensus")
def fetch_consensus(out: Path = typer.Option(Path("consensus.csv"), "--out", "-o")) -> None:
    """Fetch odds, compute sharp consensus priors, and write them to CSV."""
    try:
        config: AppConfig = load_config()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to load config: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    sharp_books = config.sharp_books

    async def _run() -> List[Dict[str, Any]]:
        return await fetch_odds(
            api_key=config.api_key,
            sport=config.sport,
            region=config.region,
            markets=config.markets,
            odds_format=config.odds_format,
        )

    try:
        events = asyncio.run(_run())
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to fetch odds: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    rows = _flatten_odds(events)
    if not rows:
        typer.echo("No valid odds rows found in API response.", err=True)
        raise typer.Exit(code=1)

    df = pl.DataFrame(rows)

    try:
        consensus = compute_sharp_consensus(df, sharp_books)
    except Exception as exc:
        typer.echo(f"Failed to compute consensus: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    consensus.write_csv(out)
    typer.echo(f"Wrote {consensus.height} consensus rows to {out}")


@app.command("init-results")
def init_results(
    snapshot: Path = typer.Argument(..., help="Snapshot JSON file."),
    out: Path = typer.Option(
        Path("data/results/results_template.csv"), "--out", "-o", help="Destination CSV."
    ),
    market: str | None = typer.Option(
        None, "--market", help="Market key, e.g. h2h (defaults to config)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file."),
) -> None:
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")

    events = _read_events(snapshot)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for event in events:
        possible = _extract_possible_outcomes(event, final_market)
        if possible is None:
            event_id = event.get("id")
            missing.append(str(event_id) if event_id is not None else "<unknown>")
            continue
        rows.append(
            {
                "event_id": event.get("id"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "possible_outcomes": "|".join(possible),
                "winner_outcome": "",
            }
        )

    if missing:
        typer.echo(
            "Missing market '"
            + str(final_market)
            + "' with two outcomes for events: "
            + ", ".join(missing),
            err=True,
        )
        raise typer.Exit(code=1)

    if not rows:
        typer.echo("No events found in snapshot.", err=True)
        raise typer.Exit(code=1)

    if out.exists() and not force:
        typer.echo(
            f"Refusing to overwrite existing file: {out}. Use --force to replace.", err=True
        )
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "event_id",
                    "home_team",
                    "away_team",
                    "possible_outcomes",
                    "winner_outcome",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Failed to write results template: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote results template with {len(rows)} events to {out}")


@app.command("build-dataset")
def build_dataset(
    snapshot_files: List[Path] = typer.Argument(..., help="Snapshot JSON files."),
    results: Path = typer.Option(..., "--results", help="CSV with event_id,winner_outcome."),
    out: Path = typer.Option(Path("training.parquet"), "--out", "-o"),
    market: str | None = typer.Option(
        None, "--market", help="Market key, e.g. h2h (defaults to config)."
    ),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    sharp_books: List[str] | None = typer.Option(
        None, "--sharp-books", "-s", help="Sharp books (defaults to config)."
    ),
) -> None:
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)
    final_odds_format = odds_format if odds_format is not None else (
        config.odds_format if config else None
    )
    final_sharp_books = sharp_books if sharp_books else (config.sharp_books if config else [])

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")
    if not final_odds_format:
        detail = "Provide --odds-format or set fetch.odds_format in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="odds_format")

    books = _ensure_books("--sharp-books or config [sharp].books", final_sharp_books)
    if not snapshot_files:
        typer.echo("Provide at least one snapshot file.", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = build_training(
            [str(path) for path in snapshot_files],
            results_csv=str(results),
            market=final_market,
            odds_format=final_odds_format,
            sharp_books=books,
        )
    except Exception as exc:
        typer.echo(f"Failed to build dataset: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(out)
    typer.echo(f"Wrote training dataset with {dataset.height} rows to {out}")


@app.command("fit-calibrator")
def fit_calibrator(
    data: Path = typer.Option(Path("training.parquet"), "--data"),
    method: str = typer.Option("beta", "--method", help="beta or bbq"),
    model: Path = typer.Option(Path("calibrator.joblib"), "--model", "-m"),
) -> None:
    try:
        df = pl.read_parquet(data)
    except Exception as exc:
        typer.echo(f"Failed to load training data: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if df.is_empty():
        typer.echo("Training data is empty.", err=True)
        raise typer.Exit(code=1)

    if "y" not in df.columns:
        typer.echo("Training data must include a 'y' column.", err=True)
        raise typer.Exit(code=1)

    df = df.drop_nulls(["y"])
    if df.is_empty():
        typer.echo("Training data has no labeled rows after dropping null targets.", err=True)
        raise typer.Exit(code=1)

    probs = df["p_prior"].to_numpy()
    labels = df["y"].to_numpy()

    method_lower = method.lower()
    if method_lower == "beta":
        try:
            calibrator = BetaCalibratorLib()
        except ImportError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    elif method_lower == "bbq":
        calibrator = BBQCalibrator()
    else:
        typer.echo("Method must be 'beta' or 'bbq'.", err=True)
        raise typer.Exit(code=1)

    calibrator.fit(probs, labels)
    in_sample = evaluate_probs(calibrator.predict(probs), labels)
    typer.echo(orjson.dumps(in_sample, option=orjson.OPT_INDENT_2).decode())

    model.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(model)
    typer.echo(f"Saved {method_lower} calibrator to {model}")


@app.command("predict")
def predict(
    input_path: Path = typer.Option(..., "--in", help="Snapshot JSON."),
    model_path: Path = typer.Option(Path("calibrator.joblib"), "--model", "-m"),
    out: Path = typer.Option(Path("calibrated.csv"), "--out", "-o"),
    market: str | None = typer.Option(None, "--market", help="Market key (defaults to config)."),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    sharp_books: List[str] | None = typer.Option(
        None, "--sharp-books", "-s", help="Sharp books (defaults to config)."
    ),
) -> None:
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)
    final_odds_format = odds_format if odds_format is not None else (
        config.odds_format if config else None
    )
    final_sharp_books = sharp_books if sharp_books else (config.sharp_books if config else [])

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")
    if not final_odds_format:
        detail = "Provide --odds-format or set fetch.odds_format in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="odds_format")

    books = _ensure_books("--sharp-books or config [sharp].books", final_sharp_books)
    events = _read_events(input_path)
    try:
        priors = flatten_snapshot(
            events,
            market=final_market,
            odds_format=final_odds_format,
            sharp_books=books,
        )
    except Exception as exc:
        typer.echo(f"Failed to compute priors: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        calibrator = load_calibrator(model_path)
    except Exception as exc:
        typer.echo(f"Failed to load calibrator: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    preds = calibrator.predict(priors["p_prior"].to_numpy())
    result = priors.select(["event_id", "outcome"]).with_columns(
        pl.Series("p_true_calibrated", preds)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out)
    typer.echo(f"Wrote calibrated probabilities for {result.height} rows to {out}")


@app.command("scan")
def scan(
    input_path: Path = typer.Option(..., "--in", help="Snapshot JSON."),
    calibrated_csv: Path | None = typer.Option(None, "--use-calibrated"),
    ev_threshold: float = typer.Option(0.02, "--ev-threshold"),
    out: Path = typer.Option(Path("positive_ev.csv"), "--out", "-o"),
    market: str | None = typer.Option(None, "--market", help="Market key (defaults to config)."),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    sharp_books: List[str] | None = typer.Option(
        None, "--sharp-books", "-s", help="Sharp books (defaults to config)."
    ),
    target_books: List[str] | None = typer.Option(
        None, "--target-books", "-t", help="Target books (defaults to config)."
    ),
    log: bool = typer.Option(True, "--log/--no-log", help="Append EV results to a log file."),
    log_path: Path | None = typer.Option(
        None, "--log-path", help="Custom path for EV log (defaults to reports/bets_log.csv)."
    ),
) -> None:
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)
    final_odds_format = odds_format if odds_format is not None else (
        config.odds_format if config else None
    )
    final_sharp_books = sharp_books if sharp_books else (config.sharp_books if config else [])
    final_target_books = target_books if target_books else (
        config.target_books if config and config.target_books else []
    )

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")
    if not final_odds_format:
        detail = "Provide --odds-format or set fetch.odds_format in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="odds_format")
    if not final_sharp_books:
        detail = "Provide --sharp-books or set sharp.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="sharp_books")
    if not final_target_books:
        detail = "Provide --target-books or set targets.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="target_books")

    books = _ensure_books("--sharp-books or config [sharp].books", final_sharp_books)
    targets = _ensure_books("--target-books or config [targets].books", final_target_books)
    events = _read_events(input_path)
    try:
        priors = flatten_snapshot(
            events,
            market=final_market,
            odds_format=final_odds_format,
            sharp_books=books,
        )
        offers = flatten_rows(events, market=final_market, odds_format=final_odds_format)
    except Exception as exc:
        typer.echo(f"Failed to prepare snapshot: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    probs = _prepare_probs(priors, calibrated_csv)
    ev_df = _build_ev_table(offers, probs, targets, ev_threshold)

    out.parent.mkdir(parents=True, exist_ok=True)
    ev_df.write_csv(out)
    if ev_df.is_empty():
        typer.echo("No wagers cleared the EV threshold.")
    else:
        typer.echo(str(ev_df))
        typer.echo(f"Wrote {ev_df.height} positive EV wagers to {out}")

    if log and not ev_df.is_empty():
        model_name = "calibrated" if calibrated_csv is not None else "prior"
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        log_dest = log_path or (get_reports_dir() / "bets_log.csv")
        entries = (
            ev_df.with_columns(
                [
                    pl.lit(timestamp).alias("timestamp"),
                    pl.lit(ev_threshold).alias("ev_threshold"),
                    pl.lit(model_name).alias("model_name"),
                    pl.lit(100.0).alias("stake"),
                ]
            )
            .select(
                [
                    "timestamp",
                    "event_id",
                    "match",
                    "start",
                    "book",
                    "outcome",
                    "odds_american",
                    "odds_decimal",
                    "p_use",
                    "ev",
                    "ev_threshold",
                    "model_name",
                    "stake",
                ]
            )
        )
        log_dest.parent.mkdir(parents=True, exist_ok=True)
        include_header = not log_dest.exists()
        try:
            with log_dest.open("a", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "event_id",
                        "match",
                        "start",
                        "book",
                        "outcome",
                        "odds_american",
                        "odds_decimal",
                        "p_use",
                        "ev",
                        "ev_threshold",
                        "model_name",
                        "stake",
                    ],
                )
                if include_header:
                    writer.writeheader()
                for row in entries.to_dicts():
                    writer.writerow(row)
        except Exception as exc:  # pragma: no cover
            typer.echo(f"Warning: failed to append EV log: {exc}", err=True)


@app.command("backtest")
def backtest(
    snapshot_files: List[Path] = typer.Argument(..., help="Snapshot JSON files."),
    results: Path = typer.Option(..., "--results", help="CSV with event_id,winner_outcome."),
    model: Path = typer.Option(..., "--model", "-m", help="Trained calibrator joblib."),
    ev_threshold: float = typer.Option(0.02, "--ev-threshold"),
    out: Path = typer.Option(Path("reports/backtest"), "--out", "-o", help="Output directory."),
    market: str | None = typer.Option(None, "--market", help="Market key (defaults to config)."),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    sharp_books: List[str] | None = typer.Option(
        None, "--sharp-books", "-s", help="Sharp books (defaults to config)."
    ),
    target_books: List[str] | None = typer.Option(
        None, "--target-books", "-t", help="Target books (defaults to config)."
    ),
) -> None:
    """
    Evaluate the EV pipeline on historical snapshots and compute ROI metrics.
    """
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)
    final_odds_format = odds_format if odds_format is not None else (
        config.odds_format if config else None
    )
    final_sharp_books = sharp_books if sharp_books else (config.sharp_books if config else [])
    final_target_books = target_books if target_books else (
        config.target_books if config and config.target_books else []
    )

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")
    if not final_odds_format:
        detail = "Provide --odds-format or set fetch.odds_format in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="odds_format")
    if not final_sharp_books:
        detail = "Provide --sharp-books or set sharp.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="sharp_books")
    if not final_target_books:
        detail = "Provide --target-books or set targets.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="target_books")
    if not snapshot_files:
        typer.echo("Provide at least one snapshot file.", err=True)
        raise typer.Exit(code=1)

    try:
        results_df = _prepare_results_frame(results)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    try:
        calibrator = load_calibrator(model)
    except Exception as exc:
        typer.echo(f"Failed to load calibrator: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    stake = 100.0
    books = _ensure_books("--sharp-books or config [sharp].books", final_sharp_books)
    targets = _ensure_books("--target-books or config [targets].books", final_target_books)

    bet_frames: List[pl.DataFrame] = []
    for snapshot in snapshot_files:
        events = _read_events(snapshot)
        try:
            offers = flatten_rows(
                events,
                market=final_market,
                odds_format=final_odds_format,
            )
        except Exception as exc:
            typer.echo(f"Failed to process snapshot {snapshot}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        if offers.is_empty():
            typer.echo(f"No odds rows found in {snapshot}, skipping.", err=True)
            continue

        try:
            priors = compute_sharp_consensus(offers, books)
        except Exception as exc:
            typer.echo(f"Failed to compute priors for {snapshot}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        try:
            preds = calibrator.predict(priors["p_prior"].to_numpy())
        except Exception as exc:
            typer.echo(f"Failed to score priors for {snapshot}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        priors = priors.with_columns(pl.Series("p_star", preds))
        bets = _select_ev_bets(
            offers,
            priors,
            targets,
            ev_threshold,
            stake,
            str(snapshot),
        )
        if not bets.is_empty():
            bet_frames.append(bets)

    if bet_frames:
        all_bets = pl.concat(bet_frames, how="vertical_relaxed")
    else:
        all_bets = pl.DataFrame(
            {
                "snapshot": pl.Series([], dtype=pl.Utf8),
                "event_id": pl.Series([], dtype=pl.Utf8),
                "match": pl.Series([], dtype=pl.Utf8),
                "start": pl.Series([], dtype=pl.Utf8),
                "book": pl.Series([], dtype=pl.Utf8),
                "outcome": pl.Series([], dtype=pl.Utf8),
                "odds_american": pl.Series([], dtype=pl.Float64),
                "odds_decimal": pl.Series([], dtype=pl.Float64),
                "p_prior": pl.Series([], dtype=pl.Float64),
                "p_star": pl.Series([], dtype=pl.Float64),
                "ev": pl.Series([], dtype=pl.Float64),
                "stake": pl.Series([], dtype=pl.Float64),
                "ev_threshold": pl.Series([], dtype=pl.Float64),
                "home_team": pl.Series([], dtype=pl.Utf8),
                "away_team": pl.Series([], dtype=pl.Utf8),
                "expected_profit": pl.Series([], dtype=pl.Float64),
            }
        )

    with_results = all_bets.join(results_df, on="event_id", how="left").with_columns(
        [
            pl.when(pl.col("winner_outcome").is_null())
            .then(None)
            .otherwise(pl.col("outcome") == pl.col("winner_outcome"))
            .alias("won"),
            pl.when(pl.col("winner_outcome").is_null())
            .then(None)
            .otherwise(
                pl.when(pl.col("outcome") == pl.col("winner_outcome"))
                .then((pl.col("odds_decimal") - 1.0) * pl.col("stake"))
                .otherwise(-pl.col("stake"))
            )
            .alias("profit"),
        ]
    )

    resolved = with_results.filter(pl.col("profit").is_not_null())
    total_bets = int(resolved.height)
    total_profit = float(resolved["profit"].sum()) if total_bets else 0.0
    total_stake = float(total_bets) * stake
    winners = int(resolved["won"].sum()) if total_bets else 0
    hit_rate = (winners / total_bets) if total_bets else 0.0
    roi = (total_profit / total_stake) if total_stake else 0.0

    bucket_df = _bucket_report(resolved, stake)
    book_df = _book_report(resolved, stake)

    summary = {
        "snapshots": len(snapshot_files),
        "total_bets": total_bets,
        "total_profit": total_profit,
        "total_stake": total_stake,
        "roi": roi,
        "hit_rate": hit_rate,
        "ev_threshold": ev_threshold,
        "stake": stake,
        "unresolved_bets": int(with_results.height - resolved.height),
    }

    out_dir = out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    buckets_path = out_dir / "ev_buckets.csv"
    books_path = out_dir / "by_book.csv"
    bets_path = out_dir / "all_bets.csv"

    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    bucket_df.write_csv(buckets_path)
    book_df.write_csv(books_path)
    with_results.write_csv(bets_path)

    typer.echo(f"Wrote summary to {summary_path}")
    typer.echo(f"Wrote EV bucket analysis to {buckets_path}")
    typer.echo(f"Wrote per-book ROI to {books_path}")
    typer.echo(f"Wrote all bets log to {bets_path}")


@app.command("bakeoff")
def bakeoff(
    train_data: Path = typer.Option(..., "--train-data"),
    test_snapshots: List[Path] = typer.Argument(..., help="Holdout snapshots."),
    results: Path = typer.Option(..., "--results", help="Holdout results CSV."),
    ev_threshold: float = typer.Option(0.02, "--ev-threshold"),
    market: str | None = typer.Option(None, "--market", help="Market key (defaults to config)."),
    odds_format: str | None = typer.Option(
        None, "--odds-format", help="Odds format (defaults to config)."
    ),
    sharp_books: List[str] | None = typer.Option(
        None, "--sharp-books", "-s", help="Sharp books (defaults to config)."
    ),
    target_books: List[str] | None = typer.Option(
        None, "--target-books", "-t", help="Target books (defaults to config)."
    ),
) -> None:
    config, config_error = _load_config_for_defaults()
    final_market = market if market is not None else (config.markets if config else None)
    final_odds_format = odds_format if odds_format is not None else (
        config.odds_format if config else None
    )
    final_sharp_books = sharp_books if sharp_books else (config.sharp_books if config else [])
    final_target_books = target_books if target_books else (
        config.target_books if config and config.target_books else []
    )

    if not final_market:
        detail = "Provide --market or set fetch.markets in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="market")
    if not final_odds_format:
        detail = "Provide --odds-format or set fetch.odds_format in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="odds_format")
    if not final_sharp_books:
        detail = "Provide --sharp-books or set sharp.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="sharp_books")
    if not final_target_books:
        detail = "Provide --target-books or set targets.books in config.toml."
        if config_error:
            detail = f"{detail} ({config_error})"
        raise typer.BadParameter(detail, param_name="target_books")

    books = _ensure_books("--sharp-books or config [sharp].books", final_sharp_books)
    targets = _ensure_books("--target-books or config [targets].books", final_target_books)

    if not test_snapshots:
        typer.echo("Provide at least one holdout snapshot.", err=True)
        raise typer.Exit(code=1)

    try:
        train_df = pl.read_parquet(train_data)
    except Exception as exc:
        typer.echo(f"Failed to load train data: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if train_df.is_empty():
        typer.echo("Training data is empty.", err=True)
        raise typer.Exit(code=1)

    try:
        holdout = build_training(
            [str(path) for path in test_snapshots],
            results_csv=str(results),
            market=final_market,
            odds_format=final_odds_format,
            sharp_books=books,
        )
    except Exception as exc:
        typer.echo(f"Failed to build holdout dataset: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    results_df = pl.read_csv(results).select(["event_id", "winner_outcome"]).unique("event_id")

    p_train = train_df["p_prior"].to_numpy()
    y_train = train_df["y"].to_numpy()
    p_holdout = holdout["p_prior"].to_numpy()
    y_holdout = holdout["y"].to_numpy()

    beta_cal: BetaCalibratorLib | None = None
    beta_metrics: dict | None = None
    beta_status = ""
    try:
        beta_cal = BetaCalibratorLib().fit(p_train, y_train)
        beta_metrics = evaluate_probs(beta_cal.predict(p_holdout), y_holdout)
    except ImportError as exc:
        beta_status = str(exc)

    bbq_cal = BBQCalibrator().fit(p_train, y_train)
    bbq_metrics = evaluate_probs(bbq_cal.predict(p_holdout), y_holdout)

    last_snapshot = _read_events(test_snapshots[-1])
    try:
        last_rows = flatten_rows(last_snapshot, market=final_market, odds_format=final_odds_format)
        last_priors = flatten_snapshot(
            last_snapshot,
            market=final_market,
            odds_format=final_odds_format,
            sharp_books=books,
        )
    except Exception as exc:
        typer.echo(f"Failed to prepare last snapshot: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    beta_ev = pl.DataFrame()
    if beta_cal is not None:
        beta_probs = last_priors.select(["event_id", "outcome", "p_prior"]).with_columns(
            pl.Series("p_use", beta_cal.predict(last_priors["p_prior"].to_numpy()))
        )
        beta_ev = _build_ev_table(last_rows, beta_probs, targets, ev_threshold)

    bbq_probs = last_priors.select(["event_id", "outcome", "p_prior"]).with_columns(
        pl.Series("p_use", bbq_cal.predict(last_priors["p_prior"].to_numpy()))
    )
    bbq_ev = _build_ev_table(last_rows, bbq_probs, targets, ev_threshold)

    typer.echo("Holdout calibration metrics:")
    if beta_metrics:
        typer.echo("beta -> " + orjson.dumps(beta_metrics).decode())
    elif beta_status:
        typer.echo(f"beta -> unavailable ({beta_status})")
    else:
        typer.echo("beta -> unavailable")
    typer.echo("bbq  -> " + orjson.dumps(bbq_metrics).decode())

    typer.echo("Decision metrics on latest snapshot:")
    if beta_cal is not None:
        beta_stats = _betting_summary(beta_ev, results_df)
        typer.echo(
            f"beta : bets={beta_stats['n_bets']} precision={beta_stats['precision']} roi={beta_stats['roi']}"
        )
    else:
        typer.echo("beta : unavailable")

    bbq_stats = _betting_summary(bbq_ev, results_df)
    typer.echo(
        f"bbq  : bets={bbq_stats['n_bets']} precision={bbq_stats['precision']} roi={bbq_stats['roi']}"
    )


@app.command("version")
def version() -> None:
    """Show the package version."""
    typer.echo(VERSION)
