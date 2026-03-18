from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl


def compute_sharp_consensus(
    rows: Iterable[dict] | pl.DataFrame,
    sharp_books: Sequence[str],
) -> pl.DataFrame:
    """
    Compute the median no-vig probability per outcome using selected books.
    """
    if not sharp_books:
        raise ValueError("Sharp book list is empty.")

    if isinstance(rows, pl.DataFrame):
        df = rows.clone()
    else:
        df = pl.DataFrame(rows)

    if df.is_empty():
        raise ValueError("No odds rows were provided.")

    sharp_lower = {book.lower() for book in sharp_books}

    sharp_df = (
        df.with_columns(pl.col("book").str.to_lowercase())
        .filter(pl.col("book").is_in(list(sharp_lower)))
    )

    if sharp_df.is_empty():
        raise ValueError("No odds available for the configured sharp books.")

    consensus = (
        sharp_df.group_by(["event_id", "outcome"])
        .agg(
            pl.first("home_team").alias("home_team"),
            pl.first("away_team").alias("away_team"),
            pl.first("commence_time").alias("commence_time"),
            pl.first("market").alias("market"),
            pl.median("p_novig").alias("p_prior"),
        )
        .sort(["event_id", "outcome"])
    )

    return consensus


__all__ = ["compute_sharp_consensus"]
