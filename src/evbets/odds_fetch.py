from __future__ import annotations

from typing import Any, Dict, List

import httpx
import orjson

ODDS_ENDPOINT = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"


async def fetch_odds(
    *,
    api_key: str,
    sport: str,
    region: str,
    markets: str,
    odds_format: str,
    date_format: str = "iso",
    timeout: float = 10.0,
) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    url = ODDS_ENDPOINT.format(sport=sport)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
    return orjson.loads(response.content)


__all__ = ["fetch_odds"]
