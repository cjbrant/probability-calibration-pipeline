from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:  # pragma: no cover - side effect only
    load_dotenv()

ENV_API_KEY = "THE_ODDS_API_KEY"
API_PLACEHOLDER = "YOUR_THE_ODDS_API_KEY"
CONFIG_FILENAME = "config.toml"
EXAMPLE_FILENAME = "config.example.toml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISSING_KEY_MSG = (
    "Missing THE_ODDS_API_KEY. Set it in your environment or config.toml "
    "(do not commit the real key)."
)


@dataclass
class AppConfig:
    api_key: str
    sport: str
    region: str
    markets: str
    odds_format: str
    sharp_books: List[str]
    target_books: Optional[List[str]] = None


def _load_toml_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser()
    if not resolved.exists():
        return {}
    with resolved.open("rb") as handle:
        return tomllib.load(handle)


def _get_scoped_value(
    section: str,
    key: str,
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Any:
    for source in (primary, fallback):
        scoped = source.get(section)
        if isinstance(scoped, dict) and key in scoped and scoped[key] not in (None, ""):
            return scoped[key]
    return None


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    result: List[str] = []
    for item in iterator:
        item_str = str(item).strip()
        if item_str:
            result.append(item_str)
    return result


def _sanitize_api_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    sanitized = value.strip()
    if not sanitized or sanitized.upper() == API_PLACEHOLDER.upper():
        return None
    return sanitized


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from environment variables, config.toml, or config.example.toml.
    Environment variables take precedence, followed by config.toml.
    """
    explicit_path = config_path if config_path is None else config_path.expanduser()
    config_data = _load_toml_if_exists(explicit_path or PROJECT_ROOT / CONFIG_FILENAME)
    example_data = _load_toml_if_exists(PROJECT_ROOT / EXAMPLE_FILENAME)

    api_key = _sanitize_api_key(os.getenv(ENV_API_KEY))
    if api_key is None:
        api_key = _sanitize_api_key(
            _get_scoped_value("api", "the_odds_api_key", config_data, example_data)
        )
    if api_key is None:
        raise RuntimeError(MISSING_KEY_MSG)

    sport = _get_scoped_value("fetch", "sport", config_data, example_data)
    region = _get_scoped_value("fetch", "region", config_data, example_data)
    markets = _get_scoped_value("fetch", "markets", config_data, example_data)
    odds_format = _get_scoped_value("fetch", "odds_format", config_data, example_data)
    sharp_books = _as_str_list(_get_scoped_value("sharp", "books", config_data, example_data))
    target_books = _as_str_list(_get_scoped_value("targets", "books", config_data, example_data))

    required = {
        "sport": sport,
        "region": region,
        "markets": markets,
        "odds_format": odds_format,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing config options: {', '.join(missing)}. "
            "Update config.toml or config.example.toml."
        )
    if not sharp_books:
        raise RuntimeError("Provide at least one entry under [sharp].books in the config.")

    return AppConfig(
        api_key=api_key,
        sport=str(sport),
        region=str(region),
        markets=str(markets),
        odds_format=str(odds_format),
        sharp_books=sharp_books,
        target_books=target_books or None,
    )


__all__ = ["AppConfig", "load_config"]
