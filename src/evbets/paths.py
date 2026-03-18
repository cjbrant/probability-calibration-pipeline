from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """
    Return the project's data directory, creating standard subfolders.
    """
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "results", "training"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """
    Return the directory for persisted models.
    """
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_reports_dir() -> Path:
    """
    Return the directory for reports and logs.
    """
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


__all__ = ["get_data_dir", "get_models_dir", "get_reports_dir", "PROJECT_ROOT"]
