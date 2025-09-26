"""High-level public API for the PARE demo package."""

from __future__ import annotations

from .pare.api.pare import run_pare
from .pare.utils.data_assets import download_data_assets

__all__ = ["run_pare", "download_data_assets"]
