# -*- coding: utf-8 -*-
"""Utilities to download the demo assets required by PARE."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import gdown

from .path_utils import data_root


_DATA_URL = "https://drive.google.com/uc?id=1qIq0CBBj-O6wVc9nJXG-JDEtWPzRQ4KC"
_ARCHIVE_NAME = "pare-github-data.zip"


def _archive_path(root: Path) -> Path:
    return root.parent / _ARCHIVE_NAME


def _expected_paths(root: Path) -> Iterable[Path]:
    return (
        root / "pare/checkpoints/pare_w_3dpw_checkpoint.ckpt",
        root / "pare/checkpoints/pare_w_3dpw_config.yaml",
        root / "smpl_mean_params.npz",
    )


def _maybe_move_yolo_weights(root: Path) -> None:
    weights = root / "yolov3.weights"
    if not weights.exists():
        return

    target_dir = Path.home() / ".torch" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / weights.name

    if target_path.exists():
        # Already moved previously; keep the copy in ``data`` as well.
        return

    shutil.move(str(weights), str(target_path))


def download_data_assets(force: bool = False, *, move_yolo_weights: bool = True) -> Path:
    """Download demo assets bundled with the original repository.

    Parameters
    ----------
    force:
        When ``False`` (default) skip the download if the key checkpoint/config
        files already exist under the data directory.
    move_yolo_weights:
        When ``True`` (default) relocate ``yolov3.weights`` to the user's
        ``~/.torch/models`` directory after extraction to mirror the original
        setup script.

    Returns
    -------
    Path
        Path to the data directory containing the extracted assets.
    """

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    expected = tuple(_expected_paths(root))
    if not force and all(path.exists() for path in expected):
        if move_yolo_weights:
            _maybe_move_yolo_weights(root)
        # Ensure auxiliary directories exist.
        (root / "dataset_folders").mkdir(parents=True, exist_ok=True)
        return root

    archive_path = _archive_path(root)

    gdown.download(_DATA_URL, str(archive_path), quiet=False, fuzzy=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root.parent)

    archive_path.unlink(missing_ok=True)

    (root / "dataset_folders").mkdir(parents=True, exist_ok=True)

    if move_yolo_weights:
        _maybe_move_yolo_weights(root)

    return root


__all__ = ["download_data_assets"]
