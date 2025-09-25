# -*- coding: utf-8 -*-
"""Helpers for locating packaged data assets regardless of install location."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable, Union


PathLike = Union[str, os.PathLike[str]]


@lru_cache(maxsize=None)
def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def _default_data_root() -> Path:
    """Best-effort discovery of the package's bundled ``data`` directory."""

    env_root = os.getenv("PARE_DATA_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser()
        if env_path.exists():
            return env_path

    try:
        data_resource = resources.files("pare").joinpath("data")
    except ModuleNotFoundError:
        data_resource = None
    else:
        with resources.as_file(data_resource) as data_path:
            if data_path.exists():
                return data_path

    package_root = _package_root()
    candidates = (
        package_root / "data",
        package_root.parent / "data",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return package_root / "data"


def data_root() -> Path:
    """Return the directory that stores packaged demo/checkpoint artefacts."""

    return _default_data_root()


def resolve_data_path(*parts: PathLike) -> Path:
    """Resolve a path relative to the discovered ``data`` directory.

    ``parts`` may contain Path objects or strings. Individual string arguments
    can include path separators; when the first portion equals ``"data"`` it is
    stripped to make invocations such as ``resolve_data_path("data/foo")`` work
    transparently for older call sites.
    """

    if len(parts) == 1:
        candidate = Path(parts[0])
    else:
        candidate = Path(*parts)

    if candidate.is_absolute():
        return candidate

    parts_iter: Iterable[str] = candidate.parts
    if parts_iter and parts_iter[0] == "data":
        candidate = Path(*parts_iter[1:])

    resolved = data_root() / candidate
    return resolved


def resolve_asset_path(path: PathLike) -> Path:
    """Resolve any path pointing at a packaged asset.

    This mirrors :func:`resolve_data_path` but falls back to returning the
    original path when no better location is found so legacy callers can still
    surface a helpful ``FileNotFoundError`` message.
    """

    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    from_root = resolve_data_path(candidate)
    if from_root.exists():
        return from_root

    # Some callers expect to work relative to their current working directory.
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return from_root

