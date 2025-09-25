# -*- coding: utf-8 -*-
"""Notebook-friendly API for running the PARE demo."""

from __future__ import annotations

import argparse
import gc
import io
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import joblib
from loguru import logger

from pare.core.tester import PARETester
from pare.utils.demo_utils import (
    download_youtube_clip,
    images_to_video,
    video_to_images,
)
from pare.utils.path_utils import resolve_asset_path, resolve_data_path

# Ensure EGL is used when available, matching the CLI demo behaviour.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


_CONSOLE_ENABLED: ContextVar[bool] = ContextVar("pare_console_enabled", default=True)
_CONSOLE_PATCHED = False


def _console_filter(record: Dict[str, Any]) -> bool:
    """Allow console logs only when the current context enables them."""

    try:
        return _CONSOLE_ENABLED.get()
    except LookupError:  # pragma: no cover - fallback for exotic contexts.
        return True


if not _CONSOLE_PATCHED:
    # Replace the default stderr sink with one honouring the context flag.
    try:
        logger.remove(0)
    except (ValueError, KeyError):  # Default sink already removed/customised.
        pass

    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| <level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        filter=_console_filter,
    )
    _CONSOLE_PATCHED = True

CFG = resolve_data_path("pare", "checkpoints", "pare_w_3dpw_config.yaml")
CKPT = resolve_data_path("pare", "checkpoints", "pare_w_3dpw_checkpoint.ckpt")

__all__ = ["run_pare"]


@contextmanager
def _silence_output(verbose: bool):
    """Mute stdout/stderr when ``verbose`` is ``False``."""

    if verbose:
        yield
        return

    buffer = io.StringIO()
    token = _CONSOLE_ENABLED.set(False)
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            yield
    finally:
        _CONSOLE_ENABLED.reset(token)
def _first_image_shape(folder: Path) -> Tuple[int, int, int]:
    """Return the shape of the first image inside ``folder``."""

    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        matches = sorted(folder.glob(pattern))
        if matches:
            image = cv2.imread(str(matches[0]))
            if image is None:
                break
            return image.shape
    raise FileNotFoundError(
        f"No readable images found in {folder} to infer dimensions."
    )


def run_pare(
    mode: str = "video",
    *,
    cfg: Union[str, Path] = CFG,
    ckpt: Union[str, Path] = CKPT,
    exp: str = "",
    vid_file: Optional[Union[str, Path]] = None,
    image_folder: Optional[Union[str, Path]] = None,
    output_folder: Union[str, Path] = "logs/demo/demo_results",
    tracking_method: str = "bbox",
    detector: str = "yolo",
    yolo_img_size: int = 416,
    tracker_batch_size: int = 12,
    staf_dir: Union[str, Path] = "/home/mkocabas/developments/openposetrack",
    batch_size: int = 16,
    display: bool = False,
    smooth: bool = False,
    min_cutoff: float = 0.004,
    beta: float = 1.0,
    no_render: bool = False,
    no_save: bool = False,
    wireframe: bool = False,
    sideview: bool = False,
    draw_keypoints: bool = False,
    save_obj: bool = False,
    smplify: bool = False,
    device: str = "auto",
    verbose: bool = False,
    return_outputs: bool = False,
    **extra_args: Any,
) -> Dict[str, Any]:
    """Execute the PARE demo logic programmatically.

    Parameters
    ----------
    mode:
        One of ``{"video", "folder"}``.
    cfg, ckpt:
        Paths to the config file and checkpoint used to initialise the model.
    exp:
        Short description appended to output artefacts.
    vid_file:
        Path or URL to the input video when ``mode="video"``.
    image_folder:
        Folder containing input images when ``mode="folder"``.
    output_folder:
        Directory where demo artefacts will be stored.
    verbose:
        When ``True`` enable INFO logs on stdout/stderr. When ``False``
        (default) stdout/stderr from the demo run are silenced to avoid noisy
        notebooks.
    return_outputs:
        When ``True`` include ``pare_results`` and ``tracking_results`` in the
        returned dictionary. Defaults to ``False`` to minimise host and GPU
        memory usage; the outputs are still written to disk unless ``no_save``
        is ``True``.
    extra_args:
        Additional keyword arguments forwarded to ``PARETester``.

    Returns
    -------
    Dict[str, Any]
        Paths to generated artefacts and, for video runs, in-memory results.
    """

    mode = mode.lower()
    if mode not in {"video", "folder"}:
        raise ValueError("mode must be either 'video' or 'folder'.")

    cfg_path = resolve_asset_path(cfg)
    ckpt_path = resolve_asset_path(ckpt)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    vid_path = str(vid_file) if vid_file is not None else None
    image_path = str(image_folder) if image_folder is not None else None

    namespace_kwargs = {
        "cfg": str(cfg_path),
        "ckpt": str(ckpt_path),
        "exp": exp,
        "mode": mode,
        "vid_file": vid_path,
        "image_folder": image_path,
        "output_folder": str(output_root),
        "tracking_method": tracking_method,
        "detector": detector,
        "yolo_img_size": yolo_img_size,
        "tracker_batch_size": tracker_batch_size,
        "staf_dir": str(staf_dir),
        "batch_size": batch_size,
        "display": display,
        "smooth": smooth,
        "min_cutoff": min_cutoff,
        "beta": beta,
        "no_render": no_render,
        "no_save": no_save,
        "wireframe": wireframe,
        "sideview": sideview,
        "draw_keypoints": draw_keypoints,
        "save_obj": save_obj,
        "smplify": smplify,
        "device": device,
    }
    namespace_kwargs.update(extra_args)
    args = argparse.Namespace(**namespace_kwargs)

    if mode == "video" and args.vid_file is None:
        raise ValueError("vid_file must be provided when mode='video'.")
    if mode == "folder" and args.image_folder is None:
        raise ValueError("image_folder must be provided when mode='folder'.")

    results: Dict[str, Any] = {"mode": mode}

    with _silence_output(verbose):
        if mode == "video":
            results.update(
                _run_video_demo(args, output_root, return_outputs)
            )
        else:
            results.update(
                _run_folder_demo(args, output_root, return_outputs)
            )

    return results


def _run_video_demo(
    args: argparse.Namespace,
    output_root: Path,
    return_outputs: bool,
) -> Dict[str, Any]:
    """Run the video demo path and return collected artefacts."""

    video_file = args.vid_file
    assert video_file is not None

    if video_file.startswith("https://www.youtube.com"):
        logger.info('Downloading YouTube video "{}"', video_file)
        downloaded = download_youtube_clip(video_file, "/tmp")
        if downloaded is None:
            raise RuntimeError("Failed to download YouTube video.")
        video_file = downloaded
        args.vid_file = video_file
        logger.info("YouTube video downloaded to {}", video_file)

    video_path = Path(video_file)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    suffix = f"_{args.exp}" if args.exp else "_"
    output_path = output_root / f"{video_path.stem}{suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger_id = logger.add(
        str(output_path / "demo.log"),
        level="INFO",
        colorize=False,
    )

    try:
        tmp_images = output_path / "tmp_images"
        if tmp_images.is_dir():
            input_image_folder = tmp_images
            logger.info('Frames already extracted in "{}"', input_image_folder)
            image_files = sorted(
                p for p in tmp_images.iterdir() if p.is_file()
            )
            num_frames = len(image_files)
            img_shape = _first_image_shape(tmp_images)
        else:
            img_folder, num_frames, img_shape = video_to_images(
                str(video_path),
                img_folder=str(tmp_images),
                return_info=True,
            )
            input_image_folder = Path(img_folder)

        output_img_folder = Path(f"{input_image_folder}_output")
        output_img_folder.mkdir(parents=True, exist_ok=True)

        args.image_folder = str(input_image_folder)

        tester = PARETester(args)

        total_start = time.time()
        logger.info("Input video number of frames {}", num_frames)
        orig_height, orig_width = img_shape[:2]

        tracking_results = tester.run_tracking(
            str(video_path),
            str(input_image_folder),
        )

        pare_start = time.time()
        pare_results = tester.run_on_video(
            tracking_results,
            str(input_image_folder),
            orig_width,
            orig_height,
        )
        pare_end = time.time()

        pare_time = max(pare_end - pare_start, 1e-8)
        fps = num_frames / pare_time

        del tester.model

        logger.info("PARE FPS: {:.2f}", fps)
        total_time = time.time() - total_start
        logger.info(
            "Total time spent: {:.2f} seconds (including model loading time).",
            total_time,
        )
        total_fps = num_frames / max(total_time, 1e-8)
        logger.info(
            "Total FPS (including model loading time): {:.2f}.",
            total_fps,
        )

        results_file = output_path / "pare_output.pkl"
        if not args.no_save:
            logger.info('Saving output results to "{}".', results_file)
            joblib.dump(pare_results, results_file)

        rendered_video = None
        if not args.no_render:
            tester.render_results(
                pare_results,
                str(input_image_folder),
                str(output_img_folder),
                str(output_path),
                orig_width,
                orig_height,
                num_frames,
            )
            vid_name = video_path.name
            save_name = output_path / (
                f"{video_path.stem}_{args.exp}_result.mp4"
            )
            logger.info("Saving result video to {}", save_name)
            images_to_video(
                img_folder=str(output_img_folder),
                output_vid_file=str(save_name),
            )
            images_to_video(
                img_folder=str(input_image_folder),
                output_vid_file=str(output_path / vid_name),
            )
            rendered_video = save_name

        results: Dict[str, Any] = {
            "output_path": str(output_path),
            "num_frames": num_frames,
            "image_shape": img_shape,
            "results_file": str(results_file) if not args.no_save else None,
            "rendered_video": str(rendered_video) if rendered_video else None,
            "fps": fps,
            "total_time": total_time,
        }

        if return_outputs:
            results["pare_results"] = pare_results
            results["tracking_results"] = tracking_results

        if not return_outputs:
            del pare_results
            del tracking_results
            gc.collect()

        return results
    finally:
        logger.remove(logger_id)


def _run_folder_demo(
    args: argparse.Namespace,
    output_root: Path,
    return_outputs: bool,
) -> Dict[str, Any]:
    """Run the folder demo path and return collected artefacts."""

    image_folder = args.image_folder
    assert image_folder is not None
    image_path = Path(image_folder)
    if not image_path.is_dir():
        raise FileNotFoundError(
            f"Input image folder does not exist: {image_path}"
        )

    args.tracker_batch_size = 1

    suffix = f"_{args.exp}" if args.exp else ""
    output_path = output_root / f"{image_path.name}{suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger_id = logger.add(
        str(output_path / "demo.log"),
        level="INFO",
        colorize=False,
    )

    try:
        output_img_folder = output_path / "pare_results"
        output_img_folder.mkdir(parents=True, exist_ok=True)

        num_frames = len([p for p in image_path.iterdir() if p.is_file()])
        logger.info("Number of input frames {}", num_frames)

        tester = PARETester(args)

        total_start = time.time()
        detections = tester.run_detector(str(image_path))
        pare_start = time.time()
        tester.run_on_image_folder(
            str(image_path),
            detections,
            str(output_path),
            str(output_img_folder),
            run_smplify=args.smplify,
        )
        pare_end = time.time()

        pare_time = max(pare_end - pare_start, 1e-8)
        fps = num_frames / pare_time

        del tester.model

        logger.info("PARE FPS: {:.2f}", fps)
        total_time = time.time() - total_start
        logger.info(
            "Total time spent: {:.2f} seconds (including model loading time).",
            total_time,
        )
        total_fps = num_frames / max(total_time, 1e-8)
        logger.info(
            "Total FPS (including model loading time): {:.2f}.",
            total_fps,
        )

        results: Dict[str, Any] = {
            "output_path": str(output_path),
            "num_frames": num_frames,
            "fps": fps,
            "total_time": total_time,
        }

        if return_outputs:
            results["detections"] = detections

        if not return_outputs:
            del detections
            gc.collect()

        return results
    finally:
        logger.remove(logger_id)
