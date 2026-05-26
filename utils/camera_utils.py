"""Shared camera/config helpers for cameraws scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    from ..drivers.camera import CameraDriver, make_camera
except ImportError:  # 支持从 cameraws/ 目录内以 ``utils.camera_utils`` 形式导入
    from drivers.camera import CameraDriver, make_camera


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hand_eye(project_root: str | Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    hand_eye_path = Path(project_root) / "config" / "calibration" / str(cam_type).lower() / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    return T, mode


def configure_camera(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> dict[str, Any]:
    cam_cfg = cfg.setdefault("camera", {})
    camera_type = getattr(args, "camera_type", None)
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    fps = getattr(args, "fps", None)

    if camera_type is not None:
        cam_cfg["type"] = camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    if width is not None:
        cam_cfg["color_width"] = int(width)
        cam_cfg["depth_width"] = int(width)
    if height is not None:
        cam_cfg["color_height"] = int(height)
        cam_cfg["depth_height"] = int(height)
    if fps is not None:
        cam_cfg["fps"] = int(fps)
    elif camera_type is not None and realsense_default_fps is not None and "realsense" in cam_type:
        cam_cfg["fps"] = int(realsense_default_fps)
    return cfg


def create_camera_from_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> CameraDriver:
    return make_camera(configure_camera(cfg, args, realsense_default_fps=realsense_default_fps))
