#!/usr/bin/env python3
"""Verify that pyorbbecsdk can open an Orbbec camera and stream RGB-D frames.

Usage:
  conda activate graspnet
  cd /home/seeed/Downloads/rebot_grasp
  python scripts/verify_pyorbbec_stream.py
  python scripts/verify_pyorbbec_stream.py --preview --seconds 10
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import cv2
import numpy as np


def _format_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _profile_summary(profile: Any) -> str:
    try:
        video = profile.as_video_stream_profile()
    except Exception:
        video = profile
    parts = []
    for label, getter in (
        ("width", "get_width"),
        ("height", "get_height"),
        ("fps", "get_fps"),
        ("format", "get_format"),
    ):
        try:
            value = getattr(video, getter)()
            parts.append(f"{label}={_format_name(value)}")
        except Exception:
            pass
    return ", ".join(parts) or str(profile)


def _select_profile(profile_list: Any, width: int, height: int, fmt: Any, fps: int) -> Any:
    if width > 0 and height > 0 and fps > 0:
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception as exc:
            print(f"[WARN] requested profile unavailable: {exc}")
    return profile_list.get_default_video_stream_profile()


def _color_to_bgr(frame: Any, ob_format: Any) -> Optional[np.ndarray]:
    width, height = frame.get_width(), frame.get_height()
    data = np.asanyarray(frame.get_data(), dtype=np.uint8)
    fmt = frame.get_format()

    try:
        if fmt == ob_format.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob_format.RGB:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if hasattr(ob_format, "BGR") and fmt == ob_format.BGR:
            return data.reshape(height, width, 3)
        if hasattr(ob_format, "YUYV") and fmt == ob_format.YUYV:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV)
        if hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
        if hasattr(ob_format, "NV12") and fmt == ob_format.NV12:
            return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
        if hasattr(ob_format, "NV21") and fmt == ob_format.NV21:
            return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    except Exception as exc:
        print(f"[WARN] color conversion failed: {exc}")
        return None

    print(f"[WARN] unsupported color format: {_format_name(fmt)}")
    return None


def _depth_to_mm(frame: Any, ob_format: Any) -> Optional[np.ndarray]:
    if frame.get_format() != ob_format.Y16:
        print(f"[WARN] unsupported depth format: {_format_name(frame.get_format())}")
        return None

    width, height = frame.get_width(), frame.get_height()
    depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width)
    scale = float(frame.get_depth_scale()) if hasattr(frame, "get_depth_scale") else 1.0
    if scale != 1.0:
        depth = (depth.astype(np.float32) * scale).astype(np.uint16)
    return depth


def _depth_stats(depth_mm: np.ndarray) -> str:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return "valid=0"
    center = depth_mm[depth_mm.shape[0] // 2, depth_mm.shape[1] // 2]
    return (
        f"valid={valid.size} center={int(center)}mm "
        f"min={int(valid.min())}mm median={float(np.median(valid)):.1f}mm max={int(valid.max())}mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify pyorbbecsdk RGB-D streaming.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--preview", action="store_true", help="show color/depth preview windows")
    parser.add_argument("--no-align", action="store_true", help="do not request hardware depth-to-color alignment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
    except ImportError as exc:
        print(f"[FAIL] pyorbbecsdk import failed: {exc}")
        return 2

    try:
        from pyorbbecsdk import OBAlignMode
    except Exception:
        OBAlignMode = None

    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    color_profile = None
    for fmt in (OBFormat.MJPG, OBFormat.RGB):
        try:
            color_profile = _select_profile(color_profiles, args.width, args.height, fmt, args.fps)
            break
        except Exception:
            color_profile = None
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()

    depth_profile = _select_profile(depth_profiles, args.width, args.height, OBFormat.Y16, args.fps)

    print(f"[INFO] color profile: {_profile_summary(color_profile)}")
    print(f"[INFO] depth profile: {_profile_summary(depth_profile)}")
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    if not args.no_align and OBAlignMode is not None:
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            print("[INFO] align mode: HW_MODE")
        except Exception as exc:
            print(f"[WARN] could not enable HW alignment: {exc}")

    color_count = 0
    depth_count = 0
    both_count = 0
    t0 = time.monotonic()
    last_print = 0.0

    try:
        pipeline.start(config)
        print("[INFO] pipeline started")

        while time.monotonic() - t0 < args.seconds:
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                print("[WARN] wait_for_frames timed out")
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_bgr = _color_to_bgr(color_frame, OBFormat) if color_frame is not None else None
            depth_mm = _depth_to_mm(depth_frame, OBFormat) if depth_frame is not None else None

            if color_bgr is not None:
                color_count += 1
            if depth_mm is not None:
                depth_count += 1
            if color_bgr is not None and depth_mm is not None:
                both_count += 1

            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                color_shape = None if color_bgr is None else color_bgr.shape
                depth_shape = None if depth_mm is None else depth_mm.shape
                stats = "no depth" if depth_mm is None else _depth_stats(depth_mm)
                elapsed = max(now - t0, 1e-6)
                print(
                    f"[FRAME] color={color_shape} depth={depth_shape} "
                    f"rgbd_fps={both_count / elapsed:.1f} {stats}"
                )

            if args.preview:
                if color_bgr is not None:
                    cv2.imshow("Orbbec color", color_bgr)
                if depth_mm is not None:
                    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imshow("Orbbec depth", depth_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except Exception as exc:
        print(f"[FAIL] stream failed: {exc}")
        return 1
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print(f"[RESULT] color_frames={color_count} depth_frames={depth_count} rgbd_frames={both_count}")
    if both_count == 0:
        print("[FAIL] no synchronized color+depth frames received")
        return 1
    print("[OK] pyorbbecsdk stream is working")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

