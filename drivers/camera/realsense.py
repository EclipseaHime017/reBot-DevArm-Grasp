"""Intel RealSense 相机驱动（D435i / D405 等）。"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver


class RealsenseCamera(CameraDriver):
    """Intel RealSense 深度相机驱动。

    Args:
        width, height, fps: 颜色流与深度流分辨率及帧率
        calib_dir: 标定目录路径；含 intrinsics.npz 时从中读取畸变系数
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
    ) -> None:
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None

        self._pipeline = None
        self._align = None
        self._depth_scale_mm: float = 1.0   # raw uint16 → mm 的乘数
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(f"未安装 pyrealsense2，请执行: pip install pyrealsense2: {e}") from e

        errors = []
        for width, height, fps in self._profile_candidates():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = pipeline.start(config)
                self._pipeline = pipeline
                self._align = rs.align(rs.stream.color)
                self._w, self._h, self._fps = width, height, fps
                break
            except RuntimeError as e:
                errors.append(f"{width}x{height}@{fps}: {e}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
        else:
            raise RuntimeError(
                "RealSense 相机未找到或无可用 RGB-D profile:\n  "
                + "\n  ".join(errors)
                + "\n  可能原因: 未插入 / USB 接口松动 / 被其他程序占用"
            )

        # 内参
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cx = getattr(intr, "ppx", getattr(intr, "cx", None))
        cy = getattr(intr, "ppy", getattr(intr, "cy", None))
        if cx is None or cy is None:
            raise RuntimeError(f"无法读取 RealSense 主点参数，intrinsics={intr!r}")
        self._K = np.array([
            [intr.fx, 0,        cx],
            [0,        intr.fy, cy],
            [0,        0,       1      ],
        ], dtype=np.float64)

        # 深度比例：raw uint16 * depth_scale (m/unit) * 1000 → mm
        ds = profile.get_device().first_depth_sensor().get_depth_scale()
        self._depth_scale_mm = ds * 1000.0

        self._D = self._load_distortion()
        print(f"[RealsenseCamera] 就绪 ({intr.width}×{intr.height}, "
              f"depth_scale={ds:.6f} m/unit)")

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ── 帧获取 ───────────────────────────────────────────────────────────────

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None:
            return None, None
        try:
            frames = self._pipeline.wait_for_frames(500)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                return None, None

            color_bgr = np.asanyarray(cf.get_data())
            depth_raw = np.asanyarray(df.get_data())   # uint16, depth units
            depth_mm  = (depth_raw * self._depth_scale_mm).astype(np.uint16)
            return color_bgr, depth_mm
        except Exception:
            return None, None

    # ── 内参 ─────────────────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("相机未打开，请先调用 open()")
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            raise RuntimeError("相机未打开，请先调用 open()")
        return self._D

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _load_distortion(self) -> np.ndarray:
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    if abs(D[0]) > 5.0:
                        print(f"[RealsenseCamera] 畸变系数 k1={D[0]:.2f} 偏大，改用零畸变")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        return np.zeros((1, 5), dtype=np.float64)

    def _profile_candidates(self) -> list[tuple[int, int, int]]:
        requested = (int(self._w), int(self._h), int(self._fps))
        candidates = [
            requested,
            (1280, 720, 15),
            (848, 480, 30),
            (640, 480, 30),
            (640, 480, 15),
        ]
        unique = []
        seen = set()
        for item in candidates:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique
