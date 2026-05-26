"""相机驱动基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraDriver(ABC):
    """统一颜色 + 深度流接口。

    子类实现 open / close / get_frame / K / D。
    调用 setup_aruco() 后可用 detect_aruco() / draw_aruco() 便捷方法。
    """

    # ── 生命周期 ────────────────────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None:
        """初始化并打开相机流。"""

    @abstractmethod
    def close(self) -> None:
        """停止并释放相机资源。"""

    # ── 帧获取 ───────────────────────────────────────────────────────────────

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取一帧。

        Returns:
            (color_bgr, depth_mm)：
              color_bgr — uint8 BGR 图像；不可用时为 None。
              depth_mm  — uint16 numpy 数组，单位毫米；不可用时为 None。
        """

    # ── 内参 ─────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def K(self) -> np.ndarray:
        """相机内参矩阵 (3, 3) float64。"""

    @property
    @abstractmethod
    def D(self) -> np.ndarray:
        """畸变系数 (1, N) float64。"""

    # ── 便捷方法 ─────────────────────────────────────────────────────────────

    def warm_up(self, n_frames: int = 20) -> None:
        """丢弃前 n_frames 帧，等待曝光和白平衡稳定。"""
        for _ in range(n_frames):
            self.get_frame()

    def setup_aruco(
        self,
        marker_length_m: float,
        dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        """初始化 ArUco 检测器（使用本相机的内参/畸变）。

        Args:
            marker_length_m:  标记实际边长（米）
            dict_id:          cv2.aruco 字典 ID，默认 DICT_4X4_50 = 0
            target_marker_id: 只检测该 ID；None = 取距离最近的
        """
        from calibration.aruco_pose import ArUcoDetector
        self._aruco = ArUcoDetector(marker_length_m, dict_id, target_marker_id)

    def detect_aruco(self, bgr: np.ndarray):
        """检测 ArUco 标记，返回 MarkerPose 或 None（需先 setup_aruco）。"""
        return self._aruco.detect(bgr, self.K, self.D)

    def draw_aruco(self, bgr: np.ndarray) -> np.ndarray:
        """在图像上绘制检测到的所有 ArUco 标记（需先 setup_aruco）。"""
        return self._aruco.draw_detected(bgr, self.K, self.D)

    # ── 上下文管理器 ─────────────────────────────────────────────────────────

    def __enter__(self) -> "CameraDriver":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
