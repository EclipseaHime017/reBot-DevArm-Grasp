"""ArUco 标记检测与位姿估计。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MarkerPose:
    """单个 ArUco 标记的检测结果。"""
    id: int
    T_marker2cam: np.ndarray   # (4, 4) 标记到相机坐标系的变换
    corners_px: Optional[np.ndarray] = None  # (4, 2) 像素角点

    @property
    def bbox_xyxy(self) -> Optional[tuple[int, int, int, int]]:
        if self.corners_px is None:
            return None
        pts = np.asarray(self.corners_px, dtype=np.float64).reshape(-1, 2)
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


class ArUcoDetector:
    """ArUco 标记检测器，返回最接近相机的目标标记的位姿。

    Args:
        marker_length_m: 标记实际边长（米，黑色边框外沿）
        aruco_dict_id:   cv2.aruco 字典 ID，默认 DICT_4X4_50 = 0
        target_marker_id: 指定只检测该 ID 的标记；None = 使用检测到的第一个
    """

    def __init__(
        self,
        marker_length_m: float = 0.05,
        aruco_dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        self._length = marker_length_m
        self._tid = target_marker_id
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def _estimate_pose_single_marker(
        self,
        corner: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Estimate one marker pose, supporting OpenCV builds without old ArUco helpers."""
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corner], self._length, K, D
            )
            if rvec is None or tvec is None:
                return None
            return (
                np.asarray(rvec[0], dtype=np.float64).reshape(3, 1),
                np.asarray(tvec[0], dtype=np.float64).reshape(3, 1),
            )

        image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
        half = float(self._length) / 2.0
        object_points = np.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(-1, 1),
            flags=flags,
        )
        if not ok and flags != cv2.SOLVEPNP_ITERATIVE:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                np.asarray(K, dtype=np.float64),
                np.asarray(D, dtype=np.float64).reshape(-1, 1),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None
        return (
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        )

    def detect(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Optional[MarkerPose]:
        """
        在图像中检测 ArUco 标记并返回位姿。

        Args:
            bgr: BGR 图像 (H, W, 3)
            K:   相机内参矩阵 (3, 3)
            D:   畸变系数 (1, N) 或 (N,)

        Returns:
            MarkerPose 或 None（未检测到）
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten()

        # 筛选目标 ID
        if self._tid is not None:
            mask = ids_flat == self._tid
            if not np.any(mask):
                return None
            corners = [corners[i] for i in np.where(mask)[0]]
            ids_flat = ids_flat[mask]

        # 多标记时取最近（Z 最小）的
        best: Optional[MarkerPose] = None
        best_z = float("inf")

        for corner, mid in zip(corners, ids_flat):
            pose = self._estimate_pose_single_marker(corner, K, D)
            if pose is None:
                continue
            rvec, tvec = pose
            z = float(tvec.reshape(3)[2])
            if z < best_z:
                best_z = z
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = tvec.reshape(3)
                best = MarkerPose(
                    id=int(mid),
                    T_marker2cam=T,
                    corners_px=np.asarray(corner, dtype=np.float64).reshape(4, 2),
                )

        return best

    def draw_detected(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        axis_length: float = 0.03,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的所有 ArUco 标记（框 + 坐标轴）。

        Returns:
            带标注的 BGR 图像副本
        """
        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return vis

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        ids_flat = ids.flatten()
        for corner, mid in zip(corners, ids_flat):
            pts = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            cv2.rectangle(
                vis,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"id={int(mid)} bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                (int(round(x1)), max(18, int(round(y1)) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            pose = self._estimate_pose_single_marker(corner, K, D)
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(vis, K, D, rvec, tvec, axis_length)

        return vis
