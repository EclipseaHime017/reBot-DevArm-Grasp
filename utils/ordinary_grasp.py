"""Utilities for simple OBB/min-area-rect grasp estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .common_utils import detection_count, tensor_to_numpy
    from .transforms import grasp_axes_to_rebot_tcp_rotation
except ImportError:  # 支持从 cameraws/ 目录内以 ``utils.ordinary_grasp`` 或脚本路径导入
    from common_utils import detection_count, tensor_to_numpy
    from transforms import grasp_axes_to_rebot_tcp_rotation


@dataclass
class GraspPose:
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    center_px: tuple[int, int]
    position: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    tcp_rotation: Optional[np.ndarray]
    jaw_width_m: float
    object_length_m: float
    angle_deg: float
    rect_points: np.ndarray
    short_edge_points: np.ndarray
    valid_depth_pixels: int
    rejected_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.rejected_reason is None and self.position is not None and self.rotation is not None


def get_depth_mm(depth_map: np.ndarray, u: int, v: int, roi_size: int = 5) -> float:
    """Sample the median valid depth from a small window."""
    h, w = depth_map.shape
    half = roi_size // 2
    x1, x2 = max(0, u - half), min(w, u + half + 1)
    y1, y2 = max(0, v - half), min(h, v + half + 1)
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[roi > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def estimate_grasps(
    results: list[Any],
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
) -> list[GraspPose]:
    grasps: list[GraspPose] = []
    for result in results:
        for index in range(detection_count(result)):
            grasps.append(estimate_grasp(result, index, depth_mm, K, depth_quantile=depth_quantile))
    return grasps


def select_best_grasp(grasps: list[GraspPose]) -> Optional[GraspPose]:
    valid = [grasp for grasp in grasps if grasp.is_valid]
    if not valid:
        return None
    return max(valid, key=lambda grasp: grasp.conf)


def draw_grasp(image: np.ndarray, grasp: GraspPose) -> None:
    x1, y1, x2, y2 = grasp.bbox_xyxy
    color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    rect_pts = np.round(grasp.rect_points).astype(np.int32)
    cv2.polylines(image, [rect_pts], True, (255, 200, 0), 2, cv2.LINE_AA)

    p0, p1 = np.round(grasp.short_edge_points).astype(np.int32)
    cv2.line(image, tuple(p0), tuple(p1), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(image, grasp.center_px, 5, (0, 0, 255), -1)

    if grasp.is_valid:
        x_m, y_m, z_m = grasp.position.tolist()
        line1 = f"{grasp.class_name} {grasp.conf:.2f} jaw={grasp.jaw_width_m * 100:.1f}cm"
        line2 = f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f} ang:{grasp.angle_deg:.1f}"
    else:
        line1 = f"{grasp.class_name} {grasp.conf:.2f}"
        line2 = grasp.rejected_reason or "invalid"

    bg_w = max(len(line1), len(line2)) * 10
    cv2.rectangle(image, (x1, y1 - 42), (x1 + bg_w, y1), (0, 0, 0), -1)
    cv2.putText(image, line1, (x1 + 4, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(image, line2, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def estimate_grasp(
    result: Any,
    index: int,
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
) -> GraspPose:
    class_name, conf, bbox_xyxy = _detection_meta(result, index)
    rect_points = _rect_points(result, index, depth_mm.shape, bbox_xyxy)
    center = rect_points.mean(axis=0).astype(np.float32)

    mask = _depth_mask(result, index, depth_mm.shape, rect_points)
    short_vec_uv, short_len_px = _short_edge(rect_points)
    short_dir_uv = _normalize(short_vec_uv)
    edge_lengths = [float(np.linalg.norm(rect_points[(i + 1) % 4] - rect_points[i])) for i in range(4)]
    long_len_px = max(edge_lengths)
    grasp_span_px = short_len_px
    short_edge_points = _line_from_center(center, short_vec_uv)

    if short_dir_uv is not None:
        refined = _refine_grasp_line_from_mask(mask, center, short_dir_uv, long_len_px)
        if refined is not None:
            center, short_edge_points, grasp_span_px = refined

    center_px = (int(round(float(center[0]))), int(round(float(center[1]))))
    depth_values = depth_mm[mask > 0]
    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0:
        center_depth = get_depth_mm(depth_mm, center_px[0], center_px[1], 5)
        if center_depth > 0:
            depth_values = np.array([center_depth], dtype=np.float32)

    if len(depth_values) == 0 or short_dir_uv is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="no_valid_depth_or_rect",
        )

    depth_quantile = float(np.clip(depth_quantile, 0.0, 1.0))
    z_m = float(np.quantile(depth_values, depth_quantile) / 1000.0)
    position = _backproject(float(center[0]), float(center[1]), z_m, K)
    approach = _normalize(-position)
    if approach is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    open_axis = _pixel_vec_to_3d(short_dir_uv, z_m, K)
    open_axis = open_axis - float(np.dot(open_axis, approach)) * approach
    open_axis = _normalize(open_axis)
    if open_axis is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="open_axis_failed",
        )

    if open_axis[0] < 0:
        open_axis = -open_axis
    grip_axis = _normalize(np.cross(open_axis, approach))
    open_axis = _normalize(np.cross(approach, grip_axis))
    if grip_axis is None or open_axis is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="grasp_axis_failed",
        )

    rotation = np.column_stack([grip_axis, open_axis, approach]).astype(np.float32)
    tcp_rotation = grasp_axes_to_rebot_tcp_rotation(rotation[:, 0], rotation[:, 1], rotation[:, 2]).astype(np.float32)

    jaw_width_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * grasp_span_px, z_m, K)))
    object_length_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * long_len_px, z_m, K)))
    angle_deg = float(np.degrees(np.arctan2(short_dir_uv[1], short_dir_uv[0])))

    return GraspPose(
        class_name=class_name,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        center_px=center_px,
        position=position,
        rotation=rotation,
        tcp_rotation=tcp_rotation,
        jaw_width_m=jaw_width_m,
        object_length_m=object_length_m,
        angle_deg=angle_deg,
        rect_points=rect_points,
        short_edge_points=short_edge_points,
        valid_depth_pixels=int(len(depth_values)),
    )


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def _line_from_center(center: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.stack([center - 0.5 * vec, center + 0.5 * vec], axis=0).astype(np.float32)


def _refine_grasp_line_from_mask(
    mask: np.ndarray,
    center: np.ndarray,
    short_dir_uv: np.ndarray,
    long_len_px: float,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Use the mask's central cross-section to refine the short-axis grasp point.

    The short-axis direction still comes from the OBB/min-area-rect. We only
    replace the grasp center with the midpoint of the mask's actual thickness
    around the object's median longitudinal slice, which is more reliable for
    curved or asymmetric shapes such as bananas.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 32:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    grip_dir_uv = np.array([-short_dir_uv[1], short_dir_uv[0]], dtype=np.float32)
    rel = points - center.reshape(1, 2)
    grip_coord = rel @ grip_dir_uv
    open_coord = rel @ short_dir_uv

    grip_center = float(np.median(grip_coord))
    band_half_width_px = float(np.clip(long_len_px * 0.04, 2.0, 12.0))
    band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    if int(np.count_nonzero(band_mask)) < 24:
        band_half_width_px = float(np.clip(long_len_px * 0.08, 4.0, 18.0))
        band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    if int(np.count_nonzero(band_mask)) < 24:
        return None

    band_open = open_coord[band_mask]
    open_min = float(np.percentile(band_open, 5.0))
    open_max = float(np.percentile(band_open, 95.0))
    grasp_span_px = open_max - open_min
    if grasp_span_px < 2.0:
        return None

    open_center = 0.5 * (open_min + open_max)
    refined_center = center + grip_center * grip_dir_uv + open_center * short_dir_uv
    short_edge_points = _line_from_center(refined_center, short_dir_uv * grasp_span_px)
    return refined_center.astype(np.float32), short_edge_points, float(grasp_span_px)


def _safe_attr_row(container: Any, attr: str, index: int) -> Optional[np.ndarray]:
    values = getattr(container, attr, None)
    if values is None:
        return None
    try:
        return tensor_to_numpy(values[index])
    except Exception:
        return None


def _obb_points(result: Any, index: int) -> Optional[np.ndarray]:
    obb = getattr(result, "obb", None)
    if obb is None:
        return None
    points = _safe_attr_row(obb, "xyxyxyxy", index)
    if points is None:
        points = _safe_attr_row(obb, "xyxyxyxyn", index)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim == 1 and points.size == 8:
        points = points.reshape(4, 2)
    if points.shape != (4, 2):
        return None
    if float(np.max(np.abs(points))) <= 1.5:
        h, w = result.orig_shape[:2]
        points = points * np.array([w, h], dtype=np.float32)
    return points.astype(np.float32)


def _detection_meta(result: Any, index: int) -> tuple[str, float, tuple[int, int, int, int]]:
    names = getattr(result, "names", {})
    obb = getattr(result, "obb", None)
    if obb is not None:
        cls_row = _safe_attr_row(obb, "cls", index)
        conf_row = _safe_attr_row(obb, "conf", index)
        xyxy_row = _safe_attr_row(obb, "xyxy", index)
        if cls_row is not None and conf_row is not None:
            cls_id = int(np.asarray(cls_row).reshape(-1)[0])
            conf = float(np.asarray(conf_row).reshape(-1)[0])
            if xyxy_row is not None:
                x1, y1, x2, y2 = [int(v) for v in np.asarray(xyxy_row).reshape(-1)[:4]]
            else:
                points = _obb_points(result, index)
                min_xy = np.floor(points.min(axis=0)).astype(np.int32)
                max_xy = np.ceil(points.max(axis=0)).astype(np.int32)
                x1, y1, x2, y2 = int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])
            return names.get(cls_id, str(cls_id)), conf, (x1, y1, x2, y2)

    box = result.boxes[index]
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    return names.get(cls_id, str(cls_id)), conf, (x1, y1, x2, y2)


def _rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def _rect_points(
    result: Any,
    index: int,
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    points = _obb_points(result, index)
    if points is not None:
        return points

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is not None and boxes is not None and len(masks.data) == len(boxes):
        mask = masks.data[index].cpu().numpy()
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        rect = _rect_from_mask((mask > 0.5).astype(np.uint8))
        if rect is not None:
            return rect

    x1, y1, x2, y2 = bbox_xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _depth_mask(result: Any, index: int, image_shape: tuple[int, int], rect_points: np.ndarray) -> np.ndarray:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is not None and boxes is not None and len(masks.data) == len(boxes):
        mask = masks.data[index].cpu().numpy()
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0.5).astype(np.uint8)

    polygon = np.round(rect_points).astype(np.int32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _short_edge(rect_points: np.ndarray) -> tuple[np.ndarray, float]:
    best_vec = rect_points[1] - rect_points[0]
    best_len = float(np.linalg.norm(best_vec))
    for i in range(4):
        p0 = rect_points[i]
        p1 = rect_points[(i + 1) % 4]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < best_len:
            best_vec = vec
            best_len = length
    return best_vec.astype(np.float32), best_len


def _backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float32)


def _pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    fx, fy = max(float(K[0, 0]), 1e-6), max(float(K[1, 1]), 1e-6)
    return np.array([float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0], dtype=np.float32)
