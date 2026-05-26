"""YOLO model loading and raw detection parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .common_utils import class_name, clip_bbox, detection_count, tensor_to_numpy
except ImportError:  # 支持从 cameraws/ 目录内以 ``utils.yolo_utils`` 形式导入
    from common_utils import class_name, clip_bbox, detection_count, tensor_to_numpy


@dataclass
class YoloDetection:
    result_index: int
    detection_index: int
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray


# Backward-compatible alias for older scripts. New code should use YoloDetection.
DetectionTarget = YoloDetection


def resolve_yolo_model_path(model_name: str, project_root: Path) -> Path:
    model_path = Path(str(model_name)).expanduser()
    if model_path.is_absolute():
        return model_path
    if len(model_path.parts) > 1:
        return project_root / model_path
    return project_root / "models" / model_path


def load_yolo(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    no_yolo: bool = False,
    model_override: Optional[str] = None,
    device_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    iou_override: Optional[float] = None,
    infer_every_override: Optional[int] = None,
) -> tuple[Optional[Any], dict[str, Any]]:
    gp_cfg = cfg.get("grasp_pipeline", {})
    yolo_opts: dict[str, Any] = {
        "enabled": not no_yolo,
        "infer_every": max(1, int(infer_every_override or gp_cfg.get("infer_every_live", 3))),
    }
    if no_yolo:
        return None, yolo_opts

    from ultralytics import YOLO

    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    model_name = str(model_override or yolo_cfg.get("model_name", "yoloe-26s-seg.pt"))
    model_path = resolve_yolo_model_path(model_name, project_root)
    device = device_override or yolo_cfg.get("device", "cpu")
    conf = float(conf_override if conf_override is not None else det_cfg.get("conf_threshold", 0.25))
    iou = float(iou_override if iou_override is not None else det_cfg.get("iou_threshold", 0.45))
    custom_classes = list(yolo_cfg.get("custom_classes", []))
    use_world = bool(yolo_cfg.get("use_world", True))

    print(f"Loading YOLO target detector: {model_path}")
    model = YOLO(str(model_path))
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()) and custom_classes:
        model.set_classes(custom_classes)
        print(f"YOLO open-vocabulary classes: {custom_classes}")

    yolo_opts.update(
        {
            "model_name": model_name,
            "device": device,
            "conf": conf,
            "iou": iou,
            "custom_classes": custom_classes,
        }
    )
    return model, yolo_opts


def detection_mask(result: Any, index: int, image_shape: tuple[int, int], bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image_shape
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is not None:
        try:
            if len(data) > index:
                mask = tensor_to_numpy(data[index])
                if mask is not None:
                    mask = np.asarray(mask, dtype=np.float32)
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    return (mask > 0.5).astype(np.uint8)
        except Exception:
            pass

    x1, y1, x2, y2 = bbox_xyxy
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def obb_points(result: Any, index: int, image_shape: tuple[int, int]) -> Optional[np.ndarray]:
    obb = getattr(result, "obb", None)
    if obb is None:
        return None
    points = None
    for attr in ("xyxyxyxy", "xyxyxyxyn"):
        values = getattr(obb, attr, None)
        if values is None:
            continue
        try:
            points = tensor_to_numpy(values[index])
        except Exception:
            points = None
        if points is not None:
            break
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
        h, w = image_shape
        points = points * np.array([w, h], dtype=np.float32)
    return points.astype(np.float32)


def obb_detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    names = getattr(result, "names", {})
    obb = getattr(result, "obb", None)
    if obb is None:
        raise ValueError("YOLO result has no OBB detections")

    cls_row = tensor_to_numpy(getattr(obb, "cls")[index])
    conf_row = tensor_to_numpy(getattr(obb, "conf")[index])
    cls_id = int(np.asarray(cls_row).reshape(-1)[0])
    conf = float(np.asarray(conf_row).reshape(-1)[0])

    xyxy = getattr(obb, "xyxy", None)
    if xyxy is not None:
        bbox = clip_bbox(np.asarray(tensor_to_numpy(xyxy[index])).reshape(-1), image_shape)
    else:
        points = obb_points(result, index, image_shape)
        if points is None:
            raise ValueError("YOLO OBB result has neither xyxy nor polygon points")
        bbox = clip_bbox(np.concatenate([points.min(axis=0), points.max(axis=0)]), image_shape)
    return class_name(names, cls_id), conf, bbox


def box_detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLO result has no box detections")
    box = boxes[index]
    xyxy = tensor_to_numpy(box.xyxy[0]).reshape(-1)
    cls_id = int(tensor_to_numpy(box.cls[0]).reshape(-1)[0])
    conf = float(tensor_to_numpy(box.conf[0]).reshape(-1)[0])
    return class_name(names, cls_id), conf, clip_bbox(xyxy, image_shape)


def detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    if getattr(result, "obb", None) is not None:
        return obb_detection_meta(result, index, image_shape)
    return box_detection_meta(result, index, image_shape)


def detection_polygon_mask(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def collect_detections(results: list[Any], image_shape: tuple[int, int]) -> list[YoloDetection]:
    detections: list[YoloDetection] = []
    for result_index, result in enumerate(results):
        for detection_index in range(detection_count(result)):
            try:
                name, conf, bbox = detection_meta(result, detection_index, image_shape)
                points = obb_points(result, detection_index, image_shape)
                mask = detection_polygon_mask(points, image_shape) if points is not None else detection_mask(result, detection_index, image_shape, bbox)
            except Exception:
                continue
            detections.append(YoloDetection(result_index, detection_index, name, conf, bbox, mask))
    return detections


def detect_objects(
    model: Any,
    color_bgr: np.ndarray,
    yolo_opts: dict[str, Any],
) -> tuple[list[Any], list[YoloDetection]]:
    results = model.predict(
        color_bgr,
        verbose=False,
        device=yolo_opts.get("device", "cpu"),
        conf=float(yolo_opts.get("conf", 0.25)),
        iou=float(yolo_opts.get("iou", 0.45)),
    )
    detections = collect_detections(results, color_bgr.shape[:2])
    return results, detections


# Compatibility names kept for existing callers; they do not perform target selection.
target_mask = detection_mask
collect_targets = collect_detections
detect_targets = detect_objects
