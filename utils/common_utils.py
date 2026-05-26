"""Small shared helpers used by perception utilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    try:
        return str(names[cls_id])
    except Exception:
        return str(cls_id)


def clip_bbox(values: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))
    return x1, y1, x2, y2


def detection_count(result: Any) -> int:
    for attr in ("obb", "boxes"):
        container = getattr(result, attr, None)
        if container is None:
            continue
        try:
            count = len(container)
        except Exception:
            continue
        if count > 0:
            return count
    return 0
