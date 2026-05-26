"""YOLO runtime helpers for PyTorch and TensorRT model files."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def resolve_yolo_model_path(project_root: Path, model_name: str) -> Path:
    model_path = Path(str(model_name)).expanduser()
    if model_path.is_absolute():
        return model_path
    if len(model_path.parts) > 1:
        return project_root / model_path
    return project_root / "models" / model_path


def is_tensorrt_engine(model_name: str | Path) -> bool:
    return Path(str(model_name)).suffix.lower() == ".engine"


def is_open_vocab_model(model_name: str | Path) -> bool:
    name = Path(str(model_name)).name.lower()
    return not is_tensorrt_engine(model_name) and ("world" in name or "yoloe" in name)


def ensure_jetson_tensorrt_importable() -> None:
    """Expose Jetson apt-installed TensorRT bindings inside conda envs."""
    try:
        importlib.import_module("tensorrt")
        return
    except ImportError:
        pass

    candidates = (
        Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
    )
    for path in candidates:
        if (path / "tensorrt").exists() and str(path) not in sys.path:
            sys.path.append(str(path))


def yolo_predict_kwargs(
    model_name: str | Path,
    device: Any = None,
    conf: float | None = None,
    iou: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"verbose": False}
    if conf is not None:
        kwargs["conf"] = float(conf)
    if iou is not None:
        kwargs["iou"] = float(iou)

    if device is None:
        return kwargs
    device_text = str(device).strip()
    if not device_text or device_text.lower() == "auto":
        return kwargs
    if is_tensorrt_engine(model_name) and device_text.lower() == "cpu":
        return kwargs
    kwargs["device"] = device
    return kwargs
