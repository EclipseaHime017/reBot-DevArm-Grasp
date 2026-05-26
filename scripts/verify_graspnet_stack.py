#!/usr/bin/env python3
"""Verify the local runtime needed by the rebot_grasp GraspNet demos."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def prepare_imports() -> None:
    for path in (
        PROJECT_ROOT,
        GRASPNET_ROOT,
        GRASPNET_ROOT / "models",
        GRASPNET_ROOT / "dataset",
        GRASPNET_ROOT / "utils",
        GRASPNET_ROOT / "pointnet2",
        GRASPNET_ROOT / "knn",
        PROJECT_ROOT / "sdk" / "graspnetAPI",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def check_import(module_name: str, attr: str | None = None) -> bool:
    try:
        module = importlib.import_module(module_name)
        if attr is not None:
            getattr(module, attr)
    except Exception as exc:
        print(f"[FAIL] {module_name}{'.' + attr if attr else ''}: {exc}")
        return False
    print(f"[OK] {module_name}{'.' + attr if attr else ''}")
    return True


def check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True
    print(f"[FAIL] {label} missing: {path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GraspNet demo dependencies")
    parser.add_argument(
        "--checkpoint",
        default=str(GRASPNET_ROOT / "checkpoints" / "checkpoint-rs.tar"),
        help="GraspNet checkpoint path",
    )
    parser.add_argument(
        "--engine",
        default=str(PROJECT_ROOT / "models" / "yolo11n-seg.engine"),
        help="YOLO TensorRT engine path",
    )
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="do not import pyorbbecsdk",
    )
    args = parser.parse_args()

    prepare_imports()
    ok = True

    ok &= check_file(Path(args.checkpoint), "GraspNet checkpoint")
    ok &= check_file(Path(args.engine), "YOLO engine")

    for module_name in ("numpy", "cv2", "yaml", "torch", "open3d", "ultralytics"):
        ok &= check_import(module_name)

    ok &= check_import("pointnet2._ext")
    ok &= check_import("knn_pytorch.knn_pytorch")
    ok &= check_import("graspnet", "GraspNet")
    ok &= check_import("graspnet", "pred_decode")
    ok &= check_import("graspnetAPI.grasp", "GraspGroup")
    ok &= check_import("drivers.camera", "make_camera")
    if not args.skip_camera:
        ok &= check_import("pyorbbecsdk", "Pipeline")
        ok &= check_import("pyorbbecsdk", "Config")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"[OK] torch CUDA: {torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
        else:
            print("[FAIL] torch CUDA is unavailable")
            ok = False
    except Exception as exc:
        print(f"[FAIL] torch CUDA check: {exc}")
        ok = False

    if ok:
        print("[OK] GraspNet stack is ready")
        return 0
    print("[FAIL] GraspNet stack is not ready")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
