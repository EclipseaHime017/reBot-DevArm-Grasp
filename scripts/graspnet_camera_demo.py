"""
Live RGB-D GraspNet demo for rebot_grasp cameras.

The preview loop follows scripts/ordinary_grasp_pipeline.py: YOLO detects the
current target, then GraspNet proposals are filtered to grasps whose centers
project into that target's mask or bbox.

Usage:
    conda activate graspnet
    cd /home/seeed/Downloads/rebot_grasp
    python scripts/graspnet_camera_demo.py
    python scripts/graspnet_camera_demo.py --camera-type orbbec_gemini2

Keys:
    g / space  Run GraspNet on the current frame
    q / esc    Quit

For periodic inference:
    python scripts/graspnet_camera_demo.py --auto --infer-interval 1.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
GRASPNET_API_ROOT = PROJECT_ROOT / "sdk" / "graspnetAPI"


def _prepare_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(GRASPNET_API_ROOT))
    for subdir in ("models", "dataset", "utils", "pointnet2", "knn"):
        sys.path.insert(0, str(GRASPNET_ROOT / subdir))
    sys.path.insert(0, str(GRASPNET_ROOT))


_prepare_imports()

from drivers.camera import make_camera  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI.grasp import GraspGroup  # noqa: E402
from utils.yolo_runtime import (  # noqa: E402
    ensure_jetson_tensorrt_importable,
    is_open_vocab_model,
    resolve_yolo_model_path,
    yolo_predict_kwargs,
)

DISPLAY_FLIP_X = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class Open3DGraspWindow:
    def __init__(self, title: str, top_k: int) -> None:
        self._top_k = top_k
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []
        self._initialized = False

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup) -> None:
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(DISPLAY_FLIP_X)
        geometries = [cloud_vis]
        if len(grasps) > 0:
            grasps_vis = GraspGroup(grasps.grasp_group_array.copy())
            grasps_vis = grasps_vis.nms()
            grasps_vis.sort_by_score()
            grasps_vis = grasps_vis[: self._top_k]
            grasps_vis.transform(DISPLAY_FLIP_X)
            geometries.extend(grasps_vis.to_open3d_geometry_list())
        for geom in geometries:
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


class DirectRealSenseCamera:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self._requested = (int(width), int(height), int(fps))
        self._pipeline = None
        self._align = None
        self._rs = None
        self._depth_scale_mm = 1.0
        self._K: Optional[np.ndarray] = None
        self._D = np.zeros((1, 5), dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("RealSense camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        return self._D

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(f"pyrealsense2 is not installed: {exc}") from exc

        self._rs = rs
        candidates = self._profile_candidates()
        errors = []
        for width, height, fps in candidates:
            try:
                self._start_profile(width, height, fps)
                print(f"[DirectRealSense] ready ({width}x{height}@{fps})")
                return
            except Exception as exc:
                errors.append(f"{width}x{height}@{fps}: {exc}")
                self.close()

        raise RuntimeError("No RealSense RGB-D profile produced frames:\n  " + "\n  ".join(errors))

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None

    def warm_up(self, n_frames: int = 20) -> None:
        for _ in range(n_frames):
            self.get_frame()

    def get_frame(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None or self._align is None:
            return None, None
        try:
            frames = self._pipeline.wait_for_frames(1000)
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None

            color_bgr = np.asanyarray(color_frame.get_data())
            if color_bgr.ndim == 1:
                h, w = color_frame.get_height(), color_frame.get_width()
                color_bgr = color_bgr.reshape(h, w, 3)

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = (depth_raw * self._depth_scale_mm).astype(np.uint16)
            return np.ascontiguousarray(color_bgr), np.ascontiguousarray(depth_mm)
        except Exception:
            return None, None

    def _profile_candidates(self) -> list[tuple[int, int, int]]:
        requested = self._requested
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

    def _start_profile(self, width: int, height: int, fps: int) -> None:
        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        frames = pipeline.wait_for_frames(3000)
        aligned = align.process(frames)
        if not aligned.get_color_frame() or not aligned.get_depth_frame():
            pipeline.stop()
            raise RuntimeError("started but did not receive both color and depth frames")

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cx = getattr(intr, "ppx", getattr(intr, "cx", None))
        cy = getattr(intr, "ppy", getattr(intr, "cy", None))
        if cx is None or cy is None:
            pipeline.stop()
            raise RuntimeError(f"cannot read RealSense principal point: {intr!r}")

        self._K = np.array(
            [
                [intr.fx, 0.0, cx],
                [0.0, intr.fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._depth_scale_mm = depth_scale * 1000.0
        self._pipeline = pipeline
        self._align = align


@dataclass
class DetectionTarget:
    result_index: int
    detection_index: int
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera GraspNet demo")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="rebot_grasp config YAML path",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(GRASPNET_ROOT / "checkpoints" / "checkpoint-rs.tar"),
        help="GraspNet checkpoint path",
    )
    parser.add_argument(
        "--camera-type",
        default=None,
        choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"),
        help="camera driver to use; defaults to camera.type in config",
    )
    parser.add_argument("--width", type=int, default=None, help="override camera color/depth width")
    parser.add_argument("--height", type=int, default=None, help="override camera color/depth height")
    parser.add_argument("--fps", type=int, default=None, help="override camera FPS")
    parser.add_argument("--debug-frames", action="store_true", help="print first RGB-D frame statistics")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05, help="meters")
    parser.add_argument("--max-depth", type=float, default=2.0, help="meters")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--auto", action="store_true", help="run inference periodically")
    parser.add_argument("--infer-interval", type=float, default=1.5, help="seconds")
    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="run inference and print counts without opening the Open3D window",
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help="class name to grasp; when omitted, the highest-confidence detection is used",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="override config yolo.model_name; relative names are loaded from rebot_grasp/models",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="override config yolo.device, for example cpu or cuda:0",
    )
    parser.add_argument("--yolo-conf", type=float, default=None, help="override detection.conf_threshold")
    parser.add_argument("--yolo-iou", type=float, default=None, help="override detection.iou_threshold")
    parser.add_argument(
        "--infer-every-live",
        type=int,
        default=None,
        help="run YOLO preview every N frames; defaults to grasp_pipeline.infer_every_live",
    )
    parser.add_argument(
        "--target-margin-px",
        type=int,
        default=12,
        help="dilate the selected target mask before filtering GraspNet grasps",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="disable target detection and run full-scene GraspNet inference",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_camera(cfg: dict, args: argparse.Namespace) -> dict:
    cam_cfg = cfg.setdefault("camera", {})
    if args.camera_type is not None:
        cam_cfg["type"] = args.camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    if args.width is not None:
        cam_cfg["color_width"] = args.width
        cam_cfg["depth_width"] = args.width
    if args.height is not None:
        cam_cfg["color_height"] = args.height
        cam_cfg["depth_height"] = args.height
    if args.fps is not None:
        cam_cfg["fps"] = args.fps
    elif args.camera_type is not None and "realsense" in cam_type:
        cam_cfg["fps"] = 15

    return cfg


def load_yolo(cfg: dict, args: argparse.Namespace) -> tuple[Optional[Any], dict[str, Any]]:
    gp_cfg = cfg.get("grasp_pipeline", {})
    yolo_opts: dict[str, Any] = {
        "enabled": not args.no_yolo,
        "infer_every": max(1, int(args.infer_every_live or gp_cfg.get("infer_every_live", 3))),
    }
    if args.no_yolo:
        return None, yolo_opts

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for target-aware mode. Use --no-yolo for full-scene mode.") from exc

    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    model_name = str(args.yolo_model or yolo_cfg.get("model_name", "yolo11n-seg.engine"))
    model_path = resolve_yolo_model_path(PROJECT_ROOT, model_name)
    device = args.yolo_device or yolo_cfg.get("device", "auto")
    conf = float(args.yolo_conf if args.yolo_conf is not None else det_cfg.get("conf_threshold", 0.25))
    iou = float(args.yolo_iou if args.yolo_iou is not None else det_cfg.get("iou_threshold", 0.45))
    custom_classes = list(yolo_cfg.get("custom_classes", []))
    use_world = bool(yolo_cfg.get("use_world", False))

    print(f"Loading YOLO target detector: {model_path}")
    ensure_jetson_tensorrt_importable()
    model = YOLO(str(model_path))
    if use_world and is_open_vocab_model(model_name) and custom_classes:
        model.set_classes(custom_classes)
        print(f"YOLO open-vocabulary classes: {custom_classes}")

    yolo_opts.update(
        {
            "model_name": model_name,
            "device": device,
            "conf": conf,
            "iou": iou,
            "custom_classes": custom_classes,
            "predict_kwargs": yolo_predict_kwargs(model_name, device, conf, iou),
        }
    )
    return model, yolo_opts


def _tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
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


def _safe_attr_row(container: Any, attr: str, index: int) -> Optional[np.ndarray]:
    values = getattr(container, attr, None)
    if values is None:
        return None
    try:
        return _tensor_to_numpy(values[index])
    except Exception:
        return None


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    try:
        return str(names[cls_id])
    except Exception:
        return str(cls_id)


def _detection_count(result: Any) -> int:
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


def _clip_bbox(values: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
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


def _detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    names = getattr(result, "names", {})
    obb = getattr(result, "obb", None)
    if obb is not None:
        cls_row = _safe_attr_row(obb, "cls", index)
        conf_row = _safe_attr_row(obb, "conf", index)
        xyxy_row = _safe_attr_row(obb, "xyxy", index)
        if cls_row is not None and conf_row is not None and xyxy_row is not None:
            cls_id = int(np.asarray(cls_row).reshape(-1)[0])
            conf = float(np.asarray(conf_row).reshape(-1)[0])
            bbox = _clip_bbox(np.asarray(xyxy_row).reshape(-1), image_shape)
            return _class_name(names, cls_id), conf, bbox

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLO result has neither OBB nor boxes")
    box = boxes[index]
    xyxy = _tensor_to_numpy(box.xyxy[0]).reshape(-1)
    cls_id = int(_tensor_to_numpy(box.cls[0]).reshape(-1)[0])
    conf = float(_tensor_to_numpy(box.conf[0]).reshape(-1)[0])
    return _class_name(names, cls_id), conf, _clip_bbox(xyxy, image_shape)


def _target_mask(result: Any, index: int, image_shape: tuple[int, int], bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image_shape
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is not None:
        try:
            if len(data) > index:
                mask = _tensor_to_numpy(data[index])
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


def collect_targets(results: list[Any], image_shape: tuple[int, int]) -> list[DetectionTarget]:
    targets: list[DetectionTarget] = []
    for result_index, result in enumerate(results):
        for detection_index in range(_detection_count(result)):
            try:
                class_name, conf, bbox = _detection_meta(result, detection_index, image_shape)
                mask = _target_mask(result, detection_index, image_shape, bbox)
            except Exception:
                continue
            targets.append(
                DetectionTarget(
                    result_index=result_index,
                    detection_index=detection_index,
                    class_name=class_name,
                    conf=conf,
                    bbox_xyxy=bbox,
                    mask=mask,
                )
            )
    return targets


def select_target(targets: list[DetectionTarget], target_class: Optional[str]) -> Optional[DetectionTarget]:
    if not targets:
        return None
    candidates = targets
    if target_class:
        target_norm = target_class.casefold()
        exact = [target for target in targets if target.class_name.casefold() == target_norm]
        contains = [target for target in targets if target_norm in target.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda target: target.conf)


def detect_targets(
    model: Any,
    color_bgr: np.ndarray,
    yolo_opts: dict[str, Any],
    target_class: Optional[str],
) -> tuple[list[Any], list[DetectionTarget], Optional[DetectionTarget]]:
    results = model.predict(color_bgr, **dict(yolo_opts.get("predict_kwargs", {})))
    targets = collect_targets(results, color_bgr.shape[:2])
    return results, targets, select_target(targets, target_class)


def target_status_text(selected: Optional[DetectionTarget], targets: list[DetectionTarget], target_class: Optional[str]) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(targets)}"
    if target_class:
        return f"target={target_class} not found detections={len(targets)}"
    return f"target not found detections={len(targets)}"


def draw_target_overlay(
    frame: np.ndarray,
    targets: list[DetectionTarget],
    selected: Optional[DetectionTarget],
    target_class: Optional[str],
) -> np.ndarray:
    display = frame.copy()
    if selected is not None:
        mask = selected.mask.astype(bool)
        if mask.shape == display.shape[:2] and int(mask.sum()) > 0:
            overlay = display.copy()
            overlay[mask] = (0, 190, 80)
            display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)

    selected_key = None
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
    for target in targets:
        is_selected = selected_key == (target.result_index, target.detection_index)
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def build_net(checkpoint_path: str, num_view: int) -> GraspNet:
    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA, but torch.cuda is unavailable.")

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0")
    net.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return net


def make_workspace_mask(depth_mm: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    return (depth_mm > min_mm) & (depth_mm < max_mm)


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[dict, o3d.geometry.PointCloud, np.ndarray]:
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(
            depth_mm,
            (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = make_workspace_mask(depth, min_depth_m, max_depth_m)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    h, w = depth.shape
    camera = CameraInfo(
        w,
        h,
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        1000.0,
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[mask]
    color_masked = color_rgb[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs].astype(np.float32)
    end_points = {
        "point_clouds": torch.from_numpy(cloud_sampled[np.newaxis]).cuda(non_blocking=True),
        "cloud_colors": color_masked[idxs],
    }

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    return end_points, o3d_cloud, cloud_masked


def filter_grasps_by_mask(
    gg: GraspGroup,
    mask: np.ndarray,
    K: np.ndarray,
    margin_px: int = 0,
) -> tuple[GraspGroup, int]:
    if len(gg) == 0:
        return gg, 0

    target_mask = (mask > 0).astype(np.uint8)
    if margin_px > 0:
        kernel_size = max(1, int(margin_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)

    h, w = target_mask.shape[:2]
    centers = gg.translations
    z = centers[:, 2]
    valid_z = z > 1e-6
    u = np.zeros(len(gg), dtype=np.int32)
    v = np.zeros(len(gg), dtype=np.int32)
    u[valid_z] = np.round(float(K[0, 0]) * centers[valid_z, 0] / z[valid_z] + float(K[0, 2])).astype(np.int32)
    v[valid_z] = np.round(float(K[1, 1]) * centers[valid_z, 1] / z[valid_z] + float(K[1, 2])).astype(np.int32)
    in_bounds = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    keep = np.zeros(len(gg), dtype=bool)
    keep[in_bounds] = target_mask[v[in_bounds], u[in_bounds]] > 0
    return gg[keep], int(np.count_nonzero(keep))


def infer_grasps(
    net: GraspNet,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float,
    target_mask: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    target_margin_px: int = 0,
) -> tuple[GraspGroup, int, int]:
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    decoded_count = len(gg)
    target_count = decoded_count
    if target_mask is not None:
        if K is None:
            raise ValueError("K is required when target_mask is provided")
        gg, target_count = filter_grasps_by_mask(gg, target_mask, K, target_margin_px)
    if len(gg) == 0:
        return gg, decoded_count, target_count

    if collision_thresh > 0:
        detector = ModelFreeCollisionDetector(raw_cloud, voxel_size=voxel_size)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]
    return gg, decoded_count, target_count


def overlay_status(frame: np.ndarray, status: str, auto: bool, target_status: str = "") -> np.ndarray:
    display = frame.copy()
    mode = "AUTO" if auto else "SNAPSHOT"
    lines = [
        f"GraspNet Target Camera Demo | {mode}",
        "G/SPACE: infer   Q/ESC: quit",
    ]
    if target_status:
        lines.append(target_status)
    lines.append(status)
    y = 28
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return display


def blank_status_frame(cfg: dict, status: str, auto: bool) -> np.ndarray:
    cam_cfg = cfg.get("camera", {})
    w = int(cam_cfg.get("color_width", 1280))
    h = int(cam_cfg.get("color_height", 720))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    return overlay_status(frame, status, auto)


def frame_stats(name: str, frame: np.ndarray) -> str:
    if frame is None:
        return f"{name}=None"
    return (
        f"{name}: shape={frame.shape}, dtype={frame.dtype}, "
        f"min={int(frame.min())}, max={int(frame.max())}, mean={float(frame.mean()):.2f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = configure_camera(cfg, args)

    yolo_model, yolo_opts = load_yolo(cfg, args)
    net = build_net(args.checkpoint, args.num_view)
    cam_cfg = cfg["camera"]
    print(
        "Using camera: "
        f"{cam_cfg['type']} "
        f"{cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')}"
    )
    cam_type = str(cam_cfg["type"]).lower()
    if "realsense" in cam_type:
        cam = DirectRealSenseCamera(
            cam_cfg.get("color_width", 1280),
            cam_cfg.get("color_height", 720),
            cam_cfg.get("fps", 15),
        )
    else:
        cam = make_camera(cfg)
    vis: Optional[Open3DGraspWindow] = None
    visualizer_enabled = not args.no_visualizer
    status = "warming up camera..."
    target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
    last_targets: list[DetectionTarget] = []
    selected_target: Optional[DetectionTarget] = None
    last_infer_t = 0.0
    frame_index = 0
    window_name = "GraspNet Live Camera"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K
        print("Camera intrinsics:")
        print(K)
        print("Press G or SPACE to infer current frame. Press Q or ESC to quit.")
        if yolo_model is not None:
            print(
                "YOLO target mode enabled: "
                f"model={yolo_opts.get('model_name')} device={yolo_opts.get('device')} "
                f"preview_every={yolo_opts.get('infer_every')} target={args.target_class or 'best detection'}"
            )
        if args.auto:
            print(f"Auto inference enabled, interval={args.infer_interval:.2f}s")

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                status = "waiting for color/depth frames..."
                cv2.imshow(window_name, blank_status_frame(cfg, status, args.auto))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            if args.debug_frames:
                print(frame_stats("color", color_bgr))
                print(frame_stats("depth", depth_mm))
                args.debug_frames = False

            frame_index += 1
            if yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_targets, selected_target = detect_targets(
                        yolo_model,
                        color_bgr,
                        yolo_opts,
                        args.target_class,
                    )
                    target_status = target_status_text(selected_target, last_targets, args.target_class)
                except Exception as exc:
                    last_targets = []
                    selected_target = None
                    target_status = f"YOLO failed: {exc}"

            now = time.time()
            should_infer = args.auto and (now - last_infer_t >= args.infer_interval)
            display_base = color_bgr
            if yolo_model is not None:
                display_base = draw_target_overlay(color_bgr, last_targets, selected_target, args.target_class)
            display = overlay_status(display_base, status, args.auto, target_status)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G"), ord(" ")):
                should_infer = True

            if should_infer:
                last_infer_t = now
                try:
                    tic = time.time()
                    target_mask = None
                    target_label = "full scene"
                    if yolo_model is not None:
                        _, last_targets, selected_target = detect_targets(
                            yolo_model,
                            color_bgr,
                            yolo_opts,
                            args.target_class,
                        )
                        target_status = target_status_text(selected_target, last_targets, args.target_class)
                        if selected_target is None:
                            status = f"inference skipped: {target_status}"
                            print(status)
                            continue
                        target_mask = selected_target.mask
                        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

                    end_points, o3d_cloud, raw_cloud = build_end_points(
                        color_bgr,
                        depth_mm,
                        K,
                        args.num_point,
                        args.min_depth,
                        args.max_depth,
                    )
                    gg, decoded_count, target_count = infer_grasps(
                        net,
                        end_points,
                        raw_cloud,
                        args.collision_thresh,
                        args.voxel_size,
                        target_mask=target_mask,
                        K=K,
                        target_margin_px=args.target_margin_px,
                    )
                    elapsed = time.time() - tic
                    if yolo_model is None:
                        status = f"grasps={len(gg)} decoded={decoded_count} inference={elapsed:.2f}s"
                    else:
                        status = (
                            f"{target_label} grasps={len(gg)} "
                            f"target={target_count}/{decoded_count} inference={elapsed:.2f}s"
                        )
                    print(status)

                    if visualizer_enabled:
                        try:
                            if vis is None:
                                vis = Open3DGraspWindow("GraspNet Grasps", args.top_k)
                            vis.update(o3d_cloud, gg)
                        except Exception as exc:
                            visualizer_enabled = False
                            if vis is not None:
                                vis.close()
                                vis = None
                            print(f"Open3D visualizer disabled: {exc}")
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(status)

            if vis is not None and not vis.poll():
                vis.close()
                vis = None

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
