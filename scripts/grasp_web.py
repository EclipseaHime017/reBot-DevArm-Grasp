#!/usr/bin/env python3
"""Local web UI for YOLO target selection and GraspNet grasp-point preview.

Usage:
    cd /home/seeed/Downloads/rebot_grasp
    conda activate graspnet
    python scripts/grasp_web.py --host 0.0.0.0 --port 8000
    python scripts/grasp_web.py --host 0.0.0.0 --port 8000 --enable-robot
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.rebot_arm import RebotArm  # noqa: E402
from scripts.grasp import (  # noqa: E402
    DirectRealSenseCamera,
    _cam_to_base,
    _execute_grasp,
    _move_ready,
    _select_best_grasp,
    _transform_grasp,
    build_place_config,
    configure_camera,
    load_config,
    load_hand_eye,
    load_yolo,
)
from scripts.graspnet_camera_demo import (  # noqa: E402
    DetectionTarget,
    build_end_points,
    build_net,
    detect_targets,
    infer_grasps,
    overlay_status,
    select_target,
    target_status_text,
)
from utils.ordinary_grasp import (  # noqa: E402
    GraspPose,
    draw_grasp as draw_ordinary_grasp,
    estimate_grasps as estimate_ordinary_grasps,
)
from graspnetAPI.grasp import Grasp, GraspGroup  # noqa: E402


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web target selector for GraspNet grasp-point preview")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(GRASPNET_ROOT / "checkpoints" / "checkpoint-rs.tar"))
    parser.add_argument("--enable-robot", action="store_true", help="allow real grasp execution")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--target-margin-px", type=int, default=12)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=1, help="run YOLO/ordinary preview every N frames")
    parser.add_argument("--pregrasp-offset", type=float, default=None)
    parser.add_argument("--retreat-offset", type=float, default=None)
    parser.add_argument("--grasp-forward-offset", type=float, default=None, help="meters; move final grasp farther along approach axis")
    parser.add_argument("--grasp-lateral-offset", type=float, default=None)
    parser.add_argument("--grasp-vertical-offset", type=float, default=None)
    parser.add_argument("--grasp-roll-offset-deg", type=float, default=None)
    parser.add_argument("--grasp-pitch-offset-deg", type=float, default=None)
    parser.add_argument("--grasp-yaw-offset-deg", type=float, default=None)
    parser.add_argument("--camera-x-offset", type=float, default=None, help="meters; extrinsic correction along camera X")
    parser.add_argument("--camera-y-offset", type=float, default=None, help="meters; extrinsic correction along camera Y")
    parser.add_argument("--camera-z-offset", type=float, default=None, help="meters; extrinsic correction along camera Z")
    parser.add_argument("--camera-roll-offset-deg", type=float, default=None, help="degrees; extrinsic correction around camera X")
    parser.add_argument("--camera-pitch-offset-deg", type=float, default=None, help="degrees; extrinsic correction around camera Y")
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=None, help="degrees; extrinsic correction around camera Z")
    parser.add_argument("--base-x-offset", type=float, default=None, help="meters; extrinsic correction along robot base X")
    parser.add_argument("--base-y-offset", type=float, default=None, help="meters; extrinsic correction along robot base Y")
    parser.add_argument("--base-z-offset", type=float, default=None, help="meters; extrinsic correction along robot base Z")
    parser.add_argument("--base-roll-offset-deg", type=float, default=None, help="degrees; extrinsic correction around robot base X")
    parser.add_argument("--base-pitch-offset-deg", type=float, default=None, help="degrees; extrinsic correction around robot base Y")
    parser.add_argument("--base-yaw-offset-deg", type=float, default=None, help="degrees; extrinsic correction around robot base Z")
    parser.add_argument("--no-place-after-grasp", action="store_true", help="disable base-rotate/place/home sequence after successful grasp")
    parser.add_argument("--place-base-delta-deg", type=float, default=None, help="degrees; base joint relative rotation after successful grasp")
    parser.add_argument("--place-base-direction", choices=("auto", "positive", "negative"), default=None, help="base joint rotation direction")
    parser.add_argument("--place-base-rotate-duration", type=float, default=None, help="seconds; base joint rotation duration")
    parser.add_argument("--place-base-safety-margin-deg", type=float, default=None, help="degrees; keep base target away from joint limits")
    parser.add_argument("--no-home-after-place", action="store_true", help="return ready pose instead of joint-zero home after placing")
    parser.add_argument("--gripper-open-width", type=float, default=0.09)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--graspnet-interval", type=float, default=1.0, help="seconds between automatic grasp-point updates")
    parser.add_argument("--auto-graspnet", dest="no_auto_graspnet", action="store_false", help="enable automatic GraspNet updates")
    parser.add_argument("--no-auto-graspnet", action="store_true", help="disable automatic GraspNet updates")
    parser.add_argument("--no-ordinary-grasp", action="store_true", help="disable high-frequency ordinary grasp preview")
    parser.add_argument("--ordinary-depth-quantile", type=float, default=None, help="depth quantile used by ordinary grasp preview")
    parser.set_defaults(no_auto_graspnet=True)
    return parser.parse_args()


def detection_tags(target: DetectionTarget, selected: DetectionTarget | None) -> list[str]:
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
        target_key = (target.result_index, target.detection_index)
        if target_key == selected_key:
            return ["target"]
    return []


def draw_detection_boxes(frame: np.ndarray, targets: list, selected, target_class: str | None) -> np.ndarray:
    display = frame.copy()
    for target in targets:
        tags = detection_tags(target, selected)
        is_selected = "target" in tags
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if tags:
            label = f"{'/'.join(tags)} {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def is_generic_class_names(names: list[str]) -> bool:
    if len(names) != 80:
        return False
    return all(name == f"class{i}" for i, name in enumerate(names))


def looks_like_generic_class_names(names: list[str]) -> bool:
    if not names:
        return True
    for name in names:
        if not name.startswith("class"):
            return False
        suffix = name[5:]
        if not suffix.isdigit():
            return False
    return True


def looks_like_numeric_class_names(names: list[str]) -> bool:
    return len(names) == 80 and all(name.isdigit() and int(name) == idx for idx, name in enumerate(names))


def remap_generic_class_name(name: str) -> str:
    if name.isdigit():
        idx = int(name)
        if 0 <= idx < len(COCO80_NAMES):
            return COCO80_NAMES[idx]
        return name
    if not name.startswith("class"):
        return name
    try:
        idx = int(name[5:])
    except ValueError:
        return name
    if 0 <= idx < len(COCO80_NAMES):
        return COCO80_NAMES[idx]
    return name


def remap_targets_to_coco(targets: list[DetectionTarget]) -> None:
    for target in targets:
        target.class_name = remap_generic_class_name(target.class_name)


def remap_ordinary_grasps_to_coco(grasps: list[GraspPose]) -> None:
    for grasp in grasps:
        grasp.class_name = remap_generic_class_name(grasp.class_name)


def select_ordinary_target(grasps: list[GraspPose], target_class: str | None) -> GraspPose | None:
    valid = [grasp for grasp in grasps if grasp.is_valid]
    if not valid:
        return None
    candidates = valid
    if target_class:
        target_norm = target_class.casefold()
        exact = [grasp for grasp in valid if grasp.class_name.casefold() == target_norm]
        contains = [grasp for grasp in valid if target_norm in grasp.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda grasp: grasp.conf)


def ordinary_grasp_tags(grasp: GraspPose, selected: GraspPose | None) -> list[str]:
    if selected is grasp:
        return ["target"]
    return []


def _project_point(K: np.ndarray, xyz: np.ndarray) -> tuple[int, int] | None:
    x, y, z = [float(v) for v in xyz]
    if z <= 1e-6:
        return None
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    return u, v


def draw_graspnet_grasp_point(frame: np.ndarray, grasp: Grasp | None, K: np.ndarray | None) -> None:
    if grasp is None:
        return
    if K is None:
        return
    center = _project_point(K, np.asarray(grasp.translation, dtype=np.float64))
    if center is None:
        return
    u, v = center
    if not (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
        return

    cv2.circle(frame, (u, v), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (u, v), 17, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (u - 28, v), (u + 28, v), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (u, v - 28), (u, v + 28), (255, 255, 255), 2, cv2.LINE_AA)

    try:
        R = np.asarray(grasp.rotation_matrix, dtype=np.float64)
        axis = R[:, 1] * max(float(grasp.width), 0.03) * 0.5
        p0 = _project_point(K, np.asarray(grasp.translation, dtype=np.float64) - axis)
        p1 = _project_point(K, np.asarray(grasp.translation, dtype=np.float64) + axis)
        if p0 is not None and p1 is not None:
            cv2.line(frame, p0, p1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(frame, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception:
        pass

    label = f"GraspNet score={float(grasp.score):.2f} width={float(grasp.width) * 100:.1f}cm"
    label_pos = (min(frame.shape[1] - 360, u + 18), max(28, v - 18))
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    xyz = [float(v) for v in grasp.translation]
    line2 = f"u={u} v={v}  X:{xyz[0]:+.3f} Y:{xyz[1]:+.3f} Z:{xyz[2]:+.3f}"
    line2_pos = (label_pos[0], min(frame.shape[0] - 14, label_pos[1] + 26))
    cv2.putText(frame, line2, line2_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, line2, line2_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)


def draw_bottom_grasp_summary(frame: np.ndarray, info: dict | None) -> None:
    if not info:
        return
    xyz = info.get("translation") or []
    uv = info.get("uv")
    if len(xyz) != 3:
        return
    method = info.get("method", "grasp")
    text = (
        f"{method}={info.get('target')} score={info.get('score'):.2f} "
        f"uv={uv} xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
        f"width={float(info.get('width_m', 0.0)) * 100:.1f}cm"
    )
    y = frame.shape[0] - 16
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 140), 2, cv2.LINE_AA)


class GraspWebApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = configure_camera(load_config(Path(args.config)), args)
        self.target_class = args.target_class or ""
        self.class_names: list[str] = []
        grasp_cfg = self.cfg.get("grasp_pipeline", {}).get("grasp", {})
        self.ordinary_depth_quantile = float(
            args.ordinary_depth_quantile
            if args.ordinary_depth_quantile is not None
            else grasp_cfg.get("depth_quantile", 0.75)
        )
        self.grasp_forward_offset_m = float(
            args.grasp_forward_offset
            if args.grasp_forward_offset is not None
            else grasp_cfg.get("grasp_forward_offset_m", 0.0)
        )
        self.grasp_lateral_offset_m = float(
            args.grasp_lateral_offset
            if args.grasp_lateral_offset is not None
            else grasp_cfg.get("grasp_lateral_offset_m", 0.0)
        )
        self.grasp_vertical_offset_m = float(
            args.grasp_vertical_offset
            if args.grasp_vertical_offset is not None
            else grasp_cfg.get("grasp_vertical_offset_m", 0.0)
        )
        self.grasp_roll_offset_deg = float(
            args.grasp_roll_offset_deg
            if args.grasp_roll_offset_deg is not None
            else grasp_cfg.get("grasp_roll_offset_deg", 0.0)
        )
        self.grasp_pitch_offset_deg = float(
            args.grasp_pitch_offset_deg
            if args.grasp_pitch_offset_deg is not None
            else grasp_cfg.get("grasp_pitch_offset_deg", 0.0)
        )
        self.grasp_yaw_offset_deg = float(
            args.grasp_yaw_offset_deg
            if args.grasp_yaw_offset_deg is not None
            else grasp_cfg.get("grasp_yaw_offset_deg", 0.0)
        )
        self.camera_x_offset_m = float(
            args.camera_x_offset if args.camera_x_offset is not None else grasp_cfg.get("camera_x_offset_m", 0.0)
        )
        self.camera_y_offset_m = float(
            args.camera_y_offset if args.camera_y_offset is not None else grasp_cfg.get("camera_y_offset_m", 0.0)
        )
        self.camera_z_offset_m = float(
            args.camera_z_offset if args.camera_z_offset is not None else grasp_cfg.get("camera_z_offset_m", 0.0)
        )
        self.camera_roll_offset_deg = float(
            args.camera_roll_offset_deg
            if args.camera_roll_offset_deg is not None
            else grasp_cfg.get("camera_roll_offset_deg", 0.0)
        )
        self.camera_pitch_offset_deg = float(
            args.camera_pitch_offset_deg
            if args.camera_pitch_offset_deg is not None
            else grasp_cfg.get("camera_pitch_offset_deg", 0.0)
        )
        self.camera_yaw_offset_deg = float(
            args.camera_yaw_offset_deg
            if args.camera_yaw_offset_deg is not None
            else grasp_cfg.get("camera_yaw_offset_deg", 0.0)
        )
        self.base_x_offset_m = float(
            args.base_x_offset if args.base_x_offset is not None else grasp_cfg.get("base_x_offset_m", 0.0)
        )
        self.base_y_offset_m = float(
            args.base_y_offset if args.base_y_offset is not None else grasp_cfg.get("base_y_offset_m", 0.0)
        )
        self.base_z_offset_m = float(
            args.base_z_offset if args.base_z_offset is not None else grasp_cfg.get("base_z_offset_m", 0.0)
        )
        self.base_roll_offset_deg = float(
            args.base_roll_offset_deg
            if args.base_roll_offset_deg is not None
            else grasp_cfg.get("base_roll_offset_deg", 0.0)
        )
        self.base_pitch_offset_deg = float(
            args.base_pitch_offset_deg
            if args.base_pitch_offset_deg is not None
            else grasp_cfg.get("base_pitch_offset_deg", 0.0)
        )
        self.base_yaw_offset_deg = float(
            args.base_yaw_offset_deg
            if args.base_yaw_offset_deg is not None
            else grasp_cfg.get("base_yaw_offset_deg", 0.0)
        )

        self.yolo_model = None
        self.yolo_opts: dict = {}
        self.net = None
        self.cam = None
        self.K: np.ndarray | None = None
        self.robot: RebotArm | None = None
        self.ready_joints: np.ndarray | None = None
        self.T_hand_eye: np.ndarray | None = None
        self.calibration_path: Path | None = None
        self.calibration_status = "not checked"

        self.frame_lock = threading.RLock()
        self.model_lock = threading.RLock()
        self.infer_lock = threading.Lock()
        self.execute_requested = threading.Event()
        self.pending_grasp_update = threading.Event()
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.auto_grasp_thread: threading.Thread | None = None
        self.last_auto_grasp_t = 0.0

        self.latest_color: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.last_targets = []
        self.selected_target = None
        self.last_grasps: GraspGroup | None = None
        self.last_best: Grasp | None = None
        self.last_grasp_info: dict | None = None
        self.last_ordinary_grasps: list[GraspPose] = []
        self.last_ordinary_best: GraspPose | None = None
        self.last_ordinary_info: dict | None = None
        self.ordinary_status = "ordinary grasp disabled" if args.no_ordinary_grasp else "ordinary grasp waiting..."

        self.status = "starting..."
        self.target_status = "target detector starting..."
        self.busy = False
        self.frame_index = 0
        self.fps_value = 0.0

    def start(self) -> None:
        self.status = "loading YOLO..."
        self.yolo_model, self.yolo_opts = load_yolo(self.cfg, self.args)
        self._patch_yolo_class_names()
        self.class_names = self._class_names_from_yolo()

        self.status = "loading GraspNet..."
        self.net = build_net(self.args.checkpoint, self.args.num_view)
        self.status = "GraspNet estimator ready"

        cam_cfg = self.cfg["camera"]
        cam_type = str(cam_cfg["type"]).lower()
        if "realsense" in cam_type:
            self.cam = DirectRealSenseCamera(
                cam_cfg.get("color_width", 1280),
                cam_cfg.get("color_height", 720),
                cam_cfg.get("fps", 15),
            )
        else:
            self.cam = make_camera(self.cfg)

        self.status = "opening camera..."
        self.cam.open()
        self.cam.warm_up(self.args.warmup)
        self.K = self.cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(self.K)

        hand_eye, mode = load_hand_eye(PROJECT_ROOT, cam_type)
        self.calibration_path = PROJECT_ROOT / "config" / "calibration" / cam_type / "hand_eye.npz"
        if hand_eye is not None and mode == "eye_in_hand":
            self.T_hand_eye = hand_eye
            self.calibration_status = f"OK: {self.calibration_path}"
            print(f"[Calibration] OK: {self.calibration_path}")
        else:
            self.calibration_status = f"missing or invalid: {self.calibration_path}"
            print(f"[WARN] hand-eye calibration missing or not eye_in_hand: {self.calibration_path}")
            print("[WARN] real grasp will be disabled until calibration is available")

        if self.args.enable_robot and self.T_hand_eye is not None:
            self._connect_robot()
        elif self.args.enable_robot:
            print("[WARN] --enable-robot was requested, but calibration is unavailable; robot not connected")

        self.status = "live preview"
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        if self.robot is not None:
            print("[Web] releasing robot...")
            acquired = self.infer_lock.acquire(timeout=8.0)
            if not acquired:
                print("[Web] robot action still busy; forcing release/home")
            try:
                self.robot.release_gripper()
                self.robot.safe_home()
            except Exception as exc:
                print(f"[Web] robot release/home failed: {exc}")
            finally:
                if acquired:
                    self.infer_lock.release()
            try:
                self.robot.disconnect()
            except Exception:
                pass
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass

    def _connect_robot(self) -> None:
        robot_cfg = self.cfg.get("robot", {})
        ready_cfg = robot_cfg.get(
            "ready_pose",
            {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
        )
        print("[Web] connecting robot...")
        self.robot = RebotArm(
            config_path=robot_cfg.get("config_path"),
            urdf_path=robot_cfg.get("urdf_path"),
            repo_root=robot_cfg.get("repo_root"),
        )
        self.robot.connect(enable=True)
        self.robot.init_gripper()
        print("[Web] moving robot to ready pose...")
        _move_ready(self.robot, ready_cfg)
        self.ready_joints = self.robot.get_joint_positions().copy()
        print("[Web] robot connected, motors enabled, ready pose reached")

    def prepare_grasp(self) -> dict:
        if not self.args.enable_robot:
            return {"ok": False, "error": "ready grasp disabled; restart with --enable-robot"}
        if self.robot is None:
            return {"ok": False, "error": "robot is not connected"}
        if not self.infer_lock.acquire(blocking=False):
            return {"ok": False, "error": "robot or grasp estimator is busy"}

        self.busy = True
        self.status = "moving robot to ready grasp pose..."
        try:
            robot_cfg = self.cfg.get("robot", {})
            ready_cfg = robot_cfg.get(
                "ready_pose",
                {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
            )
            self.robot.open_gripper(distance_m=float(self.args.gripper_open_width))
            _move_ready(self.robot, ready_cfg)
            self.ready_joints = self.robot.get_joint_positions().copy()
            with self.frame_lock:
                self.last_grasps = None
                self.last_best = None
                self.last_grasp_info = None
                self.last_ordinary_grasps = []
                self.last_ordinary_best = None
                self.last_ordinary_info = None
                self.ordinary_status = "ready pose reached; waiting for YOLO"
            self.pending_grasp_update.set()
            self.status = "ready grasp pose active; grasp update queued"
            return {
                "ok": True,
                "status": self.status,
                "ready_joints_deg": [round(float(v), 2) for v in np.degrees(self.ready_joints).tolist()],
                "grasp_update_queued": True,
            }
        except Exception as exc:
            traceback.print_exc()
            self.status = f"ready grasp failed: {exc}"
            return {"ok": False, "error": str(exc)}
        finally:
            self.busy = False
            self.infer_lock.release()

    def _class_names_from_yolo(self) -> list[str]:
        if self.yolo_model is None:
            return []
        names = getattr(self.yolo_model, "names", {})
        values: list[str] = []
        if isinstance(names, dict) and len(names) == 80:
            values = [str(names.get(i, names.get(str(i), f"class{i}"))) for i in range(80)]
        elif isinstance(names, dict):
            values = [str(v) for _, v in sorted(names.items(), key=lambda item: str(item[0]))]
        else:
            try:
                values = [str(item) for item in names]
            except Exception:
                values = []

        model_name = str(self.yolo_opts.get("model_name", "")).lower()
        if ("yolo11" in model_name or "yolo11n-seg" in model_name) and (
            looks_like_generic_class_names(values) or looks_like_numeric_class_names(values)
        ):
            return COCO80_NAMES.copy()
        if is_generic_class_names(values) or looks_like_numeric_class_names(values):
            return COCO80_NAMES.copy()

        if isinstance(names, dict):
            try:
                return [str(names[k]) for k in sorted(names, key=lambda item: int(item))]
            except Exception:
                return [str(v) for _, v in sorted(names.items(), key=lambda item: str(item[0]))]
        return values

    def _patch_yolo_class_names(self) -> None:
        if self.yolo_model is None:
            return
        names = getattr(self.yolo_model, "names", {})
        if isinstance(names, dict):
            current = [str(names.get(i, names.get(str(i), f"class{i}"))) for i in range(80)] if len(names) == 80 else []
        else:
            try:
                current = [str(item) for item in names]
            except Exception:
                current = []
        model_name = str(self.yolo_opts.get("model_name", "")).lower()
        should_patch = (
            is_generic_class_names(current)
            or looks_like_numeric_class_names(current)
            or (
                ("yolo11" in model_name or "yolo11n-seg" in model_name)
                and (looks_like_generic_class_names(current) or looks_like_numeric_class_names(current))
            )
        )
        if not should_patch:
            return

        coco = {i: name for i, name in enumerate(COCO80_NAMES)}
        for obj in (
            self.yolo_model,
            getattr(self.yolo_model, "model", None),
            getattr(self.yolo_model, "predictor", None),
            getattr(getattr(self.yolo_model, "predictor", None), "model", None),
        ):
            if obj is None:
                continue
            try:
                setattr(obj, "names", coco)
            except Exception:
                pass
        print("[YOLO] TensorRT engine names looked generic; using COCO80 class names")

    def _args_for_current_target(self) -> argparse.Namespace:
        data = vars(self.args).copy()
        data["target_class"] = self.target_class or None
        return SimpleNamespace(**data)

    def _capture_loop(self) -> None:
        fps_counter = 0
        fps_timer = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                color_bgr, depth_mm = self.cam.get_frame()
                if color_bgr is None or depth_mm is None:
                    self.status = "waiting for color/depth frames..."
                    time.sleep(0.02)
                    continue

                self.frame_index += 1
                fps_counter += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    self.fps_value = fps_counter / max(now - fps_timer, 1e-6)
                    fps_counter = 0
                    fps_timer = now

                with self.frame_lock:
                    self.latest_color = color_bgr.copy()
                    self.latest_depth = depth_mm.copy()

                self._maybe_update_yolo(color_bgr, depth_mm)
                self._maybe_start_auto_grasp()
                display = self._draw_preview(color_bgr)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    display,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self.args.jpeg_quality, 40, 95))],
                )
                if ok:
                    with self.frame_lock:
                        self.latest_jpeg = encoded.tobytes()
            except Exception as exc:
                self.status = f"capture error: {exc}"
                time.sleep(0.2)

    def _maybe_update_yolo(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> None:
        if self.yolo_model is None:
            self.target_status = "YOLO disabled: full-scene GraspNet"
            self.ordinary_status = "ordinary grasp needs YOLO"
            return
        infer_every = int(self.yolo_opts.get("infer_every", 3))
        pending_update = self.pending_grasp_update.is_set()
        if not pending_update and self.frame_index != 1 and self.frame_index % max(1, infer_every) != 0:
            return
        try:
            if not self.model_lock.acquire(blocking=False):
                return
            try:
                results, targets, selected = detect_targets(
                    self.yolo_model,
                    color_bgr,
                    self.yolo_opts,
                    self.target_class or None,
                )
            finally:
                self.model_lock.release()
            remap_targets_to_coco(targets)
            selected = select_target(targets, self.target_class or None)
            ordinary_grasps, ordinary_best, ordinary_info, ordinary_status = self._ordinary_from_yolo_results(
                results,
                depth_mm,
            )
            with self.frame_lock:
                self.last_targets = targets
                self.selected_target = selected
                self.target_status = target_status_text(selected, targets, self.target_class or None)
                self.last_ordinary_grasps = ordinary_grasps
                self.last_ordinary_best = ordinary_best
                self.last_ordinary_info = ordinary_info
                self.ordinary_status = ordinary_status
        except Exception as exc:
            self.target_status = f"YOLO failed: {exc}"
            self.ordinary_status = f"ordinary grasp failed: {exc}"

    def _ordinary_from_yolo_results(
        self,
        results: list[Any],
        depth_mm: np.ndarray,
    ) -> tuple[list[GraspPose], GraspPose | None, dict | None, str]:
        if self.args.no_ordinary_grasp:
            return [], None, None, "ordinary grasp disabled"
        if self.K is None:
            return [], None, None, "ordinary grasp waiting for camera intrinsics"

        grasps = estimate_ordinary_grasps(
            results,
            depth_mm,
            self.K,
            depth_quantile=self.ordinary_depth_quantile,
        )
        remap_ordinary_grasps_to_coco(grasps)
        target_class = self.target_class or None
        best = select_ordinary_target(grasps, target_class)
        if best is None:
            valid_count = sum(1 for grasp in grasps if grasp.is_valid)
            target_text = target_class or "best detection"
            status = f"ordinary no valid grasp for {target_text}; detections={len(grasps)} valid={valid_count}"
            return grasps, None, None, status

        info = self._ordinary_grasp_info(best)
        status = (
            f"ordinary target={best.class_name} conf={best.conf:.2f} "
            f"valid_depth={best.valid_depth_pixels} z={float(best.position[2]):.3f}m"
        )
        return grasps, best, info, status

    def _maybe_start_auto_grasp(self) -> None:
        pending_update = self.pending_grasp_update.is_set()
        if self.args.no_auto_graspnet and not pending_update:
            return
        if self.execute_requested.is_set():
            return
        if self.busy:
            return
        if self.auto_grasp_thread is not None and self.auto_grasp_thread.is_alive():
            return
        now = time.monotonic()
        interval = max(0.5, float(self.args.graspnet_interval))
        if not pending_update and now - self.last_auto_grasp_t < interval:
            return
        with self.frame_lock:
            if self.latest_color is None or self.latest_depth is None:
                return
        if self.yolo_model is not None and self.selected_target is None:
            if pending_update:
                self.status = "target changed; waiting until target is detected"
            return
        self.pending_grasp_update.clear()
        self.auto_grasp_thread = threading.Thread(
            target=self.run_grasp,
            kwargs={"execute": False, "auto": True},
            daemon=True,
        )
        self.auto_grasp_thread.start()

    def _draw_preview(self, color_bgr: np.ndarray) -> np.ndarray:
        with self.frame_lock:
            targets = list(self.last_targets)
            selected = self.selected_target
            best = self.last_best
            grasp_info = dict(self.last_grasp_info) if self.last_grasp_info is not None else None
            ordinary_grasps = list(self.last_ordinary_grasps)
            ordinary_best = self.last_ordinary_best
            ordinary_info = dict(self.last_ordinary_info) if self.last_ordinary_info is not None else None
            target_status = self.target_status
            auto_enabled = not self.args.no_auto_graspnet
            interval = max(0.5, float(self.args.graspnet_interval))
        display_base = color_bgr
        if self.yolo_model is not None:
            display_base = draw_detection_boxes(color_bgr, targets, selected, self.target_class or None)
            if not self.args.no_ordinary_grasp:
                for ordinary_grasp in ordinary_grasps:
                    draw_ordinary_grasp(display_base, ordinary_grasp, tags=ordinary_grasp_tags(ordinary_grasp, ordinary_best))
        mode = "BUSY" if self.busy else "LIVE"
        auto_text = f"auto grasp {interval:.1f}s" if auto_enabled else "auto grasp off"
        display = overlay_status(
            display_base,
            f"{mode} {self.fps_value:.1f}fps | {auto_text} | {self.status}",
            False,
            target_status,
        )
        draw_graspnet_grasp_point(display, best, self.K)
        draw_bottom_grasp_summary(display, grasp_info or ordinary_info)
        return display

    def get_jpeg(self) -> bytes | None:
        with self.frame_lock:
            return self.latest_jpeg

    def set_target(self, class_name: str) -> dict:
        class_name = remap_generic_class_name(class_name.strip())
        if class_name and self.class_names and class_name not in self.class_names:
            return {"ok": False, "error": f"unknown class: {class_name}"}
        self.target_class = class_name
        with self.frame_lock:
            self.selected_target = select_target(self.last_targets, self.target_class or None)
            self.target_status = target_status_text(self.selected_target, self.last_targets, self.target_class or None)
            self.last_grasps = None
            self.last_best = None
            self.last_grasp_info = None
            self.last_ordinary_grasps = []
            self.last_ordinary_best = None
            self.last_ordinary_info = None
            self.ordinary_status = "ordinary target changed; waiting for YOLO"
        self.pending_grasp_update.set()
        self.status = "target changed; GraspNet update queued"
        return {"ok": True, "target_class": self.target_class, "grasp_update_queued": True}

    def set_forward_offset(self, offset_m: float) -> dict:
        offset = float(offset_m)
        if not np.isfinite(offset):
            return {"ok": False, "error": "offset must be finite"}
        if offset < -0.15 or offset > 0.15:
            return {"ok": False, "error": "offset must be between -0.15 and 0.15 meters"}
        self.grasp_forward_offset_m = offset
        self.pending_grasp_update.set()
        self.status = f"forward offset set to {offset:+.3f}m; GraspNet update queued"
        return {"ok": True, "grasp_forward_offset_m": round(offset, 4), "grasp_update_queued": True}

    def set_compensation(self, payload: dict) -> dict:
        fields = {
            "grasp_forward_offset_m": ("forward_m", self.grasp_forward_offset_m, -0.15, 0.15),
            "grasp_lateral_offset_m": ("lateral_m", self.grasp_lateral_offset_m, -0.15, 0.15),
            "grasp_vertical_offset_m": ("vertical_m", self.grasp_vertical_offset_m, -0.15, 0.15),
            "grasp_roll_offset_deg": ("roll_deg", self.grasp_roll_offset_deg, -45.0, 45.0),
            "grasp_pitch_offset_deg": ("pitch_deg", self.grasp_pitch_offset_deg, -45.0, 45.0),
            "grasp_yaw_offset_deg": ("yaw_deg", self.grasp_yaw_offset_deg, -45.0, 45.0),
            "camera_x_offset_m": ("camera_x_m", self.camera_x_offset_m, -0.20, 0.20),
            "camera_y_offset_m": ("camera_y_m", self.camera_y_offset_m, -0.20, 0.20),
            "camera_z_offset_m": ("camera_z_m", self.camera_z_offset_m, -0.20, 0.20),
            "camera_roll_offset_deg": ("camera_roll_deg", self.camera_roll_offset_deg, -45.0, 45.0),
            "camera_pitch_offset_deg": ("camera_pitch_deg", self.camera_pitch_offset_deg, -45.0, 45.0),
            "camera_yaw_offset_deg": ("camera_yaw_deg", self.camera_yaw_offset_deg, -45.0, 45.0),
            "base_x_offset_m": ("base_x_m", self.base_x_offset_m, -0.20, 0.20),
            "base_y_offset_m": ("base_y_m", self.base_y_offset_m, -0.20, 0.20),
            "base_z_offset_m": ("base_z_m", self.base_z_offset_m, -0.20, 0.20),
            "base_roll_offset_deg": ("base_roll_deg", self.base_roll_offset_deg, -45.0, 45.0),
            "base_pitch_offset_deg": ("base_pitch_deg", self.base_pitch_offset_deg, -45.0, 45.0),
            "base_yaw_offset_deg": ("base_yaw_deg", self.base_yaw_offset_deg, -45.0, 45.0),
        }
        updates: dict[str, float] = {}
        for attr, (key, current, lo, hi) in fields.items():
            value = float(payload.get(key, current))
            if not np.isfinite(value):
                return {"ok": False, "error": f"{key} must be finite"}
            if value < lo or value > hi:
                return {"ok": False, "error": f"{key} must be between {lo} and {hi}"}
            updates[attr] = value
        for attr, value in updates.items():
            setattr(self, attr, value)
        self.pending_grasp_update.set()
        self.status = "grasp compensation updated; GraspNet update queued"
        return {"ok": True, "compensation": self._compensation_state(), "grasp_update_queued": True}

    def _compensation_state(self) -> dict:
        return {
            "forward_m": round(float(self.grasp_forward_offset_m), 4),
            "lateral_m": round(float(self.grasp_lateral_offset_m), 4),
            "vertical_m": round(float(self.grasp_vertical_offset_m), 4),
            "roll_deg": round(float(self.grasp_roll_offset_deg), 3),
            "pitch_deg": round(float(self.grasp_pitch_offset_deg), 3),
            "yaw_deg": round(float(self.grasp_yaw_offset_deg), 3),
            "camera_x_m": round(float(self.camera_x_offset_m), 4),
            "camera_y_m": round(float(self.camera_y_offset_m), 4),
            "camera_z_m": round(float(self.camera_z_offset_m), 4),
            "camera_roll_deg": round(float(self.camera_roll_offset_deg), 3),
            "camera_pitch_deg": round(float(self.camera_pitch_offset_deg), 3),
            "camera_yaw_deg": round(float(self.camera_yaw_offset_deg), 3),
            "base_x_m": round(float(self.base_x_offset_m), 4),
            "base_y_m": round(float(self.base_y_offset_m), 4),
            "base_z_m": round(float(self.base_z_offset_m), 4),
            "base_roll_deg": round(float(self.base_roll_offset_deg), 3),
            "base_pitch_deg": round(float(self.base_pitch_offset_deg), 3),
            "base_yaw_deg": round(float(self.base_yaw_offset_deg), 3),
        }

    def state(self) -> dict:
        with self.frame_lock:
            detections = []
            for target in self.last_targets:
                tags = detection_tags(target, self.selected_target)
                detections.append({
                    "class_name": target.class_name,
                    "conf": round(float(target.conf), 4),
                    "bbox_xyxy": list(target.bbox_xyxy),
                    "selected": "target" in tags,
                    "tag": tags[0] if tags else "",
                    "tags": tags,
                })
        if self.robot is not None:
            robot_status = "connected/enabled; waiting for explicit real grasp command"
        elif self.args.enable_robot and self.T_hand_eye is None:
            robot_status = "not connected; hand-eye calibration unavailable"
        elif self.args.enable_robot:
            robot_status = "not connected; check robot connection logs"
        else:
            robot_status = "preview only; restart with --enable-robot to connect robot"
        return {
            "ok": True,
            "busy": self.busy,
            "robot_enabled": self.args.enable_robot,
            "robot_connected": self.robot is not None,
            "robot_status": robot_status,
            "grasp_method": "graspnet",
            "ordinary_enabled": not self.args.no_ordinary_grasp,
            "ordinary_status": self.ordinary_status,
            "ordinary_grasp": self.last_ordinary_info,
            "auto_graspnet": not self.args.no_auto_graspnet,
            "grasp_update_queued": self.pending_grasp_update.is_set(),
            "grasp_forward_offset_m": round(float(self.grasp_forward_offset_m), 4),
            "compensation": self._compensation_state(),
            "place": build_place_config(self.cfg, self.args),
            "graspnet_interval_s": max(0.5, float(self.args.graspnet_interval)),
            "target_class": self.target_class,
            "class_count": len(self.class_names),
            "status": self.status,
            "target_status": self.target_status,
            "calibration": self.calibration_status,
            "detections": detections,
            "last_grasp": self.last_grasp_info,
        }

    def run_grasp(self, execute: bool, auto: bool = False) -> dict:
        if not self.infer_lock.acquire(blocking=False):
            return {"ok": False, "error": "grasp estimator is already running"}
        self.busy = True
        try:
            with self.frame_lock:
                color = None if self.latest_color is None else self.latest_color.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
            if color is None or depth is None or self.K is None:
                return {"ok": False, "error": "no RGB-D frame available"}

            call_args = self._args_for_current_target()
            best, status, target_status, targets, selected, grasps = self._infer_current_frame_web(
                color,
                depth,
                self.K,
                call_args,
            )

            info = self._grasp_info(best, selected) if best is not None else None
            if info is not None:
                self._print_grasp_info(info, auto=auto)

            with self.frame_lock:
                self.last_targets = targets
                self.selected_target = selected
                self.last_grasps = grasps
                self.last_best = best
                self.last_grasp_info = info
                self.status = status
                self.target_status = target_status

            if best is None:
                return {"ok": False, "error": "no valid GraspNet grasp", "status": status}

            self.pending_grasp_update.clear()
            if execute:
                return self._execute_best(best)

            dry_run_plan = self._dry_run_plan(best)
            result = {"ok": True, "status": status, "grasp": info, "dry_run_plan": dry_run_plan}
            if auto:
                result["auto"] = True
            return result
        except Exception as exc:
            traceback.print_exc()
            self.status = f"GraspNet grasp failed: {exc}"
            return {"ok": False, "error": str(exc)}
        finally:
            if auto:
                self.last_auto_grasp_t = time.monotonic()
            self.busy = False
            self.infer_lock.release()

    def execute_latest_grasp(self) -> dict:
        if not self.args.enable_robot:
            return {"ok": False, "error": "real grasp disabled; restart with --enable-robot"}
        if self.robot is None:
            return {"ok": False, "error": "robot is not connected"}
        if self.T_hand_eye is None:
            return {"ok": False, "error": "hand-eye calibration unavailable"}

        self.execute_requested.set()
        acquired = False
        try:
            with self.frame_lock:
                ordinary_best = self.last_ordinary_best
                ordinary_info = dict(self.last_ordinary_info) if self.last_ordinary_info is not None else None

            if ordinary_best is not None:
                self.status = "executing current ordinary grasp..."
                acquired = self.infer_lock.acquire(timeout=2.0)
                if not acquired:
                    return {
                        "ok": False,
                        "error": "robot or GraspNet update is busy; wait a moment and click Real grasp again",
                    }
                self.busy = True
                result = self._execute_ordinary_best(ordinary_best)
                if ordinary_info is not None:
                    result["grasp"] = ordinary_info
                with self.frame_lock:
                    self.status = str(result.get("status", "ordinary real grasp command finished"))
                return result

            self.busy = True
            self.status = "no ordinary grasp; waiting for current GraspNet update before real grasp..."
            acquired = self.infer_lock.acquire(timeout=20.0)
            if not acquired:
                return {"ok": False, "error": "timed out waiting for current GraspNet update"}

            self.busy = True
            with self.frame_lock:
                best = self.last_best
                info = dict(self.last_grasp_info) if self.last_grasp_info is not None else None
                selected = self.selected_target

            if best is None:
                self.status = "no displayed grasp; running immediate GraspNet before real grasp..."
                with self.frame_lock:
                    color = None if self.latest_color is None else self.latest_color.copy()
                    depth = None if self.latest_depth is None else self.latest_depth.copy()
                if color is None or depth is None or self.K is None:
                    return {"ok": False, "error": "no RGB-D frame available for immediate grasp"}

                call_args = self._args_for_current_target()
                best, status, target_status, targets, selected, grasps = self._infer_current_frame_web(
                    color,
                    depth,
                    self.K,
                    call_args,
                )
                info = self._grasp_info(best, selected) if best is not None else None
                if info is not None:
                    self._print_grasp_info(info, auto=False)
                with self.frame_lock:
                    self.last_targets = targets
                    self.selected_target = selected
                    self.last_grasps = grasps
                    self.last_best = best
                    self.last_grasp_info = info
                    self.status = status
                    self.target_status = target_status
                self.pending_grasp_update.clear()

                if best is None:
                    return {"ok": False, "error": "no valid GraspNet grasp for current target", "status": status}
            elif info is None:
                info = self._grasp_info(best, selected)

            self.status = "executing current GraspNet grasp..."
            result = self._execute_best(best)
            if info is not None:
                result["grasp"] = info
            with self.frame_lock:
                self.status = str(result.get("status", "real grasp command finished"))
            return result
        except Exception as exc:
            traceback.print_exc()
            self.status = f"real grasp failed: {exc}"
            return {"ok": False, "error": str(exc)}
        finally:
            if acquired:
                self.infer_lock.release()
            self.busy = False
            self.execute_requested.clear()

    def _infer_current_frame_web(
        self,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        K: np.ndarray,
        args: argparse.Namespace,
    ) -> tuple[Grasp | None, str, str, list[DetectionTarget], DetectionTarget | None, GraspGroup]:
        if self.net is None:
            raise RuntimeError("GraspNet is not loaded")

        target_mask = None
        target_label = "full scene"
        selected_target = None
        targets: list[DetectionTarget] = []
        grasps = GraspGroup()

        if self.yolo_model is None:
            target_status = "YOLO disabled: full-scene GraspNet"
        else:
            with self.model_lock:
                _, targets, _ = detect_targets(self.yolo_model, color_bgr, self.yolo_opts, args.target_class)
            remap_targets_to_coco(targets)
            selected_target = select_target(targets, args.target_class)
            target_status = target_status_text(selected_target, targets, args.target_class)
            if selected_target is None:
                status = f"inference skipped: {target_status}"
                return None, status, target_status, targets, selected_target, grasps
            target_mask = selected_target.mask
            target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

        tic = time.time()
        end_points, _, raw_cloud = build_end_points(
            color_bgr,
            depth_mm,
            K,
            args.num_point,
            args.min_depth,
            args.max_depth,
        )
        grasps, decoded_count, target_count = infer_grasps(
            self.net,
            end_points,
            raw_cloud,
            args.collision_thresh,
            args.voxel_size,
            target_mask=target_mask,
            K=K,
            target_margin_px=args.target_margin_px,
        )
        best = _select_best_grasp(grasps)
        elapsed = time.time() - tic

        if self.yolo_model is None:
            status = f"grasps={len(grasps)} decoded={decoded_count} inference={elapsed:.2f}s"
        else:
            status = f"{target_label} grasps={len(grasps)} target={target_count}/{decoded_count} inference={elapsed:.2f}s"
        return best, status, target_status, targets, selected_target, grasps

    def _print_grasp_info(self, info: dict, auto: bool = False) -> None:
        mode = "自动更新" if auto else "手动刷新"
        print(f"\n[G] Web GraspNet 当前最佳夹取 ({mode}):")
        print(f"  target={info.get('target')} score={info.get('score'):.4f}")
        print(f"  center_px={info.get('uv')}")
        print(f"  width_m={info.get('width_m'):.4f}")
        print(f"  position_xyz={info.get('translation')}")

    def _grasp_info(self, best: Grasp, selected: DetectionTarget | None) -> dict:
        xyz = [round(float(v), 4) for v in np.asarray(best.translation, dtype=np.float64).tolist()]
        uv = _project_point(self.K, np.asarray(best.translation, dtype=np.float64)) if self.K is not None else None
        return {
            "method": "graspnet",
            "score": round(float(best.score), 4),
            "width_m": round(float(best.width), 4),
            "height_m": round(float(best.height), 4),
            "depth_m": round(float(best.depth), 4),
            "translation": xyz,
            "uv": None if uv is None else [int(uv[0]), int(uv[1])],
            "target": selected.class_name if selected is not None else (self.target_class or "full scene"),
        }

    def _ordinary_grasp_info(self, best: GraspPose) -> dict:
        xyz = [round(float(v), 4) for v in np.asarray(best.position, dtype=np.float64).tolist()]
        return {
            "method": "ordinary",
            "score": round(float(best.conf), 4),
            "width_m": round(float(best.jaw_width_m), 4),
            "object_length_m": round(float(best.object_length_m), 4),
            "translation": xyz,
            "uv": [int(best.center_px[0]), int(best.center_px[1])],
            "target": best.class_name,
            "angle_deg": round(float(best.angle_deg), 3),
            "valid_depth_pixels": int(best.valid_depth_pixels),
        }

    def _dry_run_plan(self, best) -> dict | None:
        if self.robot is None or self.T_hand_eye is None:
            return None
        grasp_cfg = self.cfg.get("grasp_pipeline", {}).get("grasp", {})
        pregrasp_offset_m = float(
            self.args.pregrasp_offset if self.args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08)
        )
        retreat_offset_m = float(self.args.retreat_offset if self.args.retreat_offset is not None else pregrasp_offset_m)
        t_cam2base = _cam_to_base(self.T_hand_eye, self.robot)
        grasp6d, pre6d, retreat6d = _transform_grasp(
            best,
            t_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            forward_offset_m=self.grasp_forward_offset_m,
            lateral_offset_m=self.grasp_lateral_offset_m,
            vertical_offset_m=self.grasp_vertical_offset_m,
            roll_offset_rad=np.radians(self.grasp_roll_offset_deg),
            pitch_offset_rad=np.radians(self.grasp_pitch_offset_deg),
            yaw_offset_rad=np.radians(self.grasp_yaw_offset_deg),
            camera_x_offset_m=self.camera_x_offset_m,
            camera_y_offset_m=self.camera_y_offset_m,
            camera_z_offset_m=self.camera_z_offset_m,
            camera_roll_offset_rad=np.radians(self.camera_roll_offset_deg),
            camera_pitch_offset_rad=np.radians(self.camera_pitch_offset_deg),
            camera_yaw_offset_rad=np.radians(self.camera_yaw_offset_deg),
            base_x_offset_m=self.base_x_offset_m,
            base_y_offset_m=self.base_y_offset_m,
            base_z_offset_m=self.base_z_offset_m,
            base_roll_offset_rad=np.radians(self.base_roll_offset_deg),
            base_pitch_offset_rad=np.radians(self.base_pitch_offset_deg),
            base_yaw_offset_rad=np.radians(self.base_yaw_offset_deg),
        )
        return {
            "grasp6d": [round(float(v), 4) for v in grasp6d],
            "pregrasp6d": [round(float(v), 4) for v in pre6d],
            "retreat6d": [round(float(v), 4) for v in retreat6d],
        }

    def _execute_ordinary_best(self, best: GraspPose) -> dict:
        if not self.args.enable_robot:
            return {"ok": False, "error": "real grasp disabled; restart with --enable-robot"}
        if self.robot is None:
            return {"ok": False, "error": "robot is not connected"}
        if self.T_hand_eye is None:
            return {"ok": False, "error": "hand-eye calibration unavailable"}
        if best.position is None or best.tcp_rotation is None:
            return {"ok": False, "error": "ordinary grasp has no valid 3D pose"}

        grasp_cfg = self.cfg.get("grasp_pipeline", {}).get("grasp", {})
        place_cfg = build_place_config(self.cfg, self.args)
        robot_cfg = self.cfg.get("robot", {})
        ready_cfg = robot_cfg.get(
            "ready_pose",
            {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
        )
        pregrasp_offset_m = float(
            self.args.pregrasp_offset if self.args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08)
        )
        retreat_offset_m = float(self.args.retreat_offset if self.args.retreat_offset is not None else pregrasp_offset_m)
        t_cam2base = _cam_to_base(self.T_hand_eye, self.robot)
        ordinary_as_grasp = SimpleNamespace(
            translation=np.asarray(best.position, dtype=np.float64),
            rotation_matrix=np.asarray(best.tcp_rotation, dtype=np.float64),
            width=float(best.jaw_width_m),
        )
        grasp6d, pre6d, retreat6d = _transform_grasp(
            ordinary_as_grasp,
            t_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            forward_offset_m=self.grasp_forward_offset_m,
            lateral_offset_m=self.grasp_lateral_offset_m,
            vertical_offset_m=self.grasp_vertical_offset_m,
            roll_offset_rad=np.radians(self.grasp_roll_offset_deg),
            pitch_offset_rad=np.radians(self.grasp_pitch_offset_deg),
            yaw_offset_rad=np.radians(self.grasp_yaw_offset_deg),
            camera_x_offset_m=self.camera_x_offset_m,
            camera_y_offset_m=self.camera_y_offset_m,
            camera_z_offset_m=self.camera_z_offset_m,
            camera_roll_offset_rad=np.radians(self.camera_roll_offset_deg),
            camera_pitch_offset_rad=np.radians(self.camera_pitch_offset_deg),
            camera_yaw_offset_rad=np.radians(self.camera_yaw_offset_deg),
            base_x_offset_m=self.base_x_offset_m,
            base_y_offset_m=self.base_y_offset_m,
            base_z_offset_m=self.base_z_offset_m,
            base_roll_offset_rad=np.radians(self.base_roll_offset_deg),
            base_pitch_offset_rad=np.radians(self.base_pitch_offset_deg),
            base_yaw_offset_rad=np.radians(self.base_yaw_offset_deg),
        )
        print("\n[G] Web ordinary 当前最佳夹取:")
        print(f"  target={best.class_name} conf={best.conf:.4f}")
        print(f"  center_px={best.center_px} angle_deg={best.angle_deg:.2f}")
        print(f"  width_m={best.jaw_width_m:.4f}")
        print(f"  position_xyz={np.asarray(best.position, dtype=np.float64).round(4).tolist()}")
        ok = _execute_grasp(
            self.robot,
            grasp6d,
            pre6d,
            retreat6d,
            ready_cfg,
            dry_run=False,
            gripper_width_m=max(float(self.args.gripper_open_width), float(best.jaw_width_m) + 0.02),
            place_cfg=place_cfg,
            ready_joints=self.ready_joints,
        )
        return {"ok": bool(ok), "status": "ordinary real grasp executed" if ok else "ordinary real grasp attempted"}

    def _execute_best(self, best) -> dict:
        if not self.args.enable_robot:
            return {"ok": False, "error": "real grasp disabled; restart with --enable-robot"}
        if self.robot is None:
            return {"ok": False, "error": "robot is not connected"}
        if self.T_hand_eye is None:
            return {"ok": False, "error": "hand-eye calibration unavailable"}
        grasp_cfg = self.cfg.get("grasp_pipeline", {}).get("grasp", {})
        place_cfg = build_place_config(self.cfg, self.args)
        robot_cfg = self.cfg.get("robot", {})
        ready_cfg = robot_cfg.get(
            "ready_pose",
            {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
        )
        pregrasp_offset_m = float(
            self.args.pregrasp_offset if self.args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08)
        )
        retreat_offset_m = float(self.args.retreat_offset if self.args.retreat_offset is not None else pregrasp_offset_m)
        t_cam2base = _cam_to_base(self.T_hand_eye, self.robot)
        grasp6d, pre6d, retreat6d = _transform_grasp(
            best,
            t_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            forward_offset_m=self.grasp_forward_offset_m,
            lateral_offset_m=self.grasp_lateral_offset_m,
            vertical_offset_m=self.grasp_vertical_offset_m,
            roll_offset_rad=np.radians(self.grasp_roll_offset_deg),
            pitch_offset_rad=np.radians(self.grasp_pitch_offset_deg),
            yaw_offset_rad=np.radians(self.grasp_yaw_offset_deg),
            camera_x_offset_m=self.camera_x_offset_m,
            camera_y_offset_m=self.camera_y_offset_m,
            camera_z_offset_m=self.camera_z_offset_m,
            camera_roll_offset_rad=np.radians(self.camera_roll_offset_deg),
            camera_pitch_offset_rad=np.radians(self.camera_pitch_offset_deg),
            camera_yaw_offset_rad=np.radians(self.camera_yaw_offset_deg),
            base_x_offset_m=self.base_x_offset_m,
            base_y_offset_m=self.base_y_offset_m,
            base_z_offset_m=self.base_z_offset_m,
            base_roll_offset_rad=np.radians(self.base_roll_offset_deg),
            base_pitch_offset_rad=np.radians(self.base_pitch_offset_deg),
            base_yaw_offset_rad=np.radians(self.base_yaw_offset_deg),
        )
        ok = _execute_grasp(
            self.robot,
            grasp6d,
            pre6d,
            retreat6d,
            ready_cfg,
            dry_run=False,
            gripper_width_m=max(float(self.args.gripper_open_width), float(best.width) + 0.02),
            place_cfg=place_cfg,
            ready_joints=self.ready_joints,
        )
        return {"ok": bool(ok), "status": "real grasp executed" if ok else "real grasp attempted"}

    def jog_base(self, payload: dict) -> dict:
        if not self.args.enable_robot:
            return {"ok": False, "error": "base jog disabled; restart with --enable-robot"}
        if self.robot is None:
            return {"ok": False, "error": "robot is not connected"}
        if not self.infer_lock.acquire(blocking=False):
            return {"ok": False, "error": "robot or grasp estimator is busy"}

        self.busy = True
        try:
            place_cfg = build_place_config(self.cfg, self.args)
            joint_name = str(payload.get("joint", place_cfg.get("base_joint", "joint1")) or "joint1")
            delta_deg = float(payload.get("delta_deg", 0.0))
            duration = float(payload.get("duration_s", place_cfg.get("base_rotate_duration", 2.5)))
            margin_deg = float(payload.get("safety_margin_deg", place_cfg.get("base_safety_margin_deg", 5.0)))

            if not np.isfinite(delta_deg) or not np.isfinite(duration) or not np.isfinite(margin_deg):
                return {"ok": False, "error": "delta, duration, and safety margin must be finite"}
            if abs(delta_deg) < 1e-6:
                return {"ok": False, "error": "delta_deg must be non-zero"}
            if abs(delta_deg) > 180.0:
                return {"ok": False, "error": "delta_deg must be between -180 and 180"}
            if duration <= 0.0:
                return {"ok": False, "error": "duration_s must be positive"}
            if margin_deg < 0.0:
                return {"ok": False, "error": "safety_margin_deg must be non-negative"}

            idx = self.robot._joint_index(joint_name)
            lo, hi = self.robot._joint_limit(joint_name)
            before = self.robot.get_joint_positions()
            before_deg = float(np.degrees(before[idx]))
            direction = "negative" if delta_deg < 0.0 else "positive"
            ok = self.robot.rotate_base_relative(
                np.radians(abs(delta_deg)),
                duration=duration,
                direction=direction,
                safety_margin_rad=np.radians(margin_deg),
                joint_name=joint_name,
            )
            after = self.robot.get_joint_positions()
            after_deg = float(np.degrees(after[idx]))
            status = (
                f"base jog {joint_name} {delta_deg:+.1f}deg completed"
                if ok
                else f"base jog {joint_name} {delta_deg:+.1f}deg blocked or timed out"
            )
            self.status = status
            return {
                "ok": bool(ok),
                "status": status,
                "joint": joint_name,
                "direction": direction,
                "delta_deg": round(delta_deg, 3),
                "duration_s": round(duration, 3),
                "safety_margin_deg": round(margin_deg, 3),
                "before_deg": round(before_deg, 3),
                "after_deg": round(after_deg, 3),
                "limit_deg": [round(float(np.degrees(lo)), 3), round(float(np.degrees(hi)), 3)],
                "safe_limit_deg": [
                    round(float(np.degrees(lo) + margin_deg), 3),
                    round(float(np.degrees(hi) - margin_deg), 3),
                ],
            }
        except Exception as exc:
            traceback.print_exc()
            self.status = f"base jog failed: {exc}"
            return {"ok": False, "error": str(exc)}
        finally:
            self.busy = False
            self.infer_lock.release()


INDEX_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>reBot Grasp Web</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #101418; color: #eef3f8; }}
    header {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 12px 16px; background: #18202a; }}
    select, input, button {{ font: inherit; padding: 8px 10px; border-radius: 6px; border: 1px solid #44515f; background: #222b35; color: #eef3f8; }}
    input[type="number"] {{ width: 100%; min-width: 0; box-sizing: border-box; }}
    button {{ cursor: pointer; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    button.danger {{ background: #5b1e24; border-color: #9f3440; }}
    button.secondary {{ background: #1d2833; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 12px; padding: 12px; }}
    img {{ width: 100%; height: auto; background: #000; border: 1px solid #33404d; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #151b22; border: 1px solid #33404d; padding: 10px; border-radius: 6px; min-height: 260px; }}
    aside {{ display: flex; flex-direction: column; gap: 10px; }}
    .controls {{ background: #151b22; border: 1px solid #33404d; border-radius: 6px; padding: 10px; }}
    .control-title {{ font-weight: 700; margin-bottom: 8px; }}
    .control-row {{ display: grid; grid-template-columns: 120px repeat(3, minmax(0, 1fr)); gap: 6px; align-items: center; margin-bottom: 6px; }}
    .control-row label {{ color: #c7d2dd; }}
    .control-actions {{ display: flex; gap: 8px; margin-top: 8px; }}
    .status {{ color: #9ee493; }}
    .muted {{ color: #aab6c2; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <strong>reBot Grasp Web</strong>
    <button id="languageToggle" type="button" class="secondary" aria-pressed="false">English</button>
    <label data-i18n="targetClassLabel">目标类别</label>
    <select id="target">{options}</select>
    <button id="setTarget" data-i18n="setTarget">设置目标</button>
    <button id="infer" data-i18n="refreshGrasp">刷新抓取点</button>
    <button id="grasp" class="danger" data-i18n="realGrasp">真实抓取</button>
    <button id="readyPose" class="secondary" data-i18n="readyGrasp">预备抓取</button>
    <span class="muted" id="modeHint" data-i18n="modeHintInitial">自动显示 ordinary 抓取点；真实抓取需用 --enable-robot 启动</span>
  </header>
  <main>
    <section><img src="/stream.mjpg" alt="camera stream" data-i18n-alt="cameraStreamAlt"></section>
    <aside>
      <section class="controls">
        <div class="control-title" data-i18n="compensationTitle">补偿输入</div>
        <div class="control-row">
          <label data-i18n="gripperOffsetLabel">夹爪前/左/上(m)</label>
          <input id="forwardOffset" type="number" min="-0.15" max="0.15" step="0.001" value="0.000" title="夹爪前后补偿；正数沿接近方向前进" data-i18n-title="forwardOffsetTitle">
          <input id="lateralOffset" type="number" min="-0.15" max="0.15" step="0.001" value="0.000" title="夹爪左右补偿；正数沿夹爪 Y 轴" data-i18n-title="lateralOffsetTitle">
          <input id="verticalOffset" type="number" min="-0.15" max="0.15" step="0.001" value="0.000" title="夹爪上下补偿；正数沿夹爪 Z 轴" data-i18n-title="verticalOffsetTitle">
        </div>
        <div class="control-row">
          <label data-i18n="gripperRpyLabel">夹爪RPY(°)</label>
          <input id="rollOffset" type="number" min="-45" max="45" step="1" value="0.0" title="夹爪 roll 补偿" data-i18n-title="rollOffsetTitle">
          <input id="pitchOffset" type="number" min="-45" max="45" step="1" value="0.0" title="夹爪 pitch 补偿" data-i18n-title="pitchOffsetTitle">
          <input id="yawOffset" type="number" min="-45" max="45" step="1" value="0.0" title="夹爪 yaw 补偿" data-i18n-title="yawOffsetTitle">
        </div>
        <div class="control-row">
          <label data-i18n="cameraXyzLabel">相机XYZ(m)</label>
          <input id="cameraXOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="相机 X 外参补偿" data-i18n-title="cameraXOffsetTitle">
          <input id="cameraYOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="相机 Y 外参补偿" data-i18n-title="cameraYOffsetTitle">
          <input id="cameraZOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="相机 Z 外参补偿" data-i18n-title="cameraZOffsetTitle">
        </div>
        <div class="control-row">
          <label data-i18n="cameraRpyLabel">相机RPY(°)</label>
          <input id="cameraRollOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="相机 X roll 外参补偿" data-i18n-title="cameraRollOffsetTitle">
          <input id="cameraPitchOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="相机 Y pitch 外参补偿" data-i18n-title="cameraPitchOffsetTitle">
          <input id="cameraYawOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="相机 Z yaw 外参补偿" data-i18n-title="cameraYawOffsetTitle">
        </div>
        <div class="control-row">
          <label data-i18n="baseXyzLabel">基座XYZ(m)</label>
          <input id="baseXOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="机器人基座 X 外参补偿" data-i18n-title="baseXOffsetTitle">
          <input id="baseYOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="机器人基座 Y 外参补偿" data-i18n-title="baseYOffsetTitle">
          <input id="baseZOffset" type="number" min="-0.20" max="0.20" step="0.001" value="0.000" title="机器人基座 Z 外参补偿" data-i18n-title="baseZOffsetTitle">
        </div>
        <div class="control-row">
          <label data-i18n="baseRpyLabel">基座RPY(°)</label>
          <input id="baseRollOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="机器人基座 X roll 外参补偿" data-i18n-title="baseRollOffsetTitle">
          <input id="basePitchOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="机器人基座 Y pitch 外参补偿" data-i18n-title="basePitchOffsetTitle">
          <input id="baseYawOffset" type="number" min="-45" max="45" step="0.5" value="0.0" title="机器人基座 Z yaw 外参补偿" data-i18n-title="baseYawOffsetTitle">
        </div>
        <div class="control-actions">
          <button id="setOffset" data-i18n="setOffset">设置补偿</button>
        </div>
      </section>
      <section class="controls">
        <div class="control-title" data-i18n="baseJogTitle">底座电机调试</div>
        <div class="control-row">
          <label data-i18n="baseJogDegLabel">joint1 jog(°)</label>
          <input id="baseJogDeg" type="number" min="-180" max="180" step="1" value="-30" title="底座电机 joint1 相对角度；负数走负方向" data-i18n-title="baseJogDegTitle">
          <input id="baseJogDuration" type="number" min="0.2" max="10" step="0.1" value="2.5" title="底座电机运动时长，单位秒" data-i18n-title="baseJogDurationTitle">
          <input id="baseJogMargin" type="number" min="0" max="45" step="0.5" value="5.0" title="joint1 限位安全边距，单位度" data-i18n-title="baseJogMarginTitle">
        </div>
        <div class="control-actions">
          <button id="baseJogNeg">-30°</button>
          <button id="baseJogApply" data-i18n="baseJogApply">执行底座jog</button>
          <button id="baseJogPos">+30°</button>
        </div>
      </section>
      <div class="status" id="status" data-i18n="loading">loading...</div>
      <pre id="state"></pre>
    </aside>
  </main>
  <script>
    const I18N = {{
      zh: {{
        languageToggle: 'English',
        targetClassLabel: '目标类别',
        bestDetection: '最佳检测',
        setTarget: '设置目标',
        refreshGrasp: '刷新抓取点',
        realGrasp: '真实抓取',
        readyGrasp: '预备抓取',
        modeHintInitial: '自动显示 ordinary 抓取点；真实抓取需用 --enable-robot 启动',
        modeHintConnected: '自动更新 ordinary 抓取点；点击真实抓取会优先执行当前 ordinary 抓取点',
        modeHintDisconnected: '自动更新 ordinary 抓取点；真实抓取需重启脚本并加 --enable-robot',
        cameraStreamAlt: '相机画面',
        compensationTitle: '补偿输入',
        gripperOffsetLabel: '夹爪前/左/上(m)',
        gripperRpyLabel: '夹爪RPY(°)',
        cameraXyzLabel: '相机XYZ(m)',
        cameraRpyLabel: '相机RPY(°)',
        baseXyzLabel: '基座XYZ(m)',
        baseRpyLabel: '基座RPY(°)',
        setOffset: '设置补偿',
        baseJogTitle: '底座电机调试',
        baseJogDegLabel: 'joint1 jog(°)',
        baseJogApply: '执行底座jog',
        loading: 'loading...',
        forwardOffsetTitle: '夹爪前后补偿；正数沿接近方向前进',
        lateralOffsetTitle: '夹爪左右补偿；正数沿夹爪 Y 轴',
        verticalOffsetTitle: '夹爪上下补偿；正数沿夹爪 Z 轴',
        rollOffsetTitle: '夹爪 roll 补偿',
        pitchOffsetTitle: '夹爪 pitch 补偿',
        yawOffsetTitle: '夹爪 yaw 补偿',
        cameraXOffsetTitle: '相机 X 外参补偿',
        cameraYOffsetTitle: '相机 Y 外参补偿',
        cameraZOffsetTitle: '相机 Z 外参补偿',
        cameraRollOffsetTitle: '相机 X roll 外参补偿',
        cameraPitchOffsetTitle: '相机 Y pitch 外参补偿',
        cameraYawOffsetTitle: '相机 Z yaw 外参补偿',
        baseXOffsetTitle: '机器人基座 X 外参补偿',
        baseYOffsetTitle: '机器人基座 Y 外参补偿',
        baseZOffsetTitle: '机器人基座 Z 外参补偿',
        baseRollOffsetTitle: '机器人基座 X roll 外参补偿',
        basePitchOffsetTitle: '机器人基座 Y pitch 外参补偿',
        baseYawOffsetTitle: '机器人基座 Z yaw 外参补偿',
        baseJogDegTitle: '底座电机 joint1 相对角度；负数走负方向',
        baseJogDurationTitle: '底座电机运动时长，单位秒',
        baseJogMarginTitle: 'joint1 限位安全边距，单位度',
        numberExampleError: '请输入数字，例如 -0.010 或 0.080',
        numberRequired: '请输入数字',
        confirmGrasp: '确认执行真实抓取？',
        refreshingGrasp: '正在计算 GraspNet 抓取点...',
        errorPrefix: 'ERROR: ',
        failed: 'failed',
        stateErrorPrefix: 'state error: ',
        autoGraspLabel: '自动 GraspNet',
        robotConnected: '机械臂已连接',
        robotDisconnected: '机械臂未连接',
        baseJogDurationName: '运动时长',
        baseJogMarginName: '限位安全边距'
      }},
      en: {{
        languageToggle: '中文',
        targetClassLabel: 'Target class',
        bestDetection: 'Best detection',
        setTarget: 'Set target',
        refreshGrasp: 'Refresh grasp',
        realGrasp: 'Real grasp',
        readyGrasp: 'Prepare grasp',
        modeHintInitial: 'Ordinary grasp points update automatically; start with --enable-robot for real grasp.',
        modeHintConnected: 'Ordinary grasp points update automatically; Real grasp uses the current ordinary grasp first.',
        modeHintDisconnected: 'Ordinary grasp points update automatically; restart with --enable-robot for real grasp.',
        cameraStreamAlt: 'camera stream',
        compensationTitle: 'Compensation',
        gripperOffsetLabel: 'Gripper F/L/U (m)',
        gripperRpyLabel: 'Gripper RPY (deg)',
        cameraXyzLabel: 'Camera XYZ (m)',
        cameraRpyLabel: 'Camera RPY (deg)',
        baseXyzLabel: 'Base XYZ (m)',
        baseRpyLabel: 'Base RPY (deg)',
        setOffset: 'Set compensation',
        baseJogTitle: 'Base motor debug',
        baseJogDegLabel: 'joint1 jog (deg)',
        baseJogApply: 'Run base jog',
        loading: 'loading...',
        forwardOffsetTitle: 'Forward/back gripper compensation; positive moves along the approach direction',
        lateralOffsetTitle: 'Left/right gripper compensation; positive along the gripper Y axis',
        verticalOffsetTitle: 'Up/down gripper compensation; positive along the gripper Z axis',
        rollOffsetTitle: 'Gripper roll compensation',
        pitchOffsetTitle: 'Gripper pitch compensation',
        yawOffsetTitle: 'Gripper yaw compensation',
        cameraXOffsetTitle: 'Camera X extrinsic compensation',
        cameraYOffsetTitle: 'Camera Y extrinsic compensation',
        cameraZOffsetTitle: 'Camera Z extrinsic compensation',
        cameraRollOffsetTitle: 'Camera X roll extrinsic compensation',
        cameraPitchOffsetTitle: 'Camera Y pitch extrinsic compensation',
        cameraYawOffsetTitle: 'Camera Z yaw extrinsic compensation',
        baseXOffsetTitle: 'Robot base X extrinsic compensation',
        baseYOffsetTitle: 'Robot base Y extrinsic compensation',
        baseZOffsetTitle: 'Robot base Z extrinsic compensation',
        baseRollOffsetTitle: 'Robot base X roll extrinsic compensation',
        basePitchOffsetTitle: 'Robot base Y pitch extrinsic compensation',
        baseYawOffsetTitle: 'Robot base Z yaw extrinsic compensation',
        baseJogDegTitle: 'Base motor joint1 relative angle; negative moves in the negative direction',
        baseJogDurationTitle: 'Base motor motion duration in seconds',
        baseJogMarginTitle: 'joint1 safety margin from limits, in degrees',
        numberExampleError: 'must be a number, for example -0.010 or 0.080',
        numberRequired: 'must be a number',
        confirmGrasp: 'Run a real robot grasp?',
        refreshingGrasp: 'Calculating GraspNet grasp points...',
        errorPrefix: 'ERROR: ',
        failed: 'failed',
        stateErrorPrefix: 'state error: ',
        autoGraspLabel: 'Auto GraspNet',
        robotConnected: 'Robot connected',
        robotDisconnected: 'Robot not connected',
        baseJogDurationName: 'duration',
        baseJogMarginName: 'safety margin'
      }}
    }};
    const LANG_KEY = 'rebotGraspWebLanguage';
    let currentLang = localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'zh';
    let latestRobotConnected = null;

    const target = document.getElementById('target');
    const languageToggle = document.getElementById('languageToggle');
    const statusEl = document.getElementById('status');
    const stateEl = document.getElementById('state');
    const inferBtn = document.getElementById('infer');
    const graspBtn = document.getElementById('grasp');
    const readyBtn = document.getElementById('readyPose');
    const modeHint = document.getElementById('modeHint');
    const baseJogDeg = document.getElementById('baseJogDeg');
    const baseJogDuration = document.getElementById('baseJogDuration');
    const baseJogMargin = document.getElementById('baseJogMargin');
    const baseJogButtons = [
      document.getElementById('baseJogNeg'),
      document.getElementById('baseJogApply'),
      document.getElementById('baseJogPos')
    ];
    const compensationInputs = {{
      forward_m: {{ el: document.getElementById('forwardOffset'), digits: 3 }},
      lateral_m: {{ el: document.getElementById('lateralOffset'), digits: 3 }},
      vertical_m: {{ el: document.getElementById('verticalOffset'), digits: 3 }},
      roll_deg: {{ el: document.getElementById('rollOffset'), digits: 1 }},
      pitch_deg: {{ el: document.getElementById('pitchOffset'), digits: 1 }},
      yaw_deg: {{ el: document.getElementById('yawOffset'), digits: 1 }},
      camera_x_m: {{ el: document.getElementById('cameraXOffset'), digits: 3 }},
      camera_y_m: {{ el: document.getElementById('cameraYOffset'), digits: 3 }},
      camera_z_m: {{ el: document.getElementById('cameraZOffset'), digits: 3 }},
      camera_roll_deg: {{ el: document.getElementById('cameraRollOffset'), digits: 1 }},
      camera_pitch_deg: {{ el: document.getElementById('cameraPitchOffset'), digits: 1 }},
      camera_yaw_deg: {{ el: document.getElementById('cameraYawOffset'), digits: 1 }},
      base_x_m: {{ el: document.getElementById('baseXOffset'), digits: 3 }},
      base_y_m: {{ el: document.getElementById('baseYOffset'), digits: 3 }},
      base_z_m: {{ el: document.getElementById('baseZOffset'), digits: 3 }},
      base_roll_deg: {{ el: document.getElementById('baseRollOffset'), digits: 1 }},
      base_pitch_deg: {{ el: document.getElementById('basePitchOffset'), digits: 1 }},
      base_yaw_deg: {{ el: document.getElementById('baseYawOffset'), digits: 1 }}
    }};
    let compensationDirty = false;
    let inferRequestInFlight = false;
    let graspRequestInFlight = false;
    let readyRequestInFlight = false;

    function tr(key) {{
      return (I18N[currentLang] && I18N[currentLang][key]) || I18N.zh[key] || key;
    }}

    function updateModeHint(robotConnected) {{
      latestRobotConnected = robotConnected === null || robotConnected === undefined ? null : Boolean(robotConnected);
      modeHint.textContent = latestRobotConnected === null
        ? tr('modeHintInitial')
        : (latestRobotConnected ? tr('modeHintConnected') : tr('modeHintDisconnected'));
    }}

    function applyLanguage(lang) {{
      currentLang = lang === 'en' ? 'en' : 'zh';
      localStorage.setItem(LANG_KEY, currentLang);
      document.documentElement.lang = currentLang === 'en' ? 'en' : 'zh-CN';
      for (const el of document.querySelectorAll('[data-i18n]')) {{
        el.textContent = tr(el.dataset.i18n);
      }}
      for (const el of document.querySelectorAll('[data-i18n-title]')) {{
        el.title = tr(el.dataset.i18nTitle);
      }}
      for (const el of document.querySelectorAll('[data-i18n-alt]')) {{
        el.alt = tr(el.dataset.i18nAlt);
      }}
      languageToggle.textContent = tr('languageToggle');
      languageToggle.setAttribute('aria-pressed', currentLang === 'en' ? 'true' : 'false');
      updateModeHint(latestRobotConnected);
    }}

    languageToggle.onclick = () => applyLanguage(currentLang === 'en' ? 'zh' : 'en');
    applyLanguage(currentLang);

    for (const cfg of Object.values(compensationInputs)) {{
      cfg.el.addEventListener('input', () => {{
        compensationDirty = true;
      }});
    }}

    function applyCompensation(comp) {{
      for (const [key, cfg] of Object.entries(compensationInputs)) {{
        const value = comp[key];
        if (typeof value === 'number') {{
          cfg.el.value = value.toFixed(cfg.digits);
        }}
      }}
    }}

    function readCompensationValue(key, cfg) {{
      const raw = String(cfg.el.value).trim().replace('−', '-');
      const value = Number(raw);
      if (!Number.isFinite(value)) {{
        throw new Error(key + ' ' + tr('numberExampleError'));
      }}
      return value;
    }}

    function readNumberInput(el, name) {{
      const raw = String(el.value).trim().replace('−', '-');
      const value = Number(raw);
      if (!Number.isFinite(value)) {{
        throw new Error(name + ' ' + tr('numberRequired'));
      }}
      return value;
    }}

    async function post(path, body={{}}) {{
      const res = await fetch(path, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
      }});
      const data = await res.json();
      statusEl.textContent = data.ok ? 'OK' : (tr('errorPrefix') + (data.error || data.status || tr('failed')));
      stateEl.textContent = JSON.stringify(data, null, 2);
      return data;
    }}

    async function refreshGraspNow() {{
      if (inferRequestInFlight) return;
      inferRequestInFlight = true;
      inferBtn.disabled = true;
      try {{
        await post('/infer');
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }} finally {{
        inferRequestInFlight = false;
      }}
    }}

    async function setTargetAndQueueGrasp() {{
      try {{
        const data = await post('/target', {{class_name: target.value}});
        if (data.ok) {{
          statusEl.textContent = tr('refreshingGrasp');
          refreshGraspNow();
        }}
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }}
    }}

    target.onchange = setTargetAndQueueGrasp;
    document.getElementById('setTarget').onclick = setTargetAndQueueGrasp;
    document.getElementById('setOffset').onclick = async () => {{
      try {{
        const payload = {{}};
        for (const [key, cfg] of Object.entries(compensationInputs)) {{
          payload[key] = readCompensationValue(key, cfg);
        }}
        const data = await post('/compensation', payload);
        if (data.ok) {{
          compensationDirty = false;
          applyCompensation(data.compensation || {{}});
        }}
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }}
    }};
    async function jogBase(deltaOverride) {{
      try {{
        const delta = deltaOverride === undefined ? readNumberInput(baseJogDeg, 'joint1 jog') : deltaOverride;
        const payload = {{
          delta_deg: delta,
          duration_s: readNumberInput(baseJogDuration, tr('baseJogDurationName')),
          safety_margin_deg: readNumberInput(baseJogMargin, tr('baseJogMarginName'))
        }};
        const data = await post('/base_jog', payload);
        if (data.ok) {{
          baseJogDeg.value = Number(data.delta_deg || delta).toFixed(1);
        }}
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }}
    }}
    document.getElementById('baseJogNeg').onclick = () => {{
      try {{
        const step = Math.abs(readNumberInput(baseJogDeg, 'joint1 jog') || 30);
        jogBase(-step);
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }}
    }};
    document.getElementById('baseJogApply').onclick = () => jogBase();
    document.getElementById('baseJogPos').onclick = () => {{
      try {{
        const step = Math.abs(readNumberInput(baseJogDeg, 'joint1 jog') || 30);
        jogBase(step);
      }} catch (e) {{
        statusEl.textContent = tr('errorPrefix') + e.message;
      }}
    }};
    inferBtn.onclick = () => refreshGraspNow();
    document.getElementById('grasp').onclick = async () => {{
      if (!confirm(tr('confirmGrasp'))) return;
      graspRequestInFlight = true;
      graspBtn.disabled = true;
      try {{
        await post('/grasp');
      }} finally {{
        graspRequestInFlight = false;
      }}
    }};
    readyBtn.onclick = async () => {{
      readyRequestInFlight = true;
      readyBtn.disabled = true;
      try {{
        await post('/ready');
      }} finally {{
        readyRequestInFlight = false;
      }}
    }};

    async function refresh() {{
      try {{
        const res = await fetch('/state');
        const data = await res.json();
        if (document.activeElement !== target) target.value = data.target_class || '';
        const comp = data.compensation || {{}};
        if (!compensationDirty) {{
          applyCompensation(comp);
        }}
        const autoText = data.auto_graspnet ? (tr('autoGraspLabel') + ': ' + data.graspnet_interval_s + 's') : tr('autoGraspLabel') + ': off';
        const ordinaryText = data.ordinary_grasp
          ? ('ordinary: ' + data.ordinary_grasp.target + ' ' + Number(data.ordinary_grasp.score || 0).toFixed(2))
          : (data.ordinary_status || 'ordinary: waiting');
        const robotText = data.robot_connected ? tr('robotConnected') : tr('robotDisconnected');
        statusEl.textContent = autoText + ' | ' + ordinaryText + ' | ' + robotText + ' | ' + data.status + ' | ' + data.target_status;
        inferBtn.disabled = data.busy || inferRequestInFlight;
        graspBtn.disabled = !data.robot_connected || graspRequestInFlight;
        readyBtn.disabled = !data.robot_connected || data.busy || readyRequestInFlight || graspRequestInFlight;
        for (const btn of baseJogButtons) {{
          btn.disabled = !data.robot_connected || data.busy || graspRequestInFlight || readyRequestInFlight;
        }}
        updateModeHint(data.robot_connected);
        stateEl.textContent = JSON.stringify(data, null, 2);
      }} catch (e) {{
        statusEl.textContent = tr('stateErrorPrefix') + e;
      }}
    }}
    setInterval(refresh, 700);
    refresh();
  </script>
</body>
</html>
"""


class GraspWebHandler(BaseHTTPRequestHandler):
    app: GraspWebApp

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(self._index_html())
        elif path == "/state":
            self._send_json(self.app.state())
        elif path == "/stream.mjpg":
            self._stream_mjpeg()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/target":
            payload = self._read_payload()
            self._send_json(self.app.set_target(str(payload.get("class_name", ""))))
        elif path == "/compensation":
            self._send_json(self.app.set_compensation(self._read_payload()))
        elif path == "/offset":
            payload = self._read_payload()
            self._send_json(self.app.set_forward_offset(float(payload.get("offset_m", 0.0))))
        elif path == "/base_jog":
            self._send_json(self.app.jog_base(self._read_payload()))
        elif path == "/ready":
            self._send_json(self.app.prepare_grasp())
        elif path == "/infer":
            self._send_json(self.app.run_grasp(execute=False))
        elif path == "/grasp":
            self._send_json(self.app.execute_latest_grasp())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def _read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return json.loads(body.decode("utf-8") or "{}")
        parsed = parse_qs(body.decode("utf-8"))
        return {key: values[-1] for key, values in parsed.items()}

    def _index_html(self) -> str:
        options = ['<option value="" data-i18n="bestDetection">最佳检测</option>']
        for name in self.app.class_names:
            selected = " selected" if name == self.app.target_class else ""
            escaped = html.escape(name, quote=True)
            options.append(f'<option value="{escaped}"{selected}>{escaped}</option>')
        return INDEX_TEMPLATE.format(options="\n".join(options))

    def _send_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not self.app.stop_event.is_set():
                frame = self.app.get_jpeg()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    args = parse_args()
    app = GraspWebApp(args)
    app.start()

    handler_cls = type("BoundGraspWebHandler", (GraspWebHandler,), {"app": app})
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    server.daemon_threads = True
    url_host = "localhost" if args.host in {"0.0.0.0", ""} else args.host
    print(f"\nOpen web UI: http://{url_host}:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Web] stopping...")
    finally:
        server.shutdown()
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
