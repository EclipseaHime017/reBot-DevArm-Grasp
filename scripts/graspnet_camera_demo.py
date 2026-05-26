from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def _prepare_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        sys.path.insert(0, str(GRASPNET_ROOT / subdir))
    sys.path.insert(0, str(GRASPNET_ROOT))


_prepare_imports()

from cameraws.drivers.camera import make_camera  # noqa: E402
from cameraws.utils.camera_utils import configure_camera, load_config  # noqa: E402
import cameraws.utils.graspnet_utils as graspnet_utils  # noqa: E402
from cameraws.utils.yolo_utils import detect_objects, load_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal live full-scene GraspNet camera demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D 显示的候选集合：final=最终候选，bbox=宽度过滤前，pre-bbox=bbox过滤前",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = configure_camera(load_config(args.config), args)
    graspnet_cfg = cfg.get("graspnet", {})
    checkpoint = graspnet_utils.resolve_checkpoint_path(graspnet_cfg.get("checkpoint", "checkpoint-rs.tar"))
    num_point = int(graspnet_cfg.get("num_point", 20000))
    collision_thresh = float(graspnet_cfg.get("collision_thresh", 0.01))
    min_depth = float(graspnet_cfg.get("min_depth", 0.05))
    max_depth = float(graspnet_cfg.get("max_depth", 2.0))
    top_k = int(graspnet_cfg.get("top_k", 50))
    target_class = graspnet_cfg.get("target_class")
    target_margin_px = int(graspnet_cfg.get("target_margin_px", 12))
    target_expand_ratio = float(graspnet_cfg.get("target_expand_ratio", 1.0))

    net = graspnet_utils.build_net(str(checkpoint), graspnet_utils.DEFAULT_NUM_VIEW)
    yolo_model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)
    cam = make_camera(cfg)
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None
    status = "warming up camera..."
    target_status = "target detector warming up..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    last_detections = []
    selected_target = None
    frame_index = 0
    window_name = "GraspNet Live Camera"

    print(
        "Using camera: "
        f"{cfg['camera']['type']} "
        f"{cfg['camera'].get('color_width')}x{cfg['camera'].get('color_height')}@{cfg['camera'].get('fps')}"
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg["camera"].get("color_width", 1280)), int(cfg["camera"].get("color_height", 720)))

    try:
        cam.open()
        cam.warm_up(graspnet_utils.DEFAULT_WARMUP_FRAMES)
        print("Camera intrinsics:")
        print(cam.K)
        print("Press G or SPACE to infer current frame. Press Q or ESC to quit.")
        print(f"YOLO bbox post-filter mode: target={target_class or 'best detection'}")

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                cv2.imshow(window_name, graspnet_utils.draw_status(np.zeros((720, 1280, 3), dtype=np.uint8), "waiting for frames..."))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            frame_index += 1
            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = graspnet_utils.select_target(last_detections, target_class)
                    target_status = graspnet_utils.target_status_text(selected_target, last_detections, target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = graspnet_utils.draw_status(last_display, status, target_status, frozen=True)
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(color_bgr, last_detections, selected_target, target_class)
                display = graspnet_utils.draw_status(display_base, status, target_status)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue
            if key in (ord("g"), ord("G"), ord(" ")):
                try:
                    result = graspnet_utils.infer_frame(
                        net,
                        color_bgr,
                        depth_mm,
                        cam.K,
                        num_point=num_point,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        collision_thresh=collision_thresh,
                        voxel_size=graspnet_utils.DEFAULT_VOXEL_SIZE,
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=target_class,
                        target_margin_px=target_margin_px,
                        target_expand_ratio=target_expand_ratio,
                    )
                    status = result.status
                    target_status = result.target_status
                    last_detections = result.detections
                    selected_target = result.selected_target
                    print(status)
                    display_base = graspnet_utils.draw_detections_overlay(
                        color_bgr,
                        last_detections,
                        selected_target,
                        target_class,
                    )
                    graspnet_utils.draw_best_grasp_projection(display_base, result.best, cam.K)
                    last_display = display_base
                    frozen = True

                    vis_grasps = graspnet_utils.visualization_grasps(result, args.open3d_grasps)
                    if vis is None and len(vis_grasps) > 0:
                        vis = graspnet_utils.Open3DGraspWindow("GraspNet Grasps", top_k)
                    if vis is not None:
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"Open3D {args.open3d_grasps} candidates={len(vis_grasps)}")
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
