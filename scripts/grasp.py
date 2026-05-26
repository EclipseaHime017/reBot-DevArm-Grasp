"""
grasp.py - 基于 GraspNet 的机械臂视觉夹取主程序
================================================
流程：
  1. 初始化机械臂、夹爪和 RGB-D 相机，移动到预备位
  2. YOLO 选择目标，GraspNet 在当前 RGB-D 帧上估计 6D 夹取姿态
  3. G/SPACE 键：冻结当前帧并执行夹取（--dry-run 只打印坐标）
  4. R 键：恢复实时预览
  5. Q/ESC 键：退出，释放夹爪并回零位

用法：
  conda activate seeed
  python scripts/grasp.py --dry-run
  python scripts/grasp.py --target-class cup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEED_ROOT = PROJECT_ROOT.parent
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def _prepare_imports() -> None:
    for path in (SEEED_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        sys.path.insert(0, str(GRASPNET_ROOT / subdir))
    sys.path.insert(0, str(GRASPNET_ROOT))


_prepare_imports()

from cameraws.drivers.camera import make_camera  # noqa: E402
from cameraws.drivers.robot.rebot_arm import RebotArm  # noqa: E402
from cameraws.drivers.robot import rebot_arm  # noqa: E402
import cameraws.utils.graspnet_utils as graspnet_utils  # noqa: E402
from cameraws.utils.camera_utils import configure_camera, load_config, load_hand_eye  # noqa: E402
from cameraws.utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    graspnet_rotation_to_rebot_tcp_rotation,
    rotation_matrix_to_euler_zyx,
)
from cameraws.utils.yolo_utils import (  # noqa: E402
    YoloDetection,
    detect_objects,
    load_yolo as load_yolo_from_config,
)
from graspnetAPI import Grasp, GraspGroup  # noqa: E402


def _move_ready(robot: RebotArm, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    robot.move_to(
        float(ready_cfg.get("x", 0.25)),
        float(ready_cfg.get("y", 0.0)),
        float(ready_cfg.get("z", 0.35)),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg.get("pitch", 1.2)),
        float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    robot.wait_motion(duration)

def _execute_grasp(
    robot: RebotArm,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    xr, yr, zr, rxr, ryr, rzr = retreat6d

    print(f"[Grasp] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    print(f"[Grasp] retreat  xyz=({xr:+.3f},{yr:+.3f},{zr:+.3f}) rpy=({rxr:+.3f},{ryr:+.3f},{rzr:+.3f})")

    if dry_run:
        print("[Grasp] --dry-run: 跳过机械臂执行")
        return False

    print("[Grasp] 打开夹爪...")
    robot.open_gripper()

    print("[Grasp] 移动到预夹取位...")
    if not robot.move_to(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] 预夹取 IK 失败，中止")
        return False
    robot.wait_motion(2.0)

    print("[Grasp] 移动到夹取位...")
    if not robot.move_to(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] 夹取 IK 失败，中止")
        return False
    robot.wait_motion(1.5)

    print("[Grasp] 夹取中...")
    ok = robot.grasp()
    print("[Grasp] 夹取成功，力控保持中" if ok else "[Grasp] 空夹取")

    print("[Grasp] 退回预夹取位...")
    if robot.move_to(xr, yr, zr, rxr, ryr, rzr, duration=1.5):
        robot.wait_motion(1.5)

    print("[Grasp] 返回预备位...")
    _move_ready(robot, ready_cfg)
    return ok


def _print_grasp(grasp: Grasp) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix))
    print("\n[G] GraspNet 最佳夹取:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _rank_grasps(grasps: GraspGroup) -> GraspGroup:
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    return float(pose6d[2]) >= float(min_z)


def _select_executable_grasp(
    robot: RebotArm,
    grasps: GraspGroup,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: float,
) -> Optional[tuple[Grasp, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    ranked = _rank_grasps(grasps)
    skipped_low = 0
    skipped_ik = 0
    worst_err = 0.0

    for idx in range(len(ranked)):
        grasp = ranked[idx]
        grasp6d, pre6d, retreat6d = graspnet_utils.grasp_to_base_poses(
            grasp,
            T_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            insertion_depth_m,
        )
        if not (_pose_z_ok(pre6d, min_base_z_m) and _pose_z_ok(grasp6d, min_base_z_m)):
            skipped_low += 1
            continue

        pre_ok, pre_err = robot.check_ik(*pre6d)
        grasp_ok, grasp_err = robot.check_ik(*grasp6d) if pre_ok else (False, pre_err)
        worst_err = max(worst_err, pre_err, grasp_err)
        if pre_ok and grasp_ok:
            print(f"[G] 选择可执行候选 rank={idx + 1}/{len(ranked)} score={grasp.score:.4f}")
            if skipped_low or skipped_ik:
                print(f"[G] 跳过低高度={skipped_low} IK不可达={skipped_ik}")
            return grasp, grasp6d, pre6d, retreat6d
        skipped_ik += 1

    print(f"[G] 没有 IK 可达候选：低高度={skipped_low} IK不可达={skipped_ik} max_err={worst_err:.4f}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 GraspNet 的机械臂夹取主程序")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(GRASPNET_ROOT / "checkpoints" / "checkpoint-rs.tar"))
    parser.add_argument("--dry-run", action="store_true", help="只估计姿态，不移动机械臂")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05, help="meters")
    parser.add_argument("--max-depth", type=float, default=2.0, help="meters")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--target-margin-px", type=int, default=None)
    parser.add_argument("--target-expand-ratio", type=float, default=None, help="YOLO bbox 后处理筛选膨胀比例")
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO，全场景 GraspNet")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=None)
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="meters")
    parser.add_argument("--retreat-offset", type=float, default=None, help="meters")
    parser.add_argument("--min-base-z", type=float, default=None, help="base坐标系下允许执行的最低TCP高度，单位米")
    parser.add_argument("--no-open3d", action="store_true", help="按 G 推理后不打开 Open3D 3D 夹取可视化")
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D 显示的候选集合：final=最终可执行候选，bbox=宽度过滤前，pre-bbox=bbox过滤前",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_config(Path(args.config)), args)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    pregrasp_offset_m = float(args.pregrasp_offset if args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset_m = float(args.retreat_offset if args.retreat_offset is not None else pregrasp_offset_m)
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    min_base_z_m = float(args.min_base_z if args.min_base_z is not None else grasp_cfg.get("min_base_z_m", 0.03))
    graspnet_cfg = cfg.get("graspnet", {})
    target_expand_ratio = float(
        args.target_expand_ratio
        if args.target_expand_ratio is not None
        else graspnet_cfg.get("target_expand_ratio", 1.0)
    )

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
    )
    robot.connect(enable=True)
    robot.init_gripper()

    print("[Robot] 移动到预备位置...")
    _move_ready(robot, ready_cfg)

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        print("[WARN] 手眼标定不可用或非 eye_in_hand，夹取执行将被禁用")
        T_hand_eye = None

    print("=== 加载模型 ===")
    yolo_model, yolo_opts = load_yolo_from_config(
        cfg,
        project_root=PROJECT_ROOT,
        no_yolo=args.no_yolo,
        model_override=args.yolo_model,
        device_override=args.yolo_device,
        conf_override=args.yolo_conf,
        iou_override=args.yolo_iou,
        infer_every_override=args.infer_every_live,
    )
    net = graspnet_utils.build_net(args.checkpoint, args.num_view)

    cam_cfg = cfg["camera"]
    print(f"=== 初始化相机: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    cam = make_camera(cfg)

    last_detections: list[YoloDetection] = []
    selected_target: Optional[Any] = None
    last_target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
    status = "warming up camera..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    window_name = "Main - GraspNet Grasp"
    top_k = int(cfg.get("graspnet", {}).get("top_k", 50))
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cam_cfg.get("color_width", 1280)), int(cam_cfg.get("color_height", 720)))
    print("\n[Keys] G/SPACE=GraspNet夹取  R=恢复  Q/ESC=退出\n")

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(K)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = graspnet_utils.select_target(last_detections, args.target_class)
                    last_target_status = graspnet_utils.target_status_text(selected_target, last_detections, args.target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(color_bgr, last_detections, selected_target, args.target_class)
                display = graspnet_utils.draw_status(
                    display_base,
                    f"LIVE {fps_value:.1f}fps | {status}",
                    last_target_status,
                    title="Main - GraspNet Grasp",
                )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print("\n[G] 采帧并运行 GraspNet...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] 采帧失败")
                    continue

                try:
                    result = graspnet_utils.infer_frame(
                        net,
                        snap_color,
                        snap_depth,
                        K,
                        num_point=args.num_point,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        collision_thresh=args.collision_thresh,
                        voxel_size=args.voxel_size,
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=args.target_class,
                        target_margin_px=int(
                            args.target_margin_px
                            if args.target_margin_px is not None
                            else graspnet_cfg.get("target_margin_px", 12)
                        ),
                        target_expand_ratio=target_expand_ratio,
                        max_grasp_width_m=rebot_arm._G_MAX_DIST_M,
                    )
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                status = result.status
                last_target_status = result.target_status
                last_detections = result.detections
                selected_target = result.selected_target
                vis_grasps = graspnet_utils.visualization_grasps(result, args.open3d_grasps)

                print(f"[G] {status}")
                if not args.no_open3d:
                    try:
                        if vis is None:
                            vis = graspnet_utils.Open3DGraspWindow("GraspNet Grasps", top_k)
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"[G] Open3D 显示 {args.open3d_grasps} candidates={len(vis_grasps)}")
                    except Exception as exc:
                        print(f"[G] Open3D 可视化失败: {exc}")
                        if vis is not None:
                            vis.close()
                            vis = None

                if result.best is None:
                    print("[G] 未找到有效 GraspNet 夹取候选")
                    continue
                frozen = True
                display_base = snap_color
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(snap_color, last_detections, selected_target, args.target_class)
                snap_display = graspnet_utils.draw_status(
                    display_base,
                    f"SNAPSHOT | {status}",
                    last_target_status,
                    frozen=True,
                    title="Main - GraspNet Grasp",
                )
                last_display = snap_display

                if T_hand_eye is None:
                    graspnet_utils.draw_best_grasp_projection(snap_display, result.best, K)
                    last_display = snap_display
                    print("[G] 手眼标定不可用，无法执行夹取")
                    continue

                T_cam2base = robot.get_tcp_pose() @ T_hand_eye
                selected = _select_executable_grasp(
                    robot,
                    result.grasps,
                    T_cam2base,
                    pregrasp_offset_m,
                    retreat_offset_m,
                    insertion_depth_m,
                    min_base_z_m,
                )
                if selected is None:
                    print(f"[G] 没有满足 min_base_z={min_base_z_m:.3f}m 且 IK 可达的夹取候选，跳过执行")
                    continue
                best, grasp6d, pre6d, retreat6d = selected

                _print_grasp(best)
                graspnet_utils.draw_best_grasp_projection(snap_display, best, K)
                last_display = snap_display

                _execute_grasp(
                    robot,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    ready_cfg,
                    dry_run=args.dry_run,
                )

            if vis is not None and not vis.poll():
                vis.close()
                vis = None

    finally:
        print("\n[退出] 释放夹爪并回零...")
        try:
            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")
        robot.disconnect()
        try:
            cam.close()
        except Exception:
            pass
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()
        print("已退出。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
