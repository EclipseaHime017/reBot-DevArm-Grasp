"""
Eye-in-Hand 手眼标定 — 数据采集与计算（Gemini2 + reBotArm）

【模式】
  自动模式（默认）：机械臂自动遍历 50 个预设姿态，到位后若识别到 ArUco
                     则自动采集，超时则跳过该姿态。
  手动模式（--manual）：重力补偿控制，用户手动推动机械臂到任意位置，
                        放手后臂自动锁定，按 Enter 采集。

【布置方式】
  相机装在机械臂末端（随末端运动）
  ArUco 标记贴在工作台固定位置（不动）

【用法】
    cd /home/seeed/Downloads/rebot_grasp
    python scripts/collect_handeye_eih.py           # 自动模式
    python scripts/collect_handeye_eih.py --manual  # 手动重力补偿模式
"""

import os
import sys
import math
import threading
import argparse
import queue
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from drivers.camera import make_camera
from drivers.robot.rebot_arm import RebotArm, ensure_rebot_sdk_in_syspath
from calibration.hand_eye import CalibMode, HandEyeCalibrator


# ==========================================
# 预设标定姿态（笛卡尔空间，单位：米/弧度）
# (x, y, z, roll, pitch, yaw)
# pitch > 0 = 末端朝斜下方（相机俯视 ArUco）
# ==========================================
CALIB_POSES_XYZ = [
    # ── 中心区域，大 pitch，yaw 扫描 ──
    (0.28, -0.16, 0.26, -0.30, 0.80, -0.90),
    (0.28, -0.08, 0.26,  0.30, 0.80, -0.45),
    (0.28,  0.00, 0.26, -0.30, 0.80,  0.00),
    (0.28,  0.08, 0.26,  0.30, 0.80,  0.45),
    (0.28,  0.16, 0.26, -0.30, 0.80,  0.90),
    # ── 中心区域，中 pitch ──
    (0.27, -0.16, 0.31,  0.30, 0.55, -0.90),
    (0.27, -0.08, 0.31, -0.30, 0.55, -0.45),
    (0.27,  0.00, 0.31,  0.30, 0.55,  0.00),
    (0.27,  0.08, 0.31, -0.30, 0.55,  0.45),
    (0.27,  0.16, 0.31,  0.30, 0.55,  0.90),
    # ── 中心区域，小 pitch ──
    (0.26, -0.14, 0.34, -0.40, 0.35, -0.80),
    (0.26,  0.00, 0.34,  0.40, 0.35,  0.00),
    (0.26,  0.14, 0.34, -0.40, 0.35,  0.80),
    # ── 偏前方，yaw ±1 rad ──
    (0.37,  0.00, 0.27,  0.00, 0.65,  0.00),
    (0.37,  0.00, 0.27,  0.00, 0.65,  1.00),
    (0.37,  0.00, 0.27,  0.00, 0.65, -1.00),
    (0.37,  0.08, 0.27,  0.50, 0.65,  0.50),
    (0.37, -0.08, 0.27, -0.50, 0.65, -0.50),
    # ── 侧向，x 大 ──
    (0.33,  0.18, 0.27,  0.50, 0.50,  0.55),
    (0.33, -0.18, 0.27, -0.50, 0.50, -0.55),
    # ── 侧向，y 大 ──
    (0.20,  0.22, 0.28,  0.60, 0.40,  0.70),
    (0.20, -0.22, 0.28, -0.60, 0.40, -0.70),
    # ── 斜前偏 y，roll 大 ──
    (0.24, -0.20, 0.31,  0.70, 0.45, -1.00),
    (0.24,  0.20, 0.31, -0.70, 0.45,  1.00),
    (0.25, -0.15, 0.29, -0.60, 0.62, -0.50),
    (0.25,  0.15, 0.29,  0.60, 0.62,  0.50),
    # ── 高位 ──
    (0.21, -0.09, 0.40, -0.40, 0.25, -0.60),
    (0.21,  0.00, 0.40,  0.40, 0.25,  0.00),
    (0.21,  0.09, 0.40, -0.40, 0.25,  0.60),
    (0.20, -0.09, 0.40,  0.40, 0.28, -0.60),
    (0.20,  0.09, 0.40, -0.40, 0.28,  0.60),
    # ── 低位 ──
    (0.30, -0.10, 0.24,  0.40, 0.70, -0.70),
    (0.30,  0.00, 0.24,  0.00, 0.75,  0.00),
    (0.30,  0.10, 0.24, -0.40, 0.70,  0.70),
    # ── roll 极值 ──
    (0.26,  0.12, 0.30,  0.80, 0.50,  0.30),
    (0.26, -0.12, 0.30, -0.80, 0.50, -0.30),
    # ── 大 roll + 大 yaw 组合（旋转多样性补充）──
    (0.29, -0.10, 0.28,  0.90, 0.60, -0.40),
    (0.29,  0.10, 0.28, -0.90, 0.60,  0.40),
    (0.28, -0.18, 0.30,  0.85, 0.55, -0.80),
    (0.28,  0.18, 0.30, -0.85, 0.55,  0.80),
    # ── 前伸 + 不同 roll/yaw 组合 ──
    (0.35, -0.12, 0.30,  0.60, 0.58, -0.70),
    (0.35,  0.12, 0.30, -0.60, 0.58,  0.70),
    (0.34, -0.06, 0.28,  0.40, 0.72,  0.80),
    (0.34,  0.06, 0.28, -0.40, 0.72, -0.80),
    # ── 高位 + 大 roll（当前高位姿态 roll 偏小）──
    (0.22, -0.14, 0.38,  0.75, 0.32, -0.70),
    (0.22,  0.14, 0.38, -0.75, 0.32,  0.70),
    # ── 侧向 + 大 pitch + 反向 yaw ──
    (0.31,  0.20, 0.27,  0.30, 0.68,  0.95),
    (0.31, -0.20, 0.27, -0.30, 0.68, -0.95),
    # ── 中距离，roll/pitch/yaw 三轴均衡覆盖 ──
    (0.30,  0.05, 0.32,  0.75, 0.42,  0.65),
    (0.30, -0.05, 0.32, -0.75, 0.42, -0.65),
]

AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 0.6
AUTO_MARKER_TIMEOUT_S = 2.5
AUTO_MARKER_STABLE_FRAMES = 4
MIN_CALIB_SAMPLES = 5


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ==========================================
# 重力补偿控制器（手动模式）
# ==========================================
class GravityCompController:
    """MIT 模式 + 末端速度锁止，用于手动推臂定位。

    参考 reBotArm_control_py/example/10_gravity_compensation_lock.py
    """
    KP = 8.0
    KD = 1.5
    V_THRESH  = 0.04   # 末端线速度阈值 (m/s)
    W_THRESH  = 0.08   # 末端角速度阈值 (rad/s)

    def __init__(self) -> None:
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            load_robot_model, compute_fk, get_end_effector_frame_id,
        )
        import pinocchio as pin

        self._compute_gravity = compute_generalized_gravity
        self._compute_fk = compute_fk
        self._pin = pin

        self._arm = RobotArm()
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None          # 关节数（connect 后确定）
        self._q_target = None   # 锁止目标，list[ndarray] 供线程共享
        self._integral = None
        self._gc_running = threading.Event()
        self._gc_thread: threading.Thread | None = None

    def start(self) -> None:
        """连接、使能、切换 MIT 模式，启动重力补偿线程。"""
        self._arm.connect()
        print("[GravityComp] 已连接")
        self._arm.enable()
        print("[GravityComp] 已使能")

        n = self._arm.num_joints
        self._n = n
        q0 = self._arm.get_positions(request=True)
        self._q_target = [q0.copy()]
        self._integral = [np.zeros(n)]

        self._arm.mode_mit(
            kp=np.full(n, self.KP),
            kd=np.full(n, self.KD),
        )
        print(f"[GravityComp] MIT 模式，kp={self.KP} kd={self.KD}，可手动推臂")

        self._gc_running.set()
        self._gc_thread = threading.Thread(target=self._worker, daemon=True)
        self._gc_thread.start()

    def safe_home(self) -> None:
        """停止重力补偿，用 ArmEndPos 轨迹回零位，然后断开连接。"""
        self._gc_running.clear()
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=1.0)
        try:
            print("[GravityComp] 回零位中...")
            from reBotArm_control_py.controllers import ArmEndPos
            ctrl = ArmEndPos(self._arm)
            ctrl.start()
            ctrl.end()
        except Exception as e:
            print(f"[GravityComp] 回零位失败: {e}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] 已断开")

    def stop(self) -> None:
        """停止重力补偿线程并断开连接（不复位）。"""
        self._gc_running.clear()
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=1.0)
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] 已断开")

    def get_tcp_pose(self) -> np.ndarray:
        """读取当前末端位姿（4×4 T_gripper2base）。"""
        q = self._arm.get_positions()
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3,  3] = pos
        return T

    def _worker(self) -> None:
        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        while self._gc_running.is_set():
            try:
                q  = self._arm.get_positions()
                qd = self._arm.get_velocities()
                tau_g = self._compute_gravity(q=q)

                q_err = self._q_target[0] - q
                self._integral[0] += q_err * 1.0
                np.clip(self._integral[0], -0.5, 0.5, out=self._integral[0])

                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
                v = J @ qd

                if (np.linalg.norm(v[:3]) > self.V_THRESH or
                        np.linalg.norm(v[3:]) > self.W_THRESH):
                    self._q_target[0] = q.copy()
                    self._integral[0] *= 0.9

                self._arm.mit(
                    pos=self._q_target[0],
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral[0],
                )
            except Exception:
                pass
            time.sleep(0.002)


# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-Hand 手眼标定采集")
    parser.add_argument("--manual", action="store_true",
                        help="手动模式：重力补偿，用户推臂到目标位置后按 Enter 采集")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg  = load_config(root / "config" / "default.yaml")

    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"

    # ── 相机 ──
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # ── 标定器 ──
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # ── 机器人 ──
    mode_str  = "手动（重力补偿）" if args.manual else f"自动（{len(CALIB_POSES_XYZ)} 个预设姿态）"
    gc_ctrl: GravityCompController | None = None
    robot: RebotArm | None = None
    auto = {
        "enabled": not args.manual,
        "idx": 0,
        "pose_idx": None,
        "phase": "idle",
        "settle_until": 0.0,
        "timeout_at": 0.0,
        "stable_frames": 0,
        "status": "等待启动",
        "finished": False,
    }
    result_saved = [False]

    robot_cfg = cfg.get("robot", {})
    try:
        ensure_rebot_sdk_in_syspath(robot_cfg.get("repo_root"))
        if args.manual:
            gc_ctrl = GravityCompController()
            gc_ctrl.start()
            print(f"[机器人] 手动模式就绪 — 推动机械臂定位后按 Enter 采集")
        else:
            robot = RebotArm(
                config_path=robot_cfg.get("config_path"),
                urdf_path=robot_cfg.get("urdf_path"),
                repo_root=robot_cfg.get("repo_root"),
            )
            robot.connect(enable=True)
            print(f"[机器人] 自动模式就绪，共 {len(CALIB_POSES_XYZ)} 个预设姿态，将自动遍历采集")
    except Exception as e:
        print(f"[机器人] 连接失败: {e}")
        sys.exit(1)

    print(f"\n=== Eye-in-Hand 手眼标定 ===")
    print(f"相机: {cam_type}  |  模式: {mode_str}  |  算法: {he_method}")
    print(f"ArUco 边长: {aruco_cfg['marker_length_m']*100:.0f}cm  |  保存: {save_path}")
    print()
    if args.manual:
        print("【操作】Enter=采集  c/q=结束并计算  pos=当前末端位置")
    else:
        print("【操作】自动遍历姿态并自动采样  c/q=中断并计算  pos=当前末端位置")
    print()

    # ── 开启相机 ──
    cam.open()
    print("预热相机...", end="", flush=True)
    cam.warm_up(20)
    print(" 就绪\n")

    latest_pose = [None]
    line_queue: queue.Queue | None = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)
    else:
        print("[提示] 当前不是交互终端，已禁用终端输入命令")

    def _get_fk() -> np.ndarray:
        """读取当前末端位姿（两种模式均适用）。"""
        if args.manual:
            return gc_ctrl.get_tcp_pose()
        else:
            return robot.get_tcp_pose()

    def _print_fk() -> None:
        try:
            T = _get_fk()
            t = T[:3, 3]
            R = T[:3, :3]
            _p = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
            _r = math.atan2(R[2, 1] / math.cos(_p), R[2, 2] / math.cos(_p))
            _y = math.atan2(R[1, 0] / math.cos(_p), R[0, 0] / math.cos(_p))
            print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                  f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
        except Exception as e:
            print(f"  [错误] {e}")

    def capture_sample(cur, source: str) -> bool:
        if cur is None:
            print("  [跳过] 标记不在视野中，请调整后重试")
            return False

        print(f"\n[样本 {calibrator.n_samples + 1}] {source}")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        try:
            T_g2b = _get_fk()
            t = T_g2b[:3, 3]
            print(f"  末端 (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] 已记录，共 {calibrator.n_samples} 个样本"
                  + ("  <- 结束时将自动计算" if calibrator.n_samples >= 15 else ""))
            return True
        except Exception as e:
            print(f"  [错误] 获取末端位姿失败: {e}")
            return False

    def compute_and_save(reason: str) -> bool:
        print(f"\n[结束] {reason}")
        if calibrator.n_samples < MIN_CALIB_SAMPLES:
            print(f"[结果] 样本不足（{calibrator.n_samples} < {MIN_CALIB_SAMPLES}），未计算标定结果")
            if save_path.exists():
                print("[结果] 现有 hand_eye.npz 未更新")
            return False

        print(f"[结果] 计算中（{calibrator.n_samples} 个样本）...")
        try:
            result = calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            HandEyeCalibrator.save(result, save_path)
            t = result.T_result[:3, 3]
            R = result.T_result[:3, :3]
            print(f"[结果] T_cam2gripper 平移: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            print(f"[结果] 旋转矩阵:\n{R}")
            print(f"[结果] [OK] 已保存至 {save_path}")
            if calibrator.n_samples < 15:
                print("[结果] 提示：样本 < 15，建议继续采集以提高精度")
            result_saved[0] = True
            return True
        except Exception as e:
            print(f"[结果] [错误] 计算失败: {e}")
            return False

    def start_next_auto_pose() -> bool:
        if not auto["enabled"] or robot is None:
            return False

        total = len(CALIB_POSES_XYZ)
        while auto["idx"] < total:
            idx = auto["idx"]
            x, y, z, roll, pitch, yaw = CALIB_POSES_XYZ[idx]
            print(f"\n[自动] 姿态 {idx+1}/{total}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})")
            ok = robot.move_to(x, y, z, roll=roll, pitch=pitch, yaw=yaw, duration=AUTO_MOVE_DURATION_S)
            if ok:
                now = time.monotonic()
                auto["pose_idx"] = idx
                auto["phase"] = "settling"
                auto["settle_until"] = now + AUTO_MOVE_DURATION_S + AUTO_SETTLE_EXTRA_S
                auto["timeout_at"] = auto["settle_until"] + AUTO_MARKER_TIMEOUT_S
                auto["stable_frames"] = 0
                auto["status"] = f"姿态 {idx+1}/{total} 移动中"
                return False

            print(f"[自动] 姿态 {idx+1}/{total} IK 无解，跳过")
            auto["idx"] += 1

        auto["phase"] = "done"
        auto["finished"] = True
        auto["status"] = "全部姿态已遍历"
        print("\n[自动] 全部预设姿态遍历完成")
        return True

    def tick_auto(cur) -> bool:
        if not auto["enabled"] or auto["finished"]:
            return auto["finished"]

        if auto["phase"] == "idle":
            return start_next_auto_pose()

        pose_idx = auto["pose_idx"]
        total = len(CALIB_POSES_XYZ)
        now = time.monotonic()

        if auto["phase"] == "settling":
            remain = auto["settle_until"] - now
            if remain > 0.0:
                auto["stable_frames"] = 0
                auto["status"] = f"姿态 {pose_idx+1}/{total} 移动/稳定中 {remain:.1f}s"
                return False
            auto["phase"] = "searching"

        if cur is not None:
            auto["stable_frames"] += 1
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = (
                f"姿态 {pose_idx+1}/{total} 识别稳定 "
                f"{auto['stable_frames']}/{AUTO_MARKER_STABLE_FRAMES}  剩余 {remain:.1f}s"
            )
            if auto["stable_frames"] >= AUTO_MARKER_STABLE_FRAMES:
                capture_sample(cur, f"自动姿态 {pose_idx+1}/{total}")
                auto["idx"] += 1
                auto["phase"] = "idle"
                auto["stable_frames"] = 0
                return start_next_auto_pose()
        else:
            auto["stable_frames"] = 0
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = f"姿态 {pose_idx+1}/{total} 等待 ArUco {remain:.1f}s"

        if now >= auto["timeout_at"]:
            print(f"[自动] 姿态 {pose_idx+1}/{total} 未识别到 ArUco，跳过")
            auto["idx"] += 1
            auto["phase"] = "idle"
            auto["stable_frames"] = 0
            return start_next_auto_pose()

        return False

    def handle_line(raw: str) -> bool:
        if raw is None:
            print("\n[中断] 终端输入已关闭，停止采集并尝试计算")
            return True

        line = raw.strip().lower()

        if line in {"q", "c"}:
            return True

        if line == "pos":
            _print_fk()
            return False

        if args.manual and line == "":
            capture_sample(latest_pose[0], "手动采集")
            return False

        if line:
            if args.manual:
                print("  手动模式支持: Enter=采集  c/q=结束并计算  pos=当前末端位姿")
            else:
                print("  自动模式支持: c/q=结束并计算  pos=当前末端位姿")
        return False

    # ── 主循环 ──
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    finish_reason = "正常结束"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose[0] = pose
                if tick_auto(pose):
                    finish_reason = "自动遍历完成"
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                def osd(text, y, color=(220, 220, 220)):
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                if pose:
                    if args.manual:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  "
                            f"samples:{n}  Enter=capture  c/q=finish",
                            28, (80, 220, 80))
                    else:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  samples:{n}",
                            28, (80, 220, 80))
                else:
                    if args.manual:
                        osd(f"No marker  samples:{n}  move arm to see marker",
                            28, (80, 80, 220))
                    else:
                        osd(f"No marker  samples:{n}",
                            28, (80, 80, 220))

                if args.manual:
                    osd("MANUAL: Enter=capture  c/q=finish  pos=print fk", 50, (180, 180, 60))
                else:
                    osd(f"AUTO: {auto['status']}", 50, (180, 180, 60))

                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "MANUAL(GravComp)" if args.manual else "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                cv2.imshow(WIN, vis)

            if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q'), 27]:
                finish_reason = "窗口退出"
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                finish_reason = "窗口关闭"
                break

            try:
                if line_queue is not None and handle_line(line_queue.get_nowait()):
                    finish_reason = "用户中断"
                    break
            except queue.Empty:
                pass

            if auto["finished"]:
                break

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C 中断"
        print("\n[Ctrl+C] 停止采集并尝试计算")

    finally:
        cv2.destroyAllWindows()
        cam.close()
        if gc_ctrl is not None:
            gc_ctrl.safe_home()
        elif robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        compute_and_save(finish_reason)

    print(f"\n结束，共 {calibrator.n_samples} 个样本。")
    if calibrator.n_samples > 0 and not result_saved[0]:
        print("提示：本次未生成新的 hand_eye.npz，可补充样本后重试。")


if __name__ == "__main__":
    main()
    os._exit(0)
