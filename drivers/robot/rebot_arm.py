"""
cameraws.drivers.robot.rebot_arm
=================================
轻量包装层，将 reBotArm_control_py 的低层 API 封装为
相机感知系统需要的简洁接口，并内置夹爪力控状态机。

    connect()              — 使能电机
    disconnect()           — 失能并关闭
    get_tcp_pose()         — 通过 FK 读取末端位姿 (4×4 T_gripper2base)
    move_to(x,y,z)         — 通过 IK + 轨迹控制器移动末端
    safe_home()            — 回零位

    init_gripper()         — 注册夹爪电机（共用 CAN 总线）
    open_gripper(dist)     — 张开夹爪（非阻塞）
    close_gripper()        — 纯力矩闭合（非阻塞）
    grasp(force, timeout)  — 柔性夹取（阻塞）
    release_gripper()      — 张开并回零（阻塞）
    get_gripper_state()    — 读取 (pos, vel, torq)
    set_gripper_zero()     — 设置当前位置为零点
    gripper_is_holding     — 属性：是否处于力控保持状态
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"


def _is_rebot_repo_root(path: Path) -> bool:
    return path.is_dir() and (path / _REBOT_REPO_NAME).is_dir()


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    candidates = []
    if hint:
        candidates.append(Path(hint).expanduser().resolve())
    candidates += [
        _CAMERAWS_ROOT / "sdk" / _REBOT_REPO_NAME,
        _CAMERAWS_ROOT.parent / _REBOT_REPO_NAME,
        Path.home() / "seeed" / _REBOT_REPO_NAME,
        Path("/home/chlorine/seeed") / _REBOT_REPO_NAME,
    ]
    for p in candidates:
        if _is_rebot_repo_root(p):
            return p
    raise FileNotFoundError(
        "找不到 reBotArm_control_py 仓库，请在 config/default.yaml 中设置 "
        "robot.repo_root，或将 SDK 放到 cameraws/sdk/reBotArm_control_py"
    )


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


# ── 夹爪状态机常量 ────────────────────────────────────────────────────────────
_G_MAX_DIST_M      = 0.09
_G_ANGLE_OPEN      = -5.0
_G_OPEN_SOFT_LIMIT = -4.9
_G_ARRIVE_TOL      = 0.12
_G_HARD_STOP_ANGLE = -0.05
_G_TAU_MAX         = 1.5
_G_KP_MOVE         = 5.0
_G_KD_MOVE         = 1.0
_G_OPEN_RATE       = 4.0
_G_CLOSE_TORQUE    = 1.0
_G_KD_CLOSE        = 0.5
_G_STALL_VEL       = 0.05
_G_STARTUP_DIST    = 0.30
_G_KP_HOLD         = 5.0
_G_KD_HOLD         = 1.0
_G_DEFAULT_FORCE   = 0.30
_G_CTRL_RATE       = 500.0


class _GS:
    IDLE    = 0
    OPENING = 1
    CLOSING = 2
    CONTACT = 3
    HOLDING = 4
    HOMING  = 5


class RebotArm:
    """
    相机感知系统 ↔ 机械臂接口，内置夹爪力控状态机。

    Args:
        config_path: robot.yaml 路径；None = 使用仓库默认
        urdf_path:   URDF 路径；None = 使用仓库默认
        repo_root:   reBotArm_control_py 仓库根目录；None = 自动搜索
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        urdf_path:   Optional[str] = None,
        repo_root:   Optional[str] = None,
    ) -> None:
        repo = ensure_rebot_sdk_in_syspath(repo_root)
        self._repo_root = repo

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.kinematics import (
            load_robot_model,
            compute_fk,
            get_end_effector_frame_id,
            pos_rot_to_se3,
            IKSolverParams,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik
        from reBotArm_control_py.controllers import ArmEndPos

        cfg = str(config_path) if config_path else None
        self._arm = RobotArm(cfg_path=cfg)

        if urdf_path:
            self._model = load_robot_model(urdf_path=str(urdf_path))
        else:
            self._model = load_robot_model()

        self._data = self._model.createData()
        self._ee_frame_id = get_end_effector_frame_id(self._model)
        self._compute_fk = compute_fk
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._ik_check_params = IKSolverParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

        self._endpos_ctrl: Optional[ArmEndPos] = None
        self._ArmEndPos = ArmEndPos

        self._connected = False

        # 夹爪电机（注册到机械臂已有的 CAN 总线）
        self._gripper_mot  = None
        self._gripper_kp   = _G_KP_HOLD
        self._gripper_kd   = _G_KD_HOLD
        self._gripper_ctrl = None

        # 夹爪状态机
        self._g_state            = _GS.IDLE
        self._g_lock             = threading.Lock()
        self._g_pos              = 0.0
        self._g_vel              = 0.0
        self._g_torq             = 0.0
        self._g_pos_start        = 0.0
        self._g_q_contact        = 0.0
        self._g_contact_elapsed  = 0.0
        self._g_open_q_des       = _G_OPEN_SOFT_LIMIT
        self._g_open_target      = _G_OPEN_SOFT_LIMIT
        self._g_target_force     = _G_DEFAULT_FORCE
        self._g_loop_thread: Optional[threading.Thread] = None
        self._g_loop_running     = False
        self._g_loop_stop        = threading.Event()

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def connect(self, enable: bool = True) -> None:
        self._arm.connect()
        if enable:
            self._arm.enable()
            time.sleep(0.5)
            self._endpos_ctrl = self._ArmEndPos(self._arm)
            self._endpos_ctrl.start()
            print("[RebotArm] 连接成功，电机已使能")
        else:
            self._arm._request_and_poll()
            print("[RebotArm] 连接成功，电机保持失能（只读模式）")
        self._connected = True

    def disconnect(self) -> None:
        self._g_stop_loop()
        if self._endpos_ctrl is not None:
            try:
                self._endpos_ctrl.end()
            except Exception:
                pass
            self._endpos_ctrl = None
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._connected = False
        print("[RebotArm] 已断开连接")

    # ── 夹爪初始化 ────────────────────────────────────────────────────────────

    def init_gripper(self, cfg_path: Optional[str] = None) -> None:
        """将夹爪电机注册到机械臂已有的 CAN 总线，并启动力控状态机。"""
        from reBotArm_control_py.actuator.gripper import load_cfg as load_gripper_cfg
        from motorbridge import Mode, CallError

        if cfg_path is None:
            cfg_path = str(self._repo_root / "config" / "gripper.yaml")

        gcfg = load_gripper_cfg(cfg_path)
        gc = gcfg["gripper"]

        vendor = gc.vendor
        if vendor not in self._arm._ctrl_map:
            raise RuntimeError(
                f"夹爪 vendor={vendor!r} 与机械臂 vendor 不同，无法共用 Controller"
            )
        ctrl = self._arm._ctrl_map[vendor]

        if vendor == "damiao":
            self._gripper_mot = ctrl.add_damiao_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "myactuator":
            self._gripper_mot = ctrl.add_myactuator_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "robstride":
            self._gripper_mot = ctrl.add_robstride_motor(gc.motor_id, gc.feedback_id, gc.model)
        else:
            raise ValueError(f"不支持的夹爪 vendor: {vendor!r}")

        self._gripper_kp   = gc.kp
        self._gripper_kd   = gc.kd
        self._gripper_ctrl = ctrl

        # 用同一把 RLock 串行化机械臂循环与夹爪循环的所有总线操作
        if not hasattr(ctrl, '_bus_lock'):
            ctrl._bus_lock = threading.RLock()
        lock = ctrl._bus_lock

        def _wrap(fn, _lock=lock):
            def _locked(*a, **kw):
                with _lock:
                    return fn(*a, **kw)
            return _locked

        # 锁住 Controller 的读操作（所有调用方共用同一实例，只需打一次补丁）
        if not hasattr(ctrl, '_bus_lock_patched'):
            ctrl.poll_feedback_once = _wrap(ctrl.poll_feedback_once)
            ctrl._bus_lock_patched = True

        # 逐电机锁住写操作：每次只锁单关节 send_pos_vel（~0.5ms），
        # 避免 pos_vel 一次锁住全部 6 关节（~3ms）饿死 500Hz 夹爪循环。
        # _request_and_poll 通过已锁的 poll_feedback_once 间接串行化。
        if not hasattr(self._arm, '_bus_lock_patched'):
            for jc in self._arm._joints:
                mot = self._arm._motor_map[jc.name]
                for _mattr in ('send_pos_vel', 'send_mit', 'request_feedback'):
                    if hasattr(mot, _mattr):
                        setattr(mot, _mattr, _wrap(getattr(mot, _mattr)))
            self._arm._bus_lock_patched = True

        try:
            ctrl.enable_all()
            time.sleep(0.3)
        except CallError as e:
            print(f"[RebotArm] 夹爪使能警告: {e}")
        try:
            self._gripper_mot.ensure_mode(Mode.MIT, 1000)
        except CallError as e:
            raise RuntimeError(f"夹爪 MIT 模式切换失败: {e}") from e

        self._g_start_loop()
        print("[RebotArm] 夹爪已注册到 CAN 总线，力控状态机已启动")

    @property
    def has_gripper(self) -> bool:
        return self._gripper_mot is not None

    @property
    def gripper_is_holding(self) -> bool:
        with self._g_lock:
            return self._g_state == _GS.HOLDING

    # ── 夹爪状态机内部 ────────────────────────────────────────────────────────

    def _g_safe_mit(self, pos: float, vel: float, kp: float, kd: float, tau_ff: float = 0.0) -> None:
        pos_cmd  = float(np.clip(pos, _G_OPEN_SOFT_LIMIT, 0.0))
        pos_term = kp * (pos_cmd - self._g_pos) + kd * (-self._g_vel)
        tau_safe = float(np.clip(pos_term + tau_ff, -_G_TAU_MAX, _G_TAU_MAX)) - pos_term
        lock = getattr(self._gripper_ctrl, '_bus_lock', None)
        try:
            if lock:
                with lock:
                    self._gripper_mot.send_mit(pos_cmd, vel, kp, kd, tau_safe)
                    self._gripper_mot.request_feedback()
                    self._gripper_ctrl.poll_feedback_once()
            else:
                self._gripper_mot.send_mit(pos_cmd, vel, kp, kd, tau_safe)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
        except Exception:
            pass

    def _g_tick(self, dt: float) -> None:
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                self._g_pos  = float(st.pos)
                self._g_vel  = float(st.vel)
                self._g_torq = float(st.torq)
        except Exception:
            pass

        pos = self._g_pos
        vel = self._g_vel

        with self._g_lock:
            s  = self._g_state
            tf = self._g_target_force

        if s == _GS.OPENING:
            with self._g_lock:
                target = self._g_open_target
                self._g_open_q_des = max(self._g_open_q_des - _G_OPEN_RATE * dt, target)
                q = self._g_open_q_des
            self._g_safe_mit(q, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            if abs(pos - target) < _G_ARRIVE_TOL:
                with self._g_lock:
                    self._g_state = _GS.IDLE

        elif s == _GS.CLOSING:
            self._g_safe_mit(0.0, 0.0, 0.0, _G_KD_CLOSE, _G_CLOSE_TORQUE)
            with self._g_lock:
                ps = self._g_pos_start
            if abs(pos - ps) >= _G_STARTUP_DIST:
                if pos > _G_HARD_STOP_ANGLE:
                    with self._g_lock:
                        self._g_state = _GS.IDLE
                elif abs(vel) < _G_STALL_VEL:
                    with self._g_lock:
                        self._g_q_contact       = pos
                        self._g_contact_elapsed = 0.0
                        self._g_state           = _GS.CONTACT

        elif s == _GS.CONTACT:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD)
            with self._g_lock:
                self._g_contact_elapsed += dt
                if self._g_contact_elapsed >= 0.02:
                    self._g_state = _GS.HOLDING

        elif s == _GS.HOLDING:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD, tf)

        elif s == _GS.HOMING:
            self._g_safe_mit(0.0, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            if abs(pos) < _G_ARRIVE_TOL:
                with self._g_lock:
                    self._g_state = _GS.IDLE

    def _g_ctrl_loop(self) -> None:
        dt = 1.0 / _G_CTRL_RATE
        last = time.perf_counter()
        while not self._g_loop_stop.is_set():
            now = time.perf_counter()
            elapsed = now - last
            if elapsed >= dt:
                last += dt
                self._g_tick(elapsed)
            else:
                time.sleep(1e-4)

    def _g_start_loop(self) -> None:
        if self._g_loop_running:
            return
        self._g_loop_stop.clear()
        self._g_loop_thread = threading.Thread(target=self._g_ctrl_loop, daemon=True)
        self._g_loop_thread.start()
        self._g_loop_running = True

    def _g_stop_loop(self) -> None:
        if not self._g_loop_running:
            return
        self._g_loop_stop.set()
        if self._g_loop_thread is not None:
            self._g_loop_thread.join(timeout=1.0)
            self._g_loop_thread = None
        self._g_loop_running = False
        # 软停：发一帧阻尼命令，避免失能时硬件卡顿
        if self._gripper_mot is not None:
            try:
                self._gripper_mot.send_mit(self._g_pos, 0.0, 0.0, _G_KD_MOVE, 0.0)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
            except Exception:
                pass

    def _g_wait_idle(self, timeout: float = 3.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                if self._g_state == _GS.IDLE:
                    return True
            time.sleep(0.01)
        return False

    # ── 夹爪公开接口 ──────────────────────────────────────────────────────────

    def open_gripper(self, distance_m: float = _G_MAX_DIST_M) -> None:
        """张开夹爪（阻塞，等待到位，最多 3s）。"""
        if self._gripper_mot is None:
            return
        d = float(np.clip(distance_m, 0.0, _G_MAX_DIST_M))
        target = max((d / _G_MAX_DIST_M) * _G_ANGLE_OPEN, _G_OPEN_SOFT_LIMIT)
        with self._g_lock:
            self._g_open_target = target
            self._g_open_q_des  = self._g_pos
            self._g_state = _GS.OPENING
        self._g_wait_idle(3.0)

    def close_gripper(self) -> None:
        """纯力矩闭合（非阻塞）。"""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_state = _GS.CLOSING

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        """柔性夹取：闭合 → 接触检测 → 力控保持（阻塞）。

        Returns:
            True = 成功夹取（HOLDING），False = 空夹取或超时
        """
        if self._gripper_mot is None:
            return False
        if force is not None:
            with self._g_lock:
                self._g_target_force = float(np.clip(force, 0.05, _G_TAU_MAX))
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_state = _GS.CLOSING
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                s = self._g_state
            if s == _GS.HOLDING:
                return True
            if s == _GS.IDLE:
                return False
            time.sleep(0.01)
        with self._g_lock:
            self._g_state = _GS.IDLE
        return False

    def release_gripper(self, timeout: float = 4.0) -> None:
        """张开夹爪并回零（阻塞）。"""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_open_q_des = self._g_pos
            self._g_state = _GS.OPENING
        self._g_wait_idle(2.0)
        with self._g_lock:
            self._g_state = _GS.HOMING
        self._g_wait_idle(timeout)

    def get_gripper_state(self) -> tuple:
        """返回 (pos_rad, vel_rad_s, torq_nm)。"""
        return (self._g_pos, self._g_vel, self._g_torq)

    def set_gripper_zero(self) -> bool:
        """设置当前位置为零点（会暂停控制循环）。"""
        if self._gripper_mot is None:
            return False
        self._g_stop_loop()
        from motorbridge import CallError
        try:
            self._gripper_mot.set_zero_position()
            print("[RebotArm] 夹爪零点已设置")
            ok = True
        except CallError as e:
            print(f"[RebotArm] 夹爪零点设置失败: {e}")
            ok = False
        if ok:
            self._g_start_loop()
            with self._g_lock:
                self._g_state = _GS.IDLE
        return ok

    # ── 状态读取 ──────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """通过正运动学读取当前末端位姿，返回 (4, 4) 齐次变换矩阵。"""
        self._arm._request_and_poll()
        q, _, _ = self._arm.get_state()
        position, rotation, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3,  3] = position
        return T

    # ── 运动控制 ──────────────────────────────────────────────────────────────

    def check_ik(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> tuple[bool, float]:
        """只求解 IK，不发送任何运动指令。"""
        self._arm._request_and_poll()
        q_curr, _, _ = self._arm.get_state()
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        result = self._solve_ik(
            self._model,
            self._data,
            self._ee_frame_id,
            target,
            q_curr,
            self._ik_check_params,
        )
        return bool(result.success), float(result.error)

    def move_to(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        return bool(self._endpos_ctrl.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        """等待当前末端轨迹发送线程结束。"""
        if self._endpos_ctrl is None:
            return
        thread = getattr(self._endpos_ctrl, "_send_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=duration + extra + 2.0)
        else:
            time.sleep(duration + extra)

    def safe_home(self, duration: float = 3.0) -> None:
        """回零位（关节全部归零）。"""
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        self._endpos_ctrl.safe_home()

    # ── 上下文管理器 ──────────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
