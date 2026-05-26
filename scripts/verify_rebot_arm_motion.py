#!/usr/bin/env python3
"""Verify reBot arm connection with a small joint6 jog and return.

Use this before running camera-driven grasp motion:

    cd /home/seeed/Downloads/rebot_grasp
    conda activate graspnet
    python scripts/verify_rebot_arm_motion.py --read-only
    python scripts/verify_rebot_arm_motion.py --deg 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REBOT_SDK_ROOT = PROJECT_ROOT / "sdk" / "reBotArm_control_py"
DEFAULT_ARM_CONFIG = REBOT_SDK_ROOT / "config" / "arm.yaml"


def _prepare_imports() -> None:
    sdk = str(REBOT_SDK_ROOT)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)


_prepare_imports()

from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.kinematics import load_robot_model  # noqa: E402


def _joint_limits_from_model() -> list[tuple[str, float, float]]:
    model = load_robot_model()
    limits: list[tuple[str, float, float]] = []
    for joint_id, name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        idx_q = int(joint.idx_q)
        if idx_q < 0:
            continue
        limits.append(
            (
                str(name),
                float(model.lowerPositionLimit[idx_q]),
                float(model.upperPositionLimit[idx_q]),
            )
        )
    return limits


def _deg(rad: float) -> float:
    return math.degrees(float(rad))


def _format_joint_line(name: str, q: float, lo: float, hi: float) -> str:
    return (
        f"  {name:<6} q={_deg(q):8.2f} deg  "
        f"limit=[{_deg(lo):8.2f}, {_deg(hi):8.2f}] deg"
    )


def _read_state(arm: RobotArm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm._request_and_poll()
    q, vel, torq = arm.get_state()
    return (
        np.asarray(q, dtype=np.float64).reshape(-1),
        np.asarray(vel, dtype=np.float64).reshape(-1),
        np.asarray(torq, dtype=np.float64).reshape(-1),
    )


def _missing_state_names(arm: RobotArm) -> list[str]:
    missing: list[str] = []
    for joint in arm._joints:
        if arm._motor_map[joint.name].get_state() is None:
            missing.append(joint.name)
    return missing


def _not_enabled_names(arm: RobotArm) -> list[str]:
    not_enabled: list[str] = []
    for joint in arm._joints:
        state = arm._motor_map[joint.name].get_state()
        if state is None or getattr(state, "status_code", None) != 1:
            not_enabled.append(joint.name)
    return not_enabled


def _safe_vlim(arm: RobotArm, factor: float, cap: float) -> np.ndarray:
    base = np.array([joint.vlim for joint in arm._joints], dtype=np.float64)
    vlim = np.minimum(base * float(factor), float(cap))
    return np.maximum(vlim, np.full_like(vlim, 0.05))


def _target_fits(
    target: np.ndarray,
    limits: list[tuple[str, float, float]],
    moved_idx: int,
    margin_rad: float,
) -> tuple[bool, str]:
    eps = math.radians(0.2)
    for i, (name, lo, hi) in enumerate(limits):
        value = float(target[i])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        lower = lo + margin_rad if i == moved_idx else lo - eps
        upper = hi - margin_rad if i == moved_idx else hi + eps
        if value < lower or value > upper:
            return (
                False,
                f"{name} target {_deg(value):.2f} deg outside safe range "
                f"[{_deg(lower):.2f}, {_deg(upper):.2f}] deg",
            )
    return True, ""


def _choose_joint6_target(
    start_q: np.ndarray,
    limits: list[tuple[str, float, float]],
    step_rad: float,
    direction: str,
    margin_rad: float,
) -> tuple[np.ndarray, str]:
    joint_idx = 5
    candidates: list[tuple[str, np.ndarray]] = []
    if direction in ("auto", "negative"):
        q_neg = start_q.copy()
        q_neg[joint_idx] -= step_rad
        candidates.append(("negative", q_neg))
    if direction in ("auto", "positive"):
        q_pos = start_q.copy()
        q_pos[joint_idx] += step_rad
        candidates.append(("positive", q_pos))

    errors: list[str] = []
    for label, target in candidates:
        ok, message = _target_fits(target, limits, joint_idx, margin_rad)
        if ok:
            return target, label
        errors.append(f"{label}: {message}")

    raise ValueError("; ".join(errors))


def _move_joints_smooth(
    target_ref: list[np.ndarray],
    start_q: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    update_hz: float,
) -> None:
    total_steps = max(int(float(duration_s) * float(update_hz)), 20)
    interval = 1.0 / float(update_hz)
    for idx in range(1, total_steps + 1):
        t = idx / total_steps
        alpha = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        target_ref[0] = ((1.0 - alpha) * start_q + alpha * target_q).copy()
        time.sleep(interval)


def _print_state(
    title: str,
    arm: RobotArm,
    q: np.ndarray,
    limits: list[tuple[str, float, float]],
) -> None:
    names = arm.joint_names if len(arm.joint_names) == len(limits) else [x[0] for x in limits]
    print(title)
    for name, value, (_, lo, hi) in zip(names, q, limits):
        print(_format_joint_line(name, float(value), lo, hi))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small safe reBot arm motion check using joint6 only."
    )
    parser.add_argument("--config", default=str(DEFAULT_ARM_CONFIG), help="arm.yaml path")
    parser.add_argument("--read-only", action="store_true", help="connect and print state only")
    parser.add_argument("--deg", type=float, default=10.0, help="joint6 jog size in degrees")
    parser.add_argument("--max-deg", type=float, default=20.0, help="maximum allowed jog size")
    parser.add_argument(
        "--direction",
        choices=("auto", "negative", "positive"),
        default="auto",
        help="joint6 jog direction; auto prefers negative if within limits",
    )
    parser.add_argument("--duration", type=float, default=1.5, help="seconds for each leg")
    parser.add_argument("--settle", type=float, default=0.5, help="pause at jog target")
    parser.add_argument("--update-hz", type=float, default=50.0, help="trajectory update rate")
    parser.add_argument("--control-rate", type=float, default=100.0, help="POS_VEL command rate")
    parser.add_argument("--vlim-factor", type=float, default=0.35, help="scale configured vlim")
    parser.add_argument("--vlim-cap", type=float, default=0.35, help="rad/s cap for smoke test")
    parser.add_argument("--limit-margin-deg", type=float, default=2.0)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    if args.deg <= 0.0 or args.deg > args.max_deg:
        print(f"[FAIL] --deg must be in (0, {args.max_deg}]")
        return 2
    if args.duration <= 0.0 or args.update_hz <= 0.0 or args.control_rate <= 0.0:
        print("[FAIL] duration, update-hz, and control-rate must be positive")
        return 2

    limits = _joint_limits_from_model()
    if len(limits) < 6:
        print(f"[FAIL] expected at least 6 robot joints in URDF, got {len(limits)}")
        return 1

    arm: RobotArm | None = None
    try:
        print(f"[INFO] loading arm config: {args.config}")
        arm = RobotArm(cfg_path=args.config)
        print(f"[INFO] serial channel: {getattr(arm, '_channel', 'unknown')}")
        q0, _, _ = _read_state(arm)
        if q0.size != arm.num_joints:
            print(f"[FAIL] expected {arm.num_joints} joints, got {q0.size}")
            return 1
        missing = _missing_state_names(arm)
        if missing:
            print(f"[FAIL] no feedback from joints: {missing}")
            return 1

        _print_state("[INFO] current joints:", arm, q0, limits)
        if args.read_only:
            print("[OK] read-only arm check completed")
            return 0

        step_rad = math.radians(float(args.deg))
        margin_rad = math.radians(float(args.limit_margin_deg))
        target_q, chosen_direction = _choose_joint6_target(
            q0, limits, step_rad, args.direction, margin_rad
        )
        print(
            f"[INFO] joint6 jog: {args.deg:.1f} deg, "
            f"direction={chosen_direction}, duration={args.duration:.2f}s each way"
        )

        vlim = _safe_vlim(arm, args.vlim_factor, args.vlim_cap)
        print(f"[INFO] smoke-test vlim rad/s: {np.round(vlim, 3).tolist()}")

        if not arm.mode_pos_vel(vlim=vlim):
            print("[FAIL] not all joints entered POS_VEL mode")
            return 1
        arm.enable()
        time.sleep(0.2)
        q_start, _, _ = _read_state(arm)
        missing = _missing_state_names(arm)
        if missing:
            print(f"[FAIL] no feedback after enable from joints: {missing}")
            return 1
        not_enabled = _not_enabled_names(arm)
        if not_enabled:
            print(f"[FAIL] joints not enabled: {not_enabled}")
            return 1
        target_q, chosen_direction = _choose_joint6_target(
            q_start, limits, step_rad, args.direction, margin_rad
        )
        print(
            f"[INFO] confirmed joint6 jog target: {args.deg:.1f} deg, "
            f"direction={chosen_direction}"
        )
        target_ref = [q_start.copy()]

        def _hold_target(robot: RobotArm, _dt: float) -> None:
            robot.pos_vel(target_ref[0], vlim=vlim)

        arm.start_control_loop(_hold_target, rate=float(args.control_rate))
        time.sleep(0.2)

        _move_joints_smooth(target_ref, q_start, target_q, args.duration, args.update_hz)
        time.sleep(float(args.settle))
        _move_joints_smooth(target_ref, target_q, q_start, args.duration, args.update_hz)
        time.sleep(0.2)

        q_end, _, _ = _read_state(arm)
        _print_state("[INFO] joints after return:", arm, q_end, limits)
        print(f"[OK] joint6 jog completed; max return error={np.max(np.abs(q_end - q_start)):.4f} rad")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[FAIL] {exc}")
        text = str(exc).lower()
        if "serial" in text or "tty" in text or "busy" in text or "no such file" in text:
            print("[HINT] Close other arm programs and check /dev/ttyACM*, /dev/ttyUSB*, /dev/serial/by-id.")
        return 1
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
