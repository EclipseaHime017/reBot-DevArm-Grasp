#!/usr/bin/env python3
"""Print and sanity-check a saved hand-eye calibration file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify hand-eye calibration npz")
    parser.add_argument("--camera-type", default="orbbec_gemini2")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    path = Path(args.path) if args.path else (
        PROJECT_ROOT / "config" / "calibration" / args.camera_type / "hand_eye.npz"
    )
    if not path.exists():
        print(f"[FAIL] calibration file missing: {path}")
        return 1

    data = np.load(str(path), allow_pickle=False)
    required = ("T_result", "mode", "n_samples", "method")
    missing = [key for key in required if key not in data]
    if missing:
        print(f"[FAIL] missing keys: {missing}")
        return 1

    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    n_samples = int(data["n_samples"][0])
    method = str(data["method"][0])
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    t_mm = T[:3, 3] * 1000.0

    ok = True
    if T.shape != (4, 4):
        print(f"[FAIL] T_result shape is {T.shape}, expected (4, 4)")
        ok = False
    if mode != "eye_in_hand":
        print(f"[FAIL] mode is {mode!r}, expected 'eye_in_hand'")
        ok = False
    if n_samples < 5:
        print(f"[FAIL] n_samples={n_samples}, expected >= 5")
        ok = False
    if abs(det - 1.0) > 0.05 or ortho_err > 0.05:
        print(f"[FAIL] rotation invalid: det={det:.4f}, orthogonality_error={ortho_err:.4f}")
        ok = False

    print(f"[INFO] file: {path}")
    print(f"[INFO] mode={mode} method={method} n_samples={n_samples}")
    print(f"[INFO] translation_mm=({t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f})")
    print(f"[INFO] rotation_det={det:.4f} orthogonality_error={ortho_err:.6f}")
    print("[OK] hand-eye calibration looks usable" if ok else "[FAIL] hand-eye calibration check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
