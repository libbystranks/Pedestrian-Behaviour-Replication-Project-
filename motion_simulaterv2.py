# base_motion_v2.py
import numpy as np
import torch
import argparse, os

# --- helpers ---
def ease_in_out(u):
    u = np.clip(u, 0.0, 1.0)
    return 4*u*u*u if u < 0.5 else 1 - ((-2*u + 2)**3)/2

def make_s_curve_path(T, length=10.0, width=3.0, height_variation=0.0, seed=0):
    """
    Parametric S-curve in XZ with optional slight Y undulation.
    Returns (T,3) trans and per-frame forward yaw (axis-angle Y) from tangent.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T, dtype=np.float32)

    # S-curve in XZ
    x = length * t
    z = width * np.sin(2 * np.pi * (t * 0.5 + 0.1))  # half sine period over the length
    y = height_variation * np.sin(2 * np.pi * (t * 0.25 + 0.2))

    trans = np.stack([x, y, z], axis=1).astype(np.float32)

    # Tangent heading → yaw
    dx = np.gradient(x)
    dz = np.gradient(z)
    yaw = np.arctan2(dz, dx).astype(np.float32)  # radians

    return trans, yaw

def variable_speed_warp(T, slow_ratio=0.2, fast_ratio=0.2, seed=0):
    """
    Produce a time-warp curve u(t) in [0,1]: slows at start & end, speeds in middle.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T, dtype=np.float32)
    w = 0.65 + 0.1 * np.sin(2*np.pi*t)  # base undulation
    # emphasize mid-track speed
    accel = 1.0 + 0.8 * np.exp(-((t-0.5)/0.22)**2)
    s = w * accel
    s /= s.sum()
    u = np.cumsum(s)
    u = (u - u[0]) / (u[-1] - u[0])
    return u.astype(np.float32)

def pose_gait_body(T, stride_phase=0.38, arm_amp=0.25, leg_amp=0.35, noise=0.03, seed=0):
    """
    Very light-weight procedural gait in 69 DoF (body only), leaving root in yaw.
    This is just to bias the style away from V1; the dataset snippets will replace it.
    """
    rng = np.random.RandomState(seed)
    P = np.zeros((T, 69), dtype=np.float32)
    tt = np.arange(T, dtype=np.float32)

    # pick a couple of leg channels for visual cadence changes (indexes are arbitrary but consistent)
    # (these are body channels after the first 3 root channels)
    LHIP = 10; RHIP = 11
    LSHO = 20; RSHO = 21

    # variable stride
    phase = 2.0 * np.pi * stride_phase
    cadence = 1.0 + 0.15 * np.sin(2*np.pi*tt/T)          # slower near ends, faster mid
    phi = np.cumsum(cadence) * phase / cadence.mean()

    # legs out of phase
    P[:, LHIP] =  leg_amp * np.sin(phi)
    P[:, RHIP] = -leg_amp * np.sin(phi + np.pi)

    # arm counter-swing with slight asymmetry
    P[:, LSHO] =  arm_amp * np.sin(phi + np.pi) * 0.9
    P[:, RSHO] = -arm_amp * np.sin(phi) * 1.1

    # light noise for variety
    P += noise * rng.randn(T, 69).astype(np.float32)

    return P

def build_base_motion_v2(T=240, seed=0):
    # time warp to vary speed along the path
    u = variable_speed_warp(T, seed=seed)

    # S-curve path, then sample it with time warp
    trans_full, yaw_full = make_s_curve_path(T, length=10.0, width=3.0, height_variation=0.0, seed=seed)
    idx = np.clip((u * (T-1)).round().astype(int), 0, T-1)
    trans = trans_full[idx].copy()
    yaw   = yaw_full[idx].copy()

    # root (axis-angle): only yaw about Y
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 1] = yaw

    # body style (69 DoF)
    body = pose_gait_body(T, stride_phase=0.42, arm_amp=0.22, leg_amp=0.30, noise=0.02, seed=seed)

    # join into pose
    pose = np.concatenate([root, body], axis=1).astype(np.float32)

    # betas (neutral)
    betas = np.zeros((T, 10), dtype=np.float32)

    return pose, trans, betas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ptFiles/base_motion_v2.pt")
    args = ap.parse_args()

    pose, trans, betas = build_base_motion_v2(T=args.frames, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "pose":  torch.from_numpy(pose),
        "trans": torch.from_numpy(trans),
        "betas": torch.from_numpy(betas),
    }, args.out)
    print(f"Saved {args.out}  pose:{pose.shape}  trans:{trans.shape}  betas:{betas.shape}")

if __name__ == "__main__":
    main()

