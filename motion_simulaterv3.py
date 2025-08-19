# base_motion_v3.py
import numpy as np
import torch
import argparse, os

# ---------- helpers ----------
def ease_in_out(u):
    u = np.clip(u, 0.0, 1.0)
    return 4*u*u*u if u < 0.5 else 1 - ((-2*u + 2)**3)/2

def figure8_path(T, length=10.0, width=4.0, height_bob=0.05, seed=0):
    """
    Simple figure-8 in XZ: x = L*sin(t), z = W*sin(t)*cos(t)
    Y has a small bob to look alive (optional).
    Returns: trans(T,3), yaw(T) where yaw is heading from tangent.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2*np.pi, T, dtype=np.float32)

    x = length * 0.5 * np.sin(t)
    z = width  * 0.5 * (np.sin(t) * np.cos(t)) * 2.0  # = width * 0.5 * sin(2t)

    # small vertical bob (not path elevation changes)
    y = height_bob * np.sin(4*t + 0.4)

    trans = np.stack([x, y, z], axis=1).astype(np.float32)

    # tangent-based heading (yaw)
    dx = np.gradient(x)
    dz = np.gradient(z)
    yaw = np.arctan2(dz, dx).astype(np.float32)  # radians
    return trans, yaw

def make_pause_warp(T, pause_frames=(0.0, 0.5, 1.0), base_speed=1.0, pause_depth=0.15, seed=0):
    """
    Produce a monotonic time-warp u(t) in [0,1] that slows near given normalized times.
    pause_frames: tuple of normalized positions along [0,1] where we slow/“pause”
    pause_depth: how deep the slowdowns are (0..1). Higher -> slower near pauses.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T, dtype=np.float32)
    speed = np.full(T, base_speed, dtype=np.float32)

    for p in pause_frames:
        # distance in normalized time
        d = np.abs(t - p)
        # Gaussian-ish slowdown around p
        sigma = 0.06  # width of slowdown window
        slow = np.exp(-(d**2) / (2*sigma**2))
        speed *= (1.0 - pause_depth*slow)

    # keep speed positive and normalize to end at 1
    speed = np.clip(speed, 1e-4, None)
    speed /= speed.mean()
    u = np.cumsum(speed)
    u = (u - u[0]) / (u[-1] - u[0])
    return u.astype(np.float32)

def gait_body_v3(T, stride_phase=0.45, arm_down=-0.35, arm_swing=0.18, leg_amp=0.32,
                 torso_pitch=0.05, torso_roll=0.04, head_yaw=0.12, noise=0.02, seed=0):
    """
    Produce 69 body DoF with a different style than V1/V2:
      - arms biased down by sides with moderate swing
      - slight torso pitch/roll oscillation
      - gentle head yaw scan
    NOTE: channel indices are consistent with earlier scripts: body = pose[3:].
    """
    rng = np.random.RandomState(seed)
    P = np.zeros((T, 69), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)

    # Indices (example mapping you used earlier)
    LHIP = 10; RHIP = 11
    LSHO = 20; RSHO = 21
    HEAD = 5   # pick a head/neck body channel for yaw-ish motion (approx)

    # Stride phase with mild variability
    base_w = 2.0 * np.pi * stride_phase
    local_vary = 1.0 + 0.1*np.sin(2*np.pi*t/T + 0.6)
    phi = np.cumsum(local_vary) * base_w / local_vary.mean()

    # Legs (out of phase)
    P[:, LHIP] =  leg_amp * np.sin(phi)
    P[:, RHIP] = -leg_amp * np.sin(phi + np.pi)

    # Arms: bias down, small swing
    # We'll treat LSHO/RSHO as pitch-like here for visual variety
    P[:, LSHO] = arm_down + arm_swing * np.sin(phi + np.pi)
    P[:, RSHO] = arm_down - arm_swing * np.sin(phi)

    # Torso pitch/roll (choose two body channels; indices here are illustrative)
    TORSO_PITCH = 2   # small forward/back oscillation
    TORSO_ROLL  = 8   # small side-to-side
    P[:, TORSO_PITCH] =  torso_pitch * np.sin(0.5*phi + 0.3)
    P[:, TORSO_ROLL ] =  torso_roll  * np.sin(0.5*phi + 1.2)

    # Head gentle yaw scan
    P[:, HEAD] += head_yaw * np.sin(0.25*phi + 0.7)

    # Noise for micro-variation
    P += noise * rng.randn(T, 69).astype(np.float32)
    return P

def build_base_motion_v3(T=300, seed=3):
    # Figure-8 path
    trans_full, yaw_full = figure8_path(T, length=12.0, width=5.0, height_bob=0.04, seed=seed)

    # Time warp with “pauses” around start, middle (cross), and end
    u = make_pause_warp(T, pause_frames=(0.05, 0.5, 0.95), base_speed=1.0, pause_depth=0.25, seed=seed)
    idx = np.clip((u * (T-1)).round().astype(int), 0, T-1)

    trans = trans_full[idx].copy()
    yaw   = yaw_full[idx].copy()

    # Root: yaw about Y; leave X/Z in root at 0 for stability
    root = np.zeros((T, 3), dtype=np.float32)
    root[:, 1] = yaw

    # Body style V3
    body = gait_body_v3(T, stride_phase=0.45, arm_down=-0.35, arm_swing=0.18,
                        leg_amp=0.32, torso_pitch=0.05, torso_roll=0.04,
                        head_yaw=0.12, noise=0.02, seed=seed)

    pose = np.concatenate([root, body], axis=1).astype(np.float32)
    betas = np.zeros((T, 10), dtype=np.float32)

    return pose, trans, betas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default="ptFiles/base_motion_v3.pt")
    args = ap.parse_args()

    pose, trans, betas = build_base_motion_v3(T=args.frames, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "pose":  torch.from_numpy(pose),
        "trans": torch.from_numpy(trans),
        "betas": torch.from_numpy(betas),
    }, args.out)
    print(f"Saved {args.out}  pose:{pose.shape}  trans:{trans.shape}  betas:{betas.shape}")

if __name__ == "__main__":
    main()

