# procedural_base_motion.py
import os
import numpy as np
import torch

# ----------------------------
# Path + facing (yaw) helpers
# ----------------------------
def generate_path_from_waypoints(waypoints, total_steps):
    """
    Interpolate along waypoints at roughly constant arc-length.
    Returns (T,3) float32 translation in world space.
    """
    wps = np.asarray(waypoints, dtype=np.float32)
    if len(wps) < 2:
        return np.zeros((total_steps, 3), dtype=np.float32)

    segs    = wps[1:] - wps[:-1]
    seg_len = np.linalg.norm(segs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0)
    total   = max(cum_len[-1], 1e-6)

    s_targets = np.linspace(0.0, total, total_steps, dtype=np.float32)
    out = np.zeros((total_steps, 3), dtype=np.float32)

    si = 0
    for i, s in enumerate(s_targets):
        while si < len(seg_len) - 1 and s > cum_len[si + 1]:
            si += 1
        seg_d = seg_len[si] if seg_len[si] > 1e-8 else 1.0
        u = (s - cum_len[si]) / seg_d
        out[i] = (1.0 - u) * wps[si] + u * wps[si + 1]
    return out

def yaw_from_tangent(tan):  # tan: (T,3)
    """Yaw (radians) from path tangents; unwrapped for smoothness."""
    raw = np.arctan2(tan[:, 0], tan[:, 2])  # yaw=0 when facing +Z
    return np.unwrap(raw)

# ----------------------------
# SMPL body_pose index map (safe defaults)
# These are indices in the 23-joint SMPL body_pose (AFTER global_orient),
# so body_pose has 69 numbers = 23 * 3 (axis-angle x,y,z per joint).
# Adjust if your build uses a different ordering.
# ----------------------------
DEFAULT_IDX = {
    # legs
    "L_HIP"    : 0,   "R_HIP"    : 1,
    "SPINE1"   : 2,   # (unused, reference)
    "L_KNEE"   : 3,   "R_KNEE"   : 4,
    "SPINE2"   : 5,   # (unused)
    "L_ANKLE"  : 6,   "R_ANKLE"  : 7,
    # arms
    "L_SHOULDER": 15, "R_SHOULDER": 16,
    "L_ELBOW"   : 17, "R_ELBOW"   : 18,
}

# ----------------------------
# Axis-angle write helpers
# ----------------------------
def _axis_angle_add_x(body_pose, joint_idx, angle):
    base = joint_idx * 3
    body_pose[:, base + 0] += angle

def _axis_angle_add_y(body_pose, joint_idx, angle):
    base = joint_idx * 3
    body_pose[:, base + 1] += angle

def _axis_angle_add_z(body_pose, joint_idx, angle):
    base = joint_idx * 3
    body_pose[:, base + 2] += angle

# ----------------------------
# Main generator
# ----------------------------
def make_base_motion(
    total_steps=240,
    waypoints=((0,0,0),(8,0,0)),
    add_gait=True,             # turn joint motion on/off
    fps=30,
    stride_hz=1.8,             # gait frequency (~steps per second)
    # Arm posture + swing
    shoulder_drop_deg=35.0,    # static adduction (arms down)
    shoulder_internal_deg=5.0, # tiny internal rotation for natural rest
    elbow_flex_deg=12.0,       # slight elbow bend
    arm_amp_deg=16.0,          # dynamic shoulder swing amplitude
    elbow_amp_deg=10.0,        # dynamic elbow swing amplitude
    # Leg swing
    hip_amp_deg=18.0,
    knee_amp_deg=28.0,
    ankle_amp_deg=10.0,
    idx_map=DEFAULT_IDX
):
    """
    Returns dict with SMPL-friendly tensors:
      'pose':  (T,72)  [global_orient(3) + body_pose(69)]
      'trans': (T,3)
      'betas': (T,10)  zeros
    """
    T = int(total_steps)

    # -- path + yaw --
    trans = generate_path_from_waypoints(waypoints, T)     # (T,3)
    tan = np.gradient(trans, axis=0)
    tan[0]  = tan[1]
    tan[-1] = tan[-2]
    yaw = yaw_from_tangent(tan).astype(np.float32)         # (T,)

    # -- pose container --
    pose = np.zeros((T, 72), dtype=np.float32)
    pose[:, 1] = yaw  # global_orient = [0, yaw, 0]

    # -- body_pose (69) --
    body = np.zeros((T, 69), dtype=np.float32)

    # Static arm posture: bring arms down by the sides + tiny internal rotation + slight elbow flex
    shoulder_down = np.deg2rad(shoulder_drop_deg)
    shoulder_internal = np.deg2rad(shoulder_internal_deg)
    elbow_flex = np.deg2rad(elbow_flex_deg)

    # Adduction about Z: L negative, R positive (flip sign if your rig is opposite)
    _axis_angle_add_z(body, idx_map["L_SHOULDER"], -shoulder_down)
    _axis_angle_add_z(body, idx_map["R_SHOULDER"],  shoulder_down)

    # Small internal rotation about Y: L +, R -
    _axis_angle_add_y(body, idx_map["L_SHOULDER"],  shoulder_internal)
    _axis_angle_add_y(body, idx_map["R_SHOULDER"], -shoulder_internal)

    # Slight elbow flex around X for both arms
    _axis_angle_add_x(body, idx_map["L_ELBOW"], elbow_flex)
    _axis_angle_add_x(body, idx_map["R_ELBOW"], elbow_flex)

    if add_gait:
        # Dynamic gait sinusoids (sagittal rotations only → stable)
        t_sec = np.arange(T, dtype=np.float32) / float(fps)
        w = 2.0 * np.pi * stride_hz
        phase_L = 0.0
        phase_R = np.pi

        hip_amp   = np.deg2rad(hip_amp_deg)
        knee_amp  = np.deg2rad(knee_amp_deg)
        ankle_amp = np.deg2rad(ankle_amp_deg)
        arm_amp   = np.deg2rad(arm_amp_deg) * 0.6     # slightly reduced (we already dropped arms)
        elb_amp   = np.deg2rad(elbow_amp_deg) * 0.5

        # Hips (flex/extend around X)
        hip_L = hip_amp * np.sin(w*t_sec + phase_L)
        hip_R = hip_amp * np.sin(w*t_sec + phase_R)
        _axis_angle_add_x(body, idx_map["L_HIP"], hip_L)
        _axis_angle_add_x(body, idx_map["R_HIP"], hip_R)

        # Knees (slight phase lead so they flex near foot-up)
        knee_L = knee_amp * np.sin(w*t_sec + phase_L + np.pi/6.0)
        knee_R = knee_amp * np.sin(w*t_sec + phase_R + np.pi/6.0)
        _axis_angle_add_x(body, idx_map["L_KNEE"], knee_L)
        _axis_angle_add_x(body, idx_map["R_KNEE"], knee_R)

        # Ankles (small counter)
        ank_L = ankle_amp * np.sin(w*t_sec + phase_L + np.pi/3.0)
        ank_R = ankle_amp * np.sin(w*t_sec + phase_R + np.pi/3.0)
        _axis_angle_add_x(body, idx_map["L_ANKLE"], ank_L)
        _axis_angle_add_x(body, idx_map["R_ANKLE"], ank_R)

        # Arms (counter-swing around X)
        sh_L = arm_amp * np.sin(w*t_sec + phase_R)  # counter to L leg
        sh_R = arm_amp * np.sin(w*t_sec + phase_L)  # counter to R leg
        _axis_angle_add_x(body, idx_map["L_SHOULDER"], sh_L)
        _axis_angle_add_x(body, idx_map["R_SHOULDER"], sh_R)

        el_L = elb_amp * np.sin(w*t_sec + phase_R + np.pi/6.0)
        el_R = elb_amp * np.sin(w*t_sec + phase_L + np.pi/6.0)
        _axis_angle_add_x(body, idx_map["L_ELBOW"], el_L)
        _axis_angle_add_x(body, idx_map["R_ELBOW"], el_R)

    # write body_pose back
    pose[:, 3:] = body

    # betas (constant shape, duplicated per-frame for batched SMPL)
    betas = np.zeros((T, 10), dtype=np.float32)

    return {
        "pose":  torch.from_numpy(pose),
        "trans": torch.from_numpy(trans.astype(np.float32)),
        "betas": torch.from_numpy(betas),
    }

def save_pt(bundle, out_path="ptFiles/base_motion.pt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(bundle, out_path)
    print("Saved", out_path, {k: tuple(v.shape) for k, v in bundle.items()})

if __name__ == "__main__":
    # ✏️ Tweak these:
    WAYPOINTS = ((0,0,0), (4,0,0), (4,0,3), (8,0,3))  # gentle L-turn
    TOTAL_STEPS = 240
    ADD_GAIT = True  # set False to keep only path + yaw (no joint motion)

    out = make_base_motion(
        total_steps=TOTAL_STEPS,
        waypoints=WAYPOINTS,
        add_gait=ADD_GAIT,
        fps=30,
        stride_hz=1.8,
        # posture + swing settings (deg)
        shoulder_drop_deg=35.0,
        shoulder_internal_deg=5.0,
        elbow_flex_deg=12.0,
        arm_amp_deg=16.0,
        elbow_amp_deg=10.0,
        hip_amp_deg=18.0,
        knee_amp_deg=28.0,
        ankle_amp_deg=10.0,
        idx_map=DEFAULT_IDX,
    )
    save_pt(out, "ptFiles/base_motion.pt")
