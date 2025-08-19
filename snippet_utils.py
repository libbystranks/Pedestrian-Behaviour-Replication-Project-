# snippet_utils.py
import os
import numpy as np
import torch

# -----------------------------
# Axis-angle / Rot / Quat utils
# -----------------------------

def aa_to_mat(a):
    """Axis-angle (3,) -> rotation matrix (3,3)."""
    a = np.asarray(a, np.float32)
    theta = np.linalg.norm(a)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = a / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)

def mat_to_aa(R):
    """Rotation matrix (3,3) -> axis-angle (3,)."""
    R = np.asarray(R, np.float32)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]], dtype=np.float32) / (2.0 * np.sin(theta))
    return (w * theta).astype(np.float32)

def aa_to_quat(a):
    """Axis-angle (3,) -> quaternion (w,x,y,z)."""
    a = np.asarray(a, np.float32)
    theta = np.linalg.norm(a)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = a / theta
    half = theta * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32)

def quat_to_aa(q):
    """Quaternion (w,x,y,z) -> axis-angle (3,)."""
    q = np.asarray(q, np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    w = np.clip(q[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = q[1:] / s
    return (axis * theta).astype(np.float32)

def quat_slerp(q0, q1, u):
    """Slerp unit quats (w,x,y,z) at t=u in [0,1] along shortest arc."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - u) * q0 + u * q1
        return q / (np.linalg.norm(q) + 1e-8)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    theta = theta_0 * u
    s0 = np.sin(theta_0 - theta) / (sin_0 + 1e-8)
    s1 = np.sin(theta) / (sin_0 + 1e-8)
    q = s0 * q0 + s1 * q1
    return q / (np.linalg.norm(q) + 1e-8)

# -----------------------------
# Loading + windowed rebase
# -----------------------------

def load_snippet(data_folder, filename, center_idx, length):
    """
    Load a contiguous snippet (centered when possible).
    Returns pose:(L,72), trans:(L,3), betas:(L,10)  as float32 numpy arrays.
    """
    path = os.path.join(data_folder, filename)
    d = torch.load(path, map_location="cpu")

    pose = d["pose"].detach().cpu().numpy().astype(np.float32)           # (N,72)
    trans = d.get("trans", torch.zeros(len(pose), 3)).detach().cpu().numpy().astype(np.float32)
    betas = d.get("betas", torch.zeros(len(pose), 10)).detach().cpu().numpy().astype(np.float32)

    # broadcast betas if needed
    if betas.ndim == 1:
        betas = np.tile(betas[None, :], (pose.shape[0], 1))
    elif betas.ndim == 2 and betas.shape[0] == 1:
        betas = np.tile(betas, (pose.shape[0], 1))

    half = length // 2
    start = max(0, int(center_idx) - half)
    end   = min(start + length, pose.shape[0])
    start = max(0, end - length)

    pose  = pose[start:end].astype(np.float32)
    trans = trans[start:end].astype(np.float32)
    betas = betas[start:end].astype(np.float32)

    return pose, trans, betas

def rebase_snippet_to_state_windowed(
    pose_snip, trans_snip,
    base_pose_window,  # (W,72)  last W output frames
    base_trans_window, # (W,3)
):
    """
    Align snippet to current output using a small window (least squares).
    Steps:
      1) Compute full 3D rotation delta from root rotations at the window boundary.
      2) Apply delta to ALL snippet root rotations (compose in SO(3)).
      3) Rotate snippet translations (relative to first) by delta.
      4) Solve translation offset t* that minimizes mean-squared error against the base window.
         t* = mean(base_window) - mean(rotated_snippet_window)
      5) Apply t* to whole snippet translations.
    """
    W = min(len(base_pose_window), len(pose_snip))
    ps = pose_snip.copy()
    ts = trans_snip.copy()

    # 1) rotation delta from last base vs first snippet frame
    R_now   = aa_to_mat(base_pose_window[-1, :3])
    R_snip0 = aa_to_mat(ps[0, :3])
    R_delta = R_now @ R_snip0.T

    # 2) apply delta to snippet root rotations
    for i in range(ps.shape[0]):
        R_i = aa_to_mat(ps[i, :3])
        ps[i, :3] = mat_to_aa(R_delta @ R_i)

    # 3) rotate snippet translations (relative to first)
    ts_rel = ts - ts[0]
    ts_rot = (R_delta @ ts_rel.T).T

    # 4) windowed translation offset
    #    align the first W frames in LS sense: t* = mean(baseW) - mean(snippetW)
    baseW   = base_trans_window[-W:]                         # (W,3)
    snipW   = ts_rot[:W] + trans_snip[0] * 0.0               # (W,3) (zeroed origin)
    t_star  = baseW.mean(axis=0) - snipW.mean(axis=0)        # (3,)

    # 5) full adjusted snippet translations
    ts_adj = ts_rot + t_star

    return ps, ts_adj
