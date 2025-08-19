import os
import csv
import numpy as np
import torch

from motion_loader import load_pose_data_from_folder
from pose_matcher import PoseMatcher
from snippet_utils import load_snippet

# ----------------- config -----------------
DATA_FOLDER     = "ptFiles"
BASE_FILE       = "ptFiles/base_motion.pt"
OUTPUT_FILE     = "ptFiles/pipeline_output.pt"

MATCH_EVERY     = 20
FADE_LEN        = 12
SNIPPET_LEN     = 48
TOP_K           = 12
MIN_DIST        = 1e-6

# === Translation control ===
# "base"   : always follow base path (old behavior)
# "snippet": follow snippet root translation only (no base path)
# "hybrid" : follow base path only when snippet reports meaningful motion; otherwise hold position
PATH_MODE       = "hybrid"      # <- change here if you want
PATH_LOCK       = True          # kept for backward compatibility (ignored if PATH_MODE != "base")

# Motion detection (to avoid gliding on noise)
SPEED_START_TH  = 0.03          # m/frame to START moving
SPEED_STOP_TH   = 0.02          # m/frame to STOP moving (hysteresis < start)
K_CONSEC        = 5             # require k consecutive frames over/under the threshold

# Root rotation handling
ROOT_MODE           = "lock"    # "lock" or "cap"
MAX_DEG_PER_FRAME   = 8.0
POST_MAX_DEG_PER_FRAME = 8.0

# Smoothing
SMOOTH_TRANS_WIN    = 5

# Cooldown to force variety
COOLDOWN_FRAMES     = 500

# === Anti-T thresholds (stronger) ===
MIN_BODY_ENERGY         = 3.0    # L2 norm of 69-dim body; raise to be stricter
MIN_ARM_ENERGY          = 0.6    # minimum L2 in the arm subset
SELECT_WINDOW_FRAMES    = 24     # energy must be above thresholds on average here
CLEAN_WINDOW_FRAMES     = 8

# Optional quick blocklist
BLOCKLIST_SUBSTRINGS = [
    # "bad_clip_name"
]

# ----------------- helpers -----------------
def ease_in_out(u):
    if u <= 0.0: return 0.0
    if u >= 1.0: return 1.0
    return 4*u*u*u if u < 0.5 else 1 - ((-2*u + 2)**3)/2

def crossfade_body_eased(A_body, B_body, fade_len):
    A = np.asarray(A_body, np.float32); B = np.asarray(B_body, np.float32)
    n = max(1, min(fade_len, min(len(A), len(B))))
    keepA = A[:-1]
    cross = []
    last = A[-1].copy()
    for t in range(n):
        u = ease_in_out((t + 1) / n)
        cross.append(((1 - u) * last + u * B[t]).astype(np.float32))
    cross = np.stack(cross, 0)
    tailB = B[n:] if n < len(B) else B[:0]
    return np.concatenate([keepA, cross, tailB], axis=0)

def _aa_to_mat(a):
    a = np.asarray(a, np.float32)
    th = np.linalg.norm(a)
    if th < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = a / th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    return (I + np.sin(th)*K + (1-np.cos(th))*(K@K)).astype(np.float32)

def _mat_to_aa(R):
    R = np.asarray(R, np.float32)
    tr = np.trace(R); c = np.clip((tr - 1.0)*0.5, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8: return np.zeros(3, np.float32)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], np.float32) / (2.0*np.sin(th))
    return (w * th).astype(np.float32)

def _slerp_root_step(a_prev, a_target, max_deg):
    Rprev = _aa_to_mat(a_prev); Rtar = _aa_to_mat(a_target)
    Rdelta = Rtar @ Rprev.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rdelta) - 1.0)/2.0, -1.0, 1.0))))
    if ang <= max_deg + 1e-6: return a_target.astype(np.float32)
    frac = max_deg / max(ang, 1e-6)
    a_delta = _mat_to_aa(Rdelta)
    a_step  = a_delta * frac
    Rstep   = _aa_to_mat(a_step) @ Rprev
    return _mat_to_aa(Rstep).astype(np.float32)

def cap_root_delta_seq(root_start, root_target_seq, max_deg_per_frame):
    out = []; cur = root_start.astype(np.float32).copy()
    for t in range(len(root_target_seq)):
        cur = _slerp_root_step(cur, root_target_seq[t], max_deg_per_frame)
        out.append(cur.copy())
    return np.stack(out, 0).astype(np.float32)

def enforce_root_rate_limit(root_seq, max_deg_per_frame=8.0):
    root = np.asarray(root_seq, np.float32).copy()
    for i in range(1, len(root)):
        root[i] = _slerp_root_step(root[i-1], root[i], max_deg_per_frame)
    return root

def smooth_translation(trans, win=5):
    trans = np.asarray(trans, np.float32)
    if win < 3 or win % 2 == 0 or len(trans) < win: return trans
    k = win // 2; filt = np.ones((win,), np.float32) / win
    sm = []
    for c in range(trans.shape[1]):
        col = trans[:, c]
        pad = np.pad(col, (k, k), mode='edge')
        sm_col = np.convolve(pad, filt, mode='valid')
        sm.append(sm_col)
    return np.stack(sm, axis=1).astype(np.float32)

# --- energy & arms ---
def pose_energy(body_69):
    b = np.asarray(body_69, np.float32)
    return float(np.linalg.norm(b))

# rough arm channel indices in SMPL 69-dim (body only). This is heuristic!
# (kept broad to be robust across sources)
ARM_IDXS = np.r_[15:25, 35:45]  # tweak if needed

def arm_energy(body_69):
    b = np.asarray(body_69, np.float32)
    sub = b[ARM_IDXS] if ARM_IDXS.max() < b.shape[0] else b
    return float(np.linalg.norm(sub))

def snippet_motion_speed(trans_seq):
    T = np.asarray(trans_seq, np.float32)
    if len(T) < 2: return np.zeros((len(T),), np.float32)
    v = np.linalg.norm(T[1:] - T[:-1], axis=1)
    return np.concatenate([[0.0], v]).astype(np.float32)

def scaled_path_segment(base_trans_slice, start_world, speed_scale_seq):
    B = np.asarray(base_trans_slice, np.float32)
    S = np.asarray(speed_scale_seq, np.float32).reshape(-1)
    L = len(B); out = np.empty_like(B); out[0] = start_world
    if L == 1: return out
    dB = B[1:] - B[:-1]
    scale = S[1:].reshape(-1, 1)
    cur = start_world.copy()
    for i in range(1, L):
        cur = cur + dB[i-1] * scale[i-1]
        out[i] = cur
    return out

def is_blocklisted(fname: str) -> bool:
    nm = fname.lower()
    return any(s.lower() in nm for s in BLOCKLIST_SUBSTRINGS)

# hysteresis gate to decide moving vs stopped over a speed sequence
def hysteresis_move_mask(speed, start_th, stop_th, k_consec):
    s = np.asarray(speed, np.float32)
    move = np.zeros_like(s, dtype=bool)
    consec = 0; moving = False
    for i, v in enumerate(s):
        if moving:
            if v <= stop_th: consec += 1
            else: consec = 0
            if consec >= k_consec:
                moving = False; consec = 0
        else:
            if v >= start_th: consec += 1
            else: consec = 0
            if consec >= k_consec:
                moving = True; consec = 0
        move[i] = moving
    return move

# ----------------- pipeline -----------------
def main():
    # base
    base = torch.load(BASE_FILE, map_location="cpu")
    base_pose  = base["pose"].detach().cpu().numpy().astype(np.float32)
    base_trans = base["trans"].detach().cpu().numpy().astype(np.float32)
    base_betas = base["betas"].detach().cpu().numpy().astype(np.float32)
    Tb = base_pose.shape[0]

    # matcher
    pose_list, pose_matrix = load_pose_data_from_folder(DATA_FOLDER)
    try:
        matcher = PoseMatcher(pose_list, pose_matrix, n_neighbors=TOP_K, metric='euclidean', standardize=True)
    except TypeError:
        matcher = PoseMatcher(pose_matrix, n_neighbors=TOP_K, metric='euclidean')

    # outputs
    out_root = base_pose[0:1, :3]
    out_body = base_pose[0:1, 3:]
    out_trans= base_trans[0:1]
    out_betas= base_betas[0:1]

    audit_rows = []
    frame_sources = [0]
    source_file_ids = [0]
    file_to_id = {"__BASE__": 0}
    last_used_by_file = {}

    t = 1
    while t < Tb:
        if t % MATCH_EVERY == 0:
            cur_root = out_root[-1]
            cur_body = out_body[-1]
            query_pose = np.concatenate([cur_root, cur_body], axis=0)

            matches = matcher.query(query_pose, k=TOP_K)
            norm_matches = []
            if isinstance(matches, tuple) and len(matches) == 2:
                dists, idxs = matches
                for d, i in zip(dists, idxs):
                    fname, fidx, _ = pose_list[i]
                    norm_matches.append({"filename": fname, "frame": int(fidx), "dist": float(d)})
            else:
                norm_matches = matches

            chosen, chosen_dist = None, float("inf")
            for m in norm_matches:
                fname = m['filename']
                if is_blocklisted(fname): continue
                if (t - last_used_by_file.get(fname, -1e9)) < COOLDOWN_FRAMES:
                    continue

                dist = m.get('dist', float('inf'))
                center_idx = int(m['frame'])
                try:
                    Lpeek = max(SELECT_WINDOW_FRAMES, FADE_LEN)
                    peek_pose, peek_trans, _ = load_snippet(DATA_FOLDER, fname, center_idx=center_idx, length=Lpeek)
                except Exception:
                    continue

                body = peek_pose[:, 3:]
                mean_body_E = float(np.mean([pose_energy(b) for b in body]))
                mean_arm_E  = float(np.mean([arm_energy(b)  for b in body]))

                if mean_body_E < MIN_BODY_ENERGY:   # strong reject
                    continue
                if mean_arm_E < MIN_ARM_ENERGY:     # strong reject
                    continue

                if dist > MIN_DIST and dist < chosen_dist:
                    chosen, chosen_dist = m, dist

            if chosen is None and len(norm_matches):
                chosen = norm_matches[0]

            if not chosen:
                # append base
                out_root = np.concatenate([out_root, base_pose[t:t+1, :3]], axis=0)
                out_body = np.concatenate([out_body, base_pose[t:t+1, 3:]], axis=0)
                out_trans= np.concatenate([out_trans, base_trans[t:t+1]], axis=0)
                out_betas= np.concatenate([out_betas, base_betas[t:t+1]], axis=0)
                frame_sources.append(0); source_file_ids.append(0)
                t += 1
                continue

            fname, fidx = chosen['filename'], int(chosen['frame'])
            remaining = Tb - t
            use_len = min(SNIPPET_LEN, remaining + FADE_LEN)

            pose_snip, trans_snip, betas_snip = load_snippet(DATA_FOLDER, fname, center_idx=fidx, length=use_len)
            root_snip = pose_snip[:, :3]
            body_snip = pose_snip[:, 3:]

            # ---- strong in-snippet cleanup (anti-T) ----
            base_body_slice = base_pose[t : min(Tb, t + use_len), 3:]
            if len(base_body_slice) < use_len:
                pad = np.repeat(base_body_slice[-1:], use_len - len(base_body_slice), axis=0)
                base_body_slice = np.concatenate([base_body_slice, pad], axis=0)

            last_valid = out_body[-1].copy()
            for k in range(len(body_snip)):
                e_body = pose_energy(body_snip[k])
                e_arm  = arm_energy(body_snip[k])
                if e_body < MIN_BODY_ENERGY or e_arm < MIN_ARM_ENERGY:
                    # Replace whole body (arms included) by stable reference
                    repl = base_body_slice[k]
                    if pose_energy(repl) < MIN_BODY_ENERGY * 0.8:
                        repl = last_valid
                    body_snip[k] = repl
                else:
                    last_valid = body_snip[k].copy()

            # crossfade BODY
            prev_len = out_body.shape[0]
            out_body = crossfade_body_eased(out_body, body_snip, FADE_LEN)

            # ROOT
            cur_root = out_root[-1]
            if ROOT_MODE == "lock":
                base_root_slice = base_pose[t : min(Tb, t + use_len), :3]
                if len(base_root_slice) < use_len:
                    pad = np.repeat(base_root_slice[-1:], use_len - len(base_root_slice), axis=0)
                    base_root_slice = np.concatenate([base_root_slice, pad], axis=0)
                out_root = np.concatenate([out_root[:-1], base_root_slice], axis=0)
            else:
                capped_seq = cap_root_delta_seq(cur_root, root_snip, MAX_DEG_PER_FRAME)
                n = min(FADE_LEN, len(capped_seq))
                keep = out_root[:-1]
                cross = []
                for k in range(n):
                    u = ease_in_out((k + 1) / n)
                    cross.append(((1 - u) * cur_root + u * capped_seq[k]).astype(np.float32))
                cross = np.stack(cross, 0) if n > 0 else np.zeros((0,3), np.float32)
                tail = capped_seq[n:] if n < len(capped_seq) else capped_seq[:0]
                out_root = np.concatenate([keep, cross, tail], axis=0)

            # TRANSLATION — no gliding: depends on PATH_MODE
            if PATH_MODE == "snippet":
                # integrate snippet world deltas only
                seg = np.empty((use_len, 3), np.float32)
                seg[0] = out_trans[-1]
                for i in range(1, use_len):
                    seg[i] = seg[i-1] + (trans_snip[i] - trans_snip[i-1])
                out_trans = np.concatenate([out_trans[:-1], seg], axis=0)

            elif PATH_MODE == "base":
                # legacy (optionally with PATH_LOCK)
                base_trans_slice = base_trans[t : min(Tb, t + use_len)]
                if len(base_trans_slice) < use_len:
                    pad = np.repeat(base_trans_slice[-1:], use_len - len(base_trans_slice), axis=0)
                    base_trans_slice = np.concatenate([base_trans_slice, pad], axis=0)
                out_trans = np.concatenate([out_trans[:-1], base_trans_slice], axis=0)

            else:  # "hybrid"
                base_trans_slice = base_trans[t : min(Tb, t + use_len)]
                if len(base_trans_slice) < use_len:
                    pad = np.repeat(base_trans_slice[-1:], use_len - len(base_trans_slice), axis=0)
                    base_trans_slice = np.concatenate([base_trans_slice, pad], axis=0)

                spd = snippet_motion_speed(trans_snip)
                move_mask = hysteresis_move_mask(spd, SPEED_START_TH, SPEED_STOP_TH, K_CONSEC)

                seg = np.empty((use_len, 3), np.float32)
                seg[0] = out_trans[-1]
                for i in range(1, use_len):
                    if move_mask[i]:
                        seg[i] = seg[i-1] + (base_trans_slice[i] - base_trans_slice[i-1])
                    else:
                        seg[i] = seg[i-1]  # hold position (no glide)
                out_trans = np.concatenate([out_trans[:-1], seg], axis=0)

            # BETAS fade
            keepB = out_betas[:-1]; betaA = out_betas[-1]
            nB = min(FADE_LEN, len(betas_snip))
            crossB = []
            for k in range(nB):
                u = ease_in_out((k + 1) / max(1, FADE_LEN))
                crossB.append(((1 - u) * betaA + u * betas_snip[k]).astype(np.float32))
            crossB = np.stack(crossB, 0) if nB > 0 else np.zeros((0,10), np.float32)
            tailB = betas_snip[nB:] if nB < len(betas_snip) else betas_snip[:0]
            out_betas = np.concatenate([keepB, crossB, tailB], axis=0)

            # sources
            added = out_body.shape[0] - prev_len
            fade = min(FADE_LEN, added)
            frame_sources.extend([1]*fade + [2]*(added - fade))
            if fname not in file_to_id:
                file_to_id[fname] = len(file_to_id) + 0
            this_id = file_to_id[fname]
            source_file_ids.extend([this_id]*added)

            # post-append anti-T guard
            start_new = prev_len
            for i in range(start_new, start_new + added):
                if pose_energy(out_body[i]) < MIN_BODY_ENERGY or arm_energy(out_body[i]) < MIN_ARM_ENERGY:
                    out_body[i] = out_body[i-1]

            # audit
            audit_rows.append({
                "switch_at_base_frame": int(t),
                "selected_file": fname,
                "selected_center_frame": int(fidx),
                "neighbor_dist": float(chosen_dist),
                "fade_len": int(FADE_LEN),
                "snippet_len_used": int(use_len),
                "root_mode": ROOT_MODE,
                "path_mode": PATH_MODE,
                "cooldown_frames": int(COOLDOWN_FRAMES),
            })

            last_used_by_file[fname] = out_body.shape[0] - 1
            t = min(Tb, t + use_len)

        else:
            out_root = np.concatenate([out_root, base_pose[t:t+1, :3]], axis=0)
            out_body = np.concatenate([out_body, base_pose[t:t+1, 3:]], axis=0)
            out_trans= np.concatenate([out_trans, base_trans[t:t+1]], axis=0)
            out_betas= np.concatenate([out_betas, base_betas[t:t+1]], axis=0)
            frame_sources.append(0); source_file_ids.append(0)
            t += 1

    # post-passes
    out_root = enforce_root_rate_limit(out_root, max_deg_per_frame=POST_MAX_DEG_PER_FRAME)
    out_trans = smooth_translation(out_trans, win=SMOOTH_TRANS_WIN)

    out_pose = np.concatenate([out_root, out_body], axis=1).astype(np.float32)

    # align vectors
    T_out = out_pose.shape[0]
    if len(frame_sources) < T_out: frame_sources += [frame_sources[-1]]*(T_out-len(frame_sources))
    elif len(frame_sources) > T_out: frame_sources = frame_sources[:T_out]
    if len(source_file_ids) < T_out: source_file_ids += [source_file_ids[-1]]*(T_out-len(source_file_ids))
    elif len(source_file_ids) > T_out: source_file_ids = source_file_ids[:T_out]

    bundle = {
        "pose":   torch.from_numpy(out_pose),
        "trans":  torch.from_numpy(out_trans.astype(np.float32)),
        "betas":  torch.from_numpy(out_betas.astype(np.float32)),
        "source": torch.tensor(frame_sources, dtype=torch.int16),
        "source_file_id": torch.tensor(source_file_ids, dtype=torch.int16),
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(bundle, OUTPUT_FILE)

    audit_csv = os.path.join(os.path.dirname(OUTPUT_FILE), "pipeline_timeline.csv")
    with open(audit_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "switch_at_base_frame","selected_file","selected_center_frame",
            "neighbor_dist","fade_len","snippet_len_used","root_mode","path_mode",
            "cooldown_frames"
        ])
        w.writeheader(); [w.writerow(r) for r in audit_rows]

    print(f"Saved {OUTPUT_FILE} (T={T_out}) and {audit_csv}; PATH_MODE={PATH_MODE}")

if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)
    main()
