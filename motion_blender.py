# motion_blender.py
import numpy as np

def blend_poses(pose_a, pose_b, num_steps):
    """Linear blend between two pose vectors (72,)."""
    a = np.asarray(pose_a, dtype=np.float32)
    b = np.asarray(pose_b, dtype=np.float32)
    if num_steps <= 1:
        return [b.astype(np.float32)]
    out = []
    for t in range(num_steps):
        u = t / (num_steps - 1)
        out.append(((1-u)*a + u*b).astype(np.float32))
    return out

def crossfade_tracks(trackA, trackB, fade_len):
    """
    Crossfade two tracks of equal feature dimension:
      trackA: (Ta, D)  (we'll keep all of it)
      trackB: (Tb, D)  (we'll blend-in over first fade_len frames)
    Returns concatenated track: (Ta + Tb - fade_len, D)
    """
    A = np.asarray(trackA, dtype=np.float32)
    B = np.asarray(trackB, dtype=np.float32)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1], "dim mismatch"
    fade_len = max(1, min(fade_len, min(len(A), len(B))))

    # copy A up to the start of fade
    keepA = A[:-fade_len] if fade_len < len(A) else A[:0]

    # overlap region
    cross = []
    for t in range(fade_len):
        u = (t + 1) / fade_len   # 1/fade_len .. 1 (end at full B)
        cross.append(((1-u)*A[-fade_len + t] + u*B[t]).astype(np.float32))
    cross = np.stack(cross, axis=0)

    # tail of B after fade
    tailB = B[fade_len:] if fade_len < len(B) else B[:0]

    parts = [p for p in [keepA, cross, tailB] if len(p) > 0]
    return np.concatenate(parts, axis=0)
