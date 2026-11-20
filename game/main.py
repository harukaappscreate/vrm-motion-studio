import json
import math
import numpy as np
import pyxel
from pyxel.cube import Camera, Mat4, Node, Primitive, Vec3
from vrm_figure import Figure, qmul, qrot

# Bone indices in model.tmx / motions.tmx (52 humanoid bones)
BONE_IDX = {
    "Hips": 2,
    "Spine": 50,
    "Chest": 0,
    "Neck": 26,
    "Head": 1,
    "LeftUpperArm": 24,
    "LeftLowerArm": 11,
    "LeftHand": 4,
    "RightUpperArm": 48,
    "RightLowerArm": 35,
    "RightHand": 28,
    "LeftUpperLeg": 25,
    "LeftLowerLeg": 12,
    "LeftFoot": 3,
    "RightUpperLeg": 49,
    "RightLowerLeg": 36,
    "RightFoot": 27,
}

# Skeleton lines for debug overlay: (bone_a, bone_b, pyxel_color)
SKELETON_LINKS = [
    ("Hips", "Spine", 10),
    ("Spine", "Chest", 10),
    ("Chest", "Neck", 10),
    ("Neck", "Head", 10),
    ("Chest", "LeftUpperArm", 12),
    ("LeftUpperArm", "LeftLowerArm", 12),
    ("LeftLowerArm", "LeftHand", 12),
    ("Chest", "RightUpperArm", 8),
    ("RightUpperArm", "RightLowerArm", 8),
    ("RightLowerArm", "RightHand", 8),
    ("Hips", "LeftUpperLeg", 12),
    ("LeftUpperLeg", "LeftLowerLeg", 12),
    ("LeftLowerLeg", "LeftFoot", 12),
    ("Hips", "RightUpperLeg", 8),
    ("RightUpperLeg", "RightLowerLeg", 8),
    ("RightLowerLeg", "RightFoot", 8),
]

# Topological propagation order from root to extremities
CHAIN = [
    ("Hips", None),
    ("Spine", "Hips"),
    ("Chest", "Spine"),
    ("Neck", "Chest"),
    ("Head", "Neck"),
    ("LeftUpperArm", "Chest"),
    ("LeftLowerArm", "LeftUpperArm"),
    ("LeftHand", "LeftLowerArm"),
    ("RightUpperArm", "Chest"),
    ("RightLowerArm", "RightUpperArm"),
    ("RightHand", "RightLowerArm"),
    ("LeftUpperLeg", "Hips"),
    ("LeftLowerLeg", "LeftUpperLeg"),
    ("RightUpperLeg", "Hips"),
    ("RightLowerLeg", "RightUpperLeg"),
]


def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix (columns = X, Y, Z axes) to quaternion [x, y, z, w]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def quat_from_two_vectors(u, v):
    """Shortest-arc quaternion rotating unit vector u to unit vector v."""
    u = u / max(np.linalg.norm(u), 1e-6)
    v = v / max(np.linalg.norm(v), 1e-6)
    d = float(np.dot(u, v))
    if d > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if d < -0.9999:
        axis = np.cross(np.array([1.0, 0.0, 0.0]), u)
        if np.linalg.norm(axis) < 1e-4:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), u)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    c = np.cross(u, v)
    q = np.array([c[0], c[1], c[2], 1.0 + d], dtype=np.float32)
    return q / np.linalg.norm(q)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float64)


def qslerp(a, b, t):
    """Shortest-path normalized lerp between quaternions a and b."""
    if float(np.dot(a, b)) < 0.0:
        b = -b
    q = a + (b - a) * t
    n = np.linalg.norm(q)
    return (q / n).astype(np.float32) if n > 1e-6 else a.astype(np.float32)


def solve_leg_ik(leg_vec, pole_vec, fwd_fallback, L1=0.353, L2=0.415):
    """2-bone analytical leg IK returning (thigh_dir, shin_dir) with forward-only knee hinge."""
    d_norm = float(np.linalg.norm(leg_vec))
    d = max(0.55 * (L1 + L2), min(L1 + L2 - 0.005, d_norm))
    u = leg_vec / max(d_norm, 1e-6)
    fwd_perp = fwd_fallback - float(np.dot(fwd_fallback, u)) * u
    if np.linalg.norm(fwd_perp) < 1e-4:
        fwd_perp = np.array([0.0, 0.0, 1.0], dtype=np.float64) - float(u[2]) * u
    fwd_perp = _norm(fwd_perp)
    p_perp = pole_vec - float(np.dot(pole_vec, u)) * u
    if np.linalg.norm(p_perp) < 1e-4 or float(np.dot(p_perp, fwd_perp)) <= 0.0:
        n_fwd = fwd_perp
    else:
        n_fwd = _norm(0.35 * _norm(p_perp) + 0.65 * fwd_perp)
    cos_a = max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    knee_pos = L1 * (cos_a * u + sin_a * n_fwd)
    ankle_pos = d * u
    shin_vec = ankle_pos - knee_pos
    if shin_vec[1] > -0.15 * L2:
        shin_vec[1] = -0.15 * L2
    return knee_pos, shin_vec


def _flat_foot_quat(fwd_vec):
    """Horizontal yaw-only orientation matching fwd_vec projected onto the XZ floor."""
    hz_f = np.array([float(fwd_vec[0]), 0.0, float(fwd_vec[2])], dtype=np.float64)
    if float(np.linalg.norm(hz_f)) < 1e-4:
        hz_f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        hz_f = _norm(hz_f)
    hy_f = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    hx_f = _norm(np.cross(hy_f, hz_f))
    return rotmat_to_quat(np.column_stack([hx_f, hy_f, hz_f]))


def world_landmarks_to_D(
    lm3d,
    prev_hips_fwd=None,
    cam_pitch=0.0,
    prev_hx=None,
    prev_sx=None,
    lm2d=None,
    leg_l1=0.353,
    leg_l2=0.415,
):
    """Build 52-bone world rotation delta D directly from MediaPipe worldLandmarks."""
    D = np.zeros((52, 4), dtype=np.float32)
    D[:, 3] = 1.0

    # Convert MediaPipe world coords (+X right, +Y down, +Z away)
    # to VRM 1.0 world coords (+X right, +Y up, +Z toward camera)
    P = {int(k): np.array([v[0], -v[1], -v[2]], dtype=np.float64) for k, v in lm3d.items()}
    required = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    if not all(idx in P for idx in required):
        return D, np.zeros(3, dtype=np.float32), {}

    # Refine camera-plane (X, Y) arm/hand positions from 2D image landmarks while keeping 3D depth Z
    if lm2d:
        L2 = {int(k): np.array([v[0], v[1]], dtype=np.float64) for k, v in lm2d.items()}
        if all(idx in L2 for idx in (11, 12, 23, 24)):
            c3 = np.mean([P[k][:2] for k in (11, 12, 23, 24)], axis=0)
            c2 = np.mean([L2[k] for k in (11, 12, 23, 24)], axis=0)
            r3 = math.sqrt(sum(float(np.sum((P[k][:2] - c3) ** 2)) for k in (11, 12, 23, 24)) / 4.0)
            r2 = math.sqrt(sum(float(np.sum((L2[k] - c2) ** 2)) for k in (11, 12, 23, 24)) / 4.0)
            if r2 > 0.04 and r3 > 0.12:
                s23 = r3 / r2
                for sh, el, wr, pink, ind in ((11, 13, 15, 17, 19), (12, 14, 16, 18, 20)):
                    orig_el = P[el].copy()
                    orig_wr = P[wr].copy()
                    if el in L2 and sh in L2:
                        target_el_xy = P[sh][:2] + (L2[el] - L2[sh]) * s23
                        blended_xy = P[el][:2] * 0.45 + target_el_xy * 0.55
                        max_r = max(float(np.linalg.norm(orig_el - P[sh])) * 1.05, 0.10)
                        rel_xy = blended_xy - P[sh][:2]
                        rn = float(np.linalg.norm(rel_xy))
                        if rn > max_r:
                            blended_xy = P[sh][:2] + rel_xy * (max_r / rn)
                        P[el][0], P[el][1] = blended_xy[0], blended_xy[1]
                    if wr in L2 and el in L2:
                        target_wr_xy = P[el][:2] + (L2[wr] - L2[el]) * s23
                        blended_xy = P[wr][:2] * 0.40 + target_wr_xy * 0.60
                        max_r = max(float(np.linalg.norm(orig_wr - orig_el)) * 1.05, 0.10)
                        rel_xy = blended_xy - P[el][:2]
                        rn = float(np.linalg.norm(rel_xy))
                        if rn > max_r:
                            blended_xy = P[el][:2] + rel_xy * (max_r / rn)
                        P[wr][0], P[wr][1] = blended_xy[0], blended_xy[1]
                    for hk in (pink, ind):
                        if hk in P and hk in L2 and wr in L2:
                            orig_hk = P[hk].copy()
                            target_hk_xy = P[wr][:2] + (L2[hk] - L2[wr]) * s23
                            blended_xy = P[hk][:2] * 0.40 + target_hk_xy * 0.60
                            max_r = max(float(np.linalg.norm(orig_hk - orig_wr)) * 1.10, 0.04)
                            rel_xy = blended_xy - P[wr][:2]
                            rn = float(np.linalg.norm(rel_xy))
                            if rn > max_r:
                                blended_xy = P[wr][:2] + rel_xy * (max_r / rn)
                            P[hk][0], P[hk][1] = blended_xy[0], blended_xy[1]

    # Estimate camera downward pitch from straight-legged upright frames where legs and spine tilt together
    obs_cam_pitch = None
    mid_hip_raw = 0.5 * (P[23] + P[24])
    mid_sh_raw = 0.5 * (P[11] + P[12])
    mid_ank_raw = 0.5 * (P[27] + P[28])
    l_raw_len = float(np.linalg.norm(P[25] - P[23]) + np.linalg.norm(P[27] - P[25]))
    r_raw_len = float(np.linalg.norm(P[26] - P[24]) + np.linalg.norm(P[28] - P[26]))
    l_str = float(np.linalg.norm(P[27] - P[23])) / max(l_raw_len, 1e-6)
    r_str = float(np.linalg.norm(P[28] - P[24])) / max(r_raw_len, 1e-6)
    leg_up_raw = mid_hip_raw - mid_ank_raw
    spine_raw = mid_sh_raw - mid_hip_raw
    if l_str > 0.94 and r_str > 0.94 and leg_up_raw[1] > 0.55 and spine_raw[1] > 0.35:
        leg_p = math.atan2(float(leg_up_raw[2]), float(leg_up_raw[1]))
        sp_p = math.atan2(float(spine_raw[2]), float(spine_raw[1]))
        if abs(leg_p - sp_p) < 0.18 and abs(leg_p) < 0.38:
            obs_cam_pitch = 0.5 * (leg_p + sp_p)

    # Apply camera-pitch leveling around camera X axis
    applied_pitch = obs_cam_pitch if (cam_pitch == 0.0 and obs_cam_pitch is not None) else cam_pitch
    if abs(applied_pitch) > 1e-4:
        cp, sp = math.cos(applied_pitch), math.sin(applied_pitch)
        for k, pt in P.items():
            P[k] = np.array([pt[0], pt[1] * cp + pt[2] * sp, -pt[1] * sp + pt[2] * cp], dtype=np.float64)

    def _swap_upper():
        for a, b in ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (7, 8)):
            if a in P and b in P:
                P[a], P[b] = P[b], P[a]

    # Detect & repair MediaPipe single-frame / short-burst Left<->Right landmark swaps (e.g. during deep forward bends)
    hip_span = float(np.linalg.norm(P[23] - P[24]))
    hx_raw0 = _norm(P[23] - P[24])
    sx_raw0 = _norm(P[11] - P[12])
    ank_span = float(np.linalg.norm(P[27] - P[28]))
    ank_x0 = _norm(P[27] - P[28])
    lr_swapped = False
    if prev_hx is not None:
        hip_flip = (hip_span >= 0.08 and float(np.dot(hx_raw0, prev_hx)) < -0.70)
        ank_flip = (ank_span > 0.14 and float(np.dot(ank_x0, prev_hx)) < -0.70)
        sh_flip = (prev_sx is not None and abs(float(sx_raw0[1])) < 0.7 and float(np.dot(sx_raw0, prev_sx)) < -0.70)
        if hip_flip or (hip_span < 0.08 and ank_flip):
            lr_swapped = True
            P[23], P[24] = P[24], P[23]
            if ank_flip:
                P[25], P[26] = P[26], P[25]
                P[27], P[28] = P[28], P[27]
                if 31 in P and 32 in P:
                    P[31], P[32] = P[32], P[31]
            if sh_flip:
                _swap_upper()
        elif sh_flip:
            lr_swapped = True
            _swap_upper()

    mid_hip = 0.5 * (P[23] + P[24])
    # Level whole-body global pitch when standing leg axis tilts > 10 deg from vertical
    # (prevents MediaPipe worldLandmarks monocular depth pitch errors / "plank" poses)
    l_len2 = float(np.linalg.norm(P[25] - P[23]) + np.linalg.norm(P[27] - P[25]))
    r_len2 = float(np.linalg.norm(P[26] - P[24]) + np.linalg.norm(P[28] - P[26]))
    l_str2 = float(np.linalg.norm(P[27] - P[23])) / max(l_len2, 1e-6)
    r_str2 = float(np.linalg.norm(P[28] - P[24])) / max(r_len2, 1e-6)
    if abs(float(P[27][1] - P[28][1])) < 0.14:
        stand_leg_up = 0.5 * ((P[23] - P[27]) + (P[24] - P[28]))
        stand_str = min(l_str2, r_str2)
    elif float(P[27][1]) < float(P[28][1]):
        stand_leg_up = P[23] - P[27]
        stand_str = l_str2
    else:
        stand_leg_up = P[24] - P[28]
        stand_str = r_str2
    gravity_leveled_deg = 0.0
    spine_dir0 = _norm(0.5 * (P[11] + P[12]) - mid_hip)
    if stand_leg_up[1] > 0.20 and stand_str > 0.85 and spine_dir0[1] > 0.35:
        leg_dir = _norm(stand_leg_up)
        if float(np.dot(spine_dir0, leg_dir)) > 0.65:
            theta_leg = math.acos(max(-1.0, min(1.0, float(leg_dir[1]))))
            if theta_leg > 0.175:
                q_full = quat_from_two_vectors(leg_dir, np.array([0.0, 1.0, 0.0], dtype=np.float64))
                frac = (theta_leg - 0.175) / theta_leg
                q_lvl = qslerp(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), q_full, frac)
                for k, pt in P.items():
                    P[k] = mid_hip + qrot(q_lvl, pt - mid_hip)
                gravity_leveled_deg = math.degrees(theta_leg - 0.175)

    mid_hip = 0.5 * (P[23] + P[24])
    mid_sh = 0.5 * (P[11] + P[12])

    # Upper torso forward reference (used to prevent Hips from locking 180-deg opposite to Chest)
    spine_up = _norm(mid_sh - mid_hip)
    sx_raw = _norm(P[11] - P[12])
    sy = spine_up
    fwd_sh_raw = _norm(np.cross(sx_raw, sy))
    w_sh = max(0.0, min(1.0, 1.0 - abs(float(sx_raw[1])) * 1.3))

    # Hips orthonormal basis: stabilize lateral axis when hip landmarks collapse during forward bends (< 8cm)
    hx = _norm(P[23] - P[24])
    if hip_span < 0.08 and prev_hx is not None:
        hx_ref = ank_x0 if (ank_span > 0.14 and float(np.dot(ank_x0, prev_hx)) > 0.0) else prev_hx
        hx = _norm(0.3 * hx + 0.7 * hx_ref)
    spine_up_safe = np.array([spine_up[0], max(0.0, float(spine_up[1])), spine_up[2]], dtype=np.float64)
    hy = _norm(0.15 * spine_up_safe + 0.85 * np.array([0.0, 1.0, 0.0], dtype=np.float64))
    hz = _norm(np.cross(hx, hy))
    hip_flip_guarded = False
    # Guard against 180-degree single-frame L/R hip swap in MediaPipe without locking into an inverted state
    # Compare lateral axes (hx vs sx_raw) rather than forward axes so deep forward bends (>90 deg) don't false-trigger
    if prev_hips_fwd is not None and float(np.dot(hz, prev_hips_fwd)) < -0.3:
        if w_sh < 0.4 or float(np.dot(-hx, sx_raw)) > float(np.dot(hx, sx_raw)):
            hx = -hx
            hz = -hz
            hip_flip_guarded = True
    if w_sh >= 0.4 and float(np.dot(hx, sx_raw)) < -0.25:
        hx = -hx
        hz = -hz
        hip_flip_guarded = True
    hy = _norm(np.cross(hz, hx))
    q_hips = rotmat_to_quat(np.column_stack([hx, hy, hz]))

    # Upper torso (Chest/Spine): stabilize yaw during sideways bends where shoulder vector tilts vertically
    sz_ref = _norm(w_sh * fwd_sh_raw + (1.0 - w_sh) * hz)
    sx = _norm(np.cross(sy, sz_ref))
    sz = _norm(np.cross(sx, sy))
    q_chest = rotmat_to_quat(np.column_stack([sx, sy, sz]))

    D[BONE_IDX["Hips"]] = q_hips
    D[BONE_IDX["Spine"]] = qslerp(q_hips, q_chest, 0.50)
    D[BONE_IDX["Chest"]] = q_chest
    D[51] = q_chest  # upperChest
    D[19] = q_chest  # leftShoulder
    D[43] = q_chest  # rightShoulder
    D[BONE_IDX["Neck"]] = q_chest
    D[BONE_IDX["Head"]] = q_chest

    # Orient Neck & Head from nose (0) and ears (7, 8) when clearly tracked
    if 0 in P and 7 in P and 8 in P:
        ear_vec = P[7] - P[8]
        ear_span = float(np.linalg.norm(ear_vec))
        mid_ear = 0.5 * (P[7] + P[8])
        nose_vec = P[0] - mid_ear
        if ear_span > 0.08 and float(np.linalg.norm(nose_vec)) > 0.04:
            hx_hd = _norm(ear_vec)
            if float(np.dot(hx_hd, sx)) > 0.2:
                hz_hd = _norm(np.array([nose_vec[0], nose_vec[1] * 0.45, nose_vec[2]], dtype=np.float64))
                hy_hd = _norm(np.cross(hz_hd, hx_hd))
                hz_hd = _norm(np.cross(hx_hd, hy_hd))
                q_head_raw = rotmat_to_quat(np.column_stack([hx_hd, hy_hd, hz_hd]))
                d_hc = abs(float(np.dot(q_chest, q_head_raw)))
                ang_hc = 2.0 * math.acos(min(1.0, d_hc))
                max_head = math.radians(55.0)
                if ang_hc > max_head and ang_hc > 1e-6:
                    q_head_raw = qslerp(q_chest, q_head_raw, max_head / ang_hc)
                q_head = qslerp(q_chest, q_head_raw, 0.75)
                D[BONE_IDX["Neck"]] = qslerp(q_chest, q_head, 0.45)
                D[BONE_IDX["Head"]] = q_head

    def align_dir(bname, q_base, rest_dir, tdir):
        tdir = _norm(tdir)
        cur_dir = qrot(q_base.astype(np.float64), rest_dir)
        q_sw = quat_from_two_vectors(cur_dir, tdir)
        q_out = qmul(q_sw, q_base).astype(np.float32)
        D[BONE_IDX[bname]] = q_out
        return q_out

    def solve_arm_dirs(p_sh, p_el, p_wr):
        u_dir = _norm(p_el - p_sh)
        l_dir = _norm(p_wr - p_el)
        arm_dir = _norm(p_wr - p_sh)
        dot_el = float(np.dot(u_dir, l_dir))
        if dot_el > 0.82:
            t = min(1.0, (dot_el - 0.82) / 0.10)
            u_dir = _norm(u_dir + (arm_dir - u_dir) * t)
            l_dir = _norm(l_dir + (arm_dir - l_dir) * t)
        return u_dir, l_dir

    lu_dir, la_dir = solve_arm_dirs(P[11], P[13], P[15])
    q_lu = align_dir("LeftUpperArm", q_chest, np.array([1.0, 0.0, 0.0]), lu_dir)
    q_la = align_dir("LeftLowerArm", q_lu, np.array([1.0, 0.0, 0.0]), la_dir)
    D[BONE_IDX["LeftHand"]] = q_la
    for fidx in (5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22):
        D[fidx] = q_la

    ru_dir, ra_dir = solve_arm_dirs(P[12], P[14], P[16])
    q_ru = align_dir("RightUpperArm", q_chest, np.array([-1.0, 0.0, 0.0]), ru_dir)
    q_ra = align_dir("RightLowerArm", q_ru, np.array([-1.0, 0.0, 0.0]), ra_dir)
    D[BONE_IDX["RightHand"]] = q_ra
    for fidx in (29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 46):
        D[fidx] = q_ra

    # 2-bone analytical leg IK so both feet stay level on the ground when ankles match in video
    L1, L2 = float(leg_l1), float(leg_l2)
    mp_l_len = max(np.linalg.norm(P[25] - P[23]) + np.linalg.norm(P[27] - P[25]), 0.25)
    mp_r_len = max(np.linalg.norm(P[26] - P[24]) + np.linalg.norm(P[28] - P[26]), 0.25)
    leg_scale = (L1 + L2) / (0.5 * (mp_l_len + mp_r_len))

    hz_horiz = _norm(np.array([hz[0], 0.0, hz[2]], dtype=np.float64))

    def fix_standing_leg(raw, mp_len):
        n = max(float(np.linalg.norm(raw)), 1e-6)
        vert = max(0.0, -float(raw[1]) / n)
        v = raw.copy()
        back_z = float(np.dot(v, hz_horiz))
        # Clamp monocular leg depth along body forward so standing/dancing legs cannot flip forward/backward
        if v[1] < -0.10:
            clamped_z = max(-0.14, min(0.16, back_z))
            if abs(clamped_z - back_z) > 1e-5:
                v = v + (clamped_z - back_z) * hz_horiz
                horiz_sq = float(v[0] * v[0] + v[2] * v[2])
                v[1] = -math.sqrt(max(0.09, n * n - horiz_sq))
                back_z = clamped_z
        # During forward bends (spine_up[1] < 0.45), MediaPipe often pushes ankles 40-60cm backward;
        # keep feet under the hips while preserving the torso's forward bend.
        if spine_up[1] < 0.45 and back_z < -0.16 and v[1] < 0.0:
            v = v + (-0.16 - back_z) * hz_horiz
            horiz_sq = float(v[0] * v[0] + v[2] * v[2])
            v[1] = -math.sqrt(max(0.04, n * n - horiz_sq))
        # Compensate MediaPipe's systematic backward ankle Z bias along horizontal forward when standing upright
        elif vert > 0.88:
            w = min(1.0, (vert - 0.88) / 0.07)
            if back_z < 0.0:
                v = v - max(back_z, -0.20 * w) * hz_horiz
        # Extend nearly-straight legs (>94% of thigh+shin) to full standing length
        straightness = n / max(mp_len, 1e-6)
        vec = v * leg_scale
        if straightness > 0.94:
            t_ext = min(1.0, (straightness - 0.94) / 0.04)
            vn = float(np.linalg.norm(vec))
            if vn > 1e-6:
                target_len = vn + ((L1 + L2 - 0.005) - vn) * t_ext
                vec = vec * (target_len / vn)
        return vec

    l_vec = fix_standing_leg(P[27] - P[23], mp_l_len)
    r_vec = fix_standing_leg(P[28] - P[24], mp_r_len)
    # If ankles are nearly level in video (< 12cm apart), damp front-to-back depth split and match vertical drop
    if abs(float(P[27][1] - P[28][1])) < 0.12:
        lz = float(np.dot(l_vec, hz_horiz))
        rz = float(np.dot(r_vec, hz_horiz))
        avg_z = 0.5 * (lz + rz)
        l_vec = l_vec + (avg_z - lz) * 0.60 * hz_horiz
        r_vec = r_vec + (avg_z - rz) * 0.60 * hz_horiz
        avg_y = 0.5 * (l_vec[1] + r_vec[1])
        l_vec[1] = avg_y
        r_vec[1] = avg_y

    q_foot_flat = _flat_foot_quat(hz_horiz)
    min_ank_y = min(float(P[27][1]), float(P[28][1]))

    lt_dir, ls_dir = solve_leg_ik(l_vec, P[25] - P[23], hz, L1, L2)
    q_ll = align_dir("LeftUpperLeg", q_hips, np.array([0.0, -1.0, 0.0]), lt_dir)
    q_lla = align_dir("LeftLowerLeg", q_ll, np.array([0.0, -1.0, 0.0]), ls_dir)
    l_lift = max(0.0, min(1.0, (float(P[27][1]) - min_ank_y - 0.05) / 0.12))
    q_lfoot = qslerp(q_foot_flat, q_lla, l_lift)
    D[BONE_IDX["LeftFoot"]] = q_lfoot
    D[23] = q_lfoot  # leftToes

    rt_dir, rs_dir = solve_leg_ik(r_vec, P[26] - P[24], hz, L1, L2)
    q_rl = align_dir("RightUpperLeg", q_hips, np.array([0.0, -1.0, 0.0]), rt_dir)
    q_rla = align_dir("RightLowerLeg", q_rl, np.array([0.0, -1.0, 0.0]), rs_dir)
    r_lift = max(0.0, min(1.0, (float(P[28][1]) - min_ank_y - 0.05) / 0.12))
    q_rfoot = qslerp(q_foot_flat, q_rla, r_lift)
    D[BONE_IDX["RightFoot"]] = q_rfoot
    D[47] = q_rfoot  # rightToes

    # Auto-grounding is applied in App after pose(); start with hp = 0
    hp = np.zeros(3, dtype=np.float32)
    dbg_meta = {
        "derived": {
            "hx": [round(float(v), 4) for v in hx],
            "hz": [round(float(v), 4) for v in hz],
            "sx_raw": [round(float(v), 4) for v in sx_raw],
            "sz": [round(float(v), 4) for v in sz],
            "w_sh": round(float(w_sh), 3),
            "hip_flip_guarded": hip_flip_guarded,
            "lr_swapped": lr_swapped,
            "cam_pitch_deg": round(math.degrees(applied_pitch), 2),
            "gravity_leveled_deg": round(gravity_leveled_deg, 2),
        },
        "_obs_cam_pitch": obs_cam_pitch,
        "_hx": hx,
        "_sx": sx_raw,
        "mp_lengths": {
            "l_thigh": round(float(np.linalg.norm(P[25] - P[23])), 4),
            "r_thigh": round(float(np.linalg.norm(P[26] - P[24])), 4),
            "l_shin": round(float(np.linalg.norm(P[27] - P[25])), 4),
            "r_shin": round(float(np.linalg.norm(P[28] - P[26])), 4),
        },
    }
    return D, hp, dbg_meta


def euler_xyz_to_quat(rx, ry, rz):
    """Convert Three.js XYZ Euler angles (radians) to quaternion [x, y, z, w]."""
    hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    return np.array([
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    ], dtype=np.float32)


def kalidokit_to_D(pose_data):
    """Convert Kalidokit.Pose.solve() dict to 52-bone world delta D and hips offset hp."""
    D = np.zeros((52, 4), dtype=np.float32)
    D[:, 3] = 1.0
    world_delta = {}
    axis_mode = pose_data.get("axis", "direct")

    for name, parent in CHAIN:
        if name == "Spine":
            rot = pose_data.get("Spine")
            rot = [rot[0] * 0.5, rot[1] * 0.5, rot[2] * 0.5] if rot is not None else None
        elif name == "Chest":
            rot = pose_data.get("Spine")
            rot = [rot[0] * 0.3, rot[1] * 0.3, rot[2] * 0.3] if rot is not None else None
        else:
            rot = pose_data.get(name)
        if rot is not None:
            q = euler_xyz_to_quat(float(rot[0]), float(rot[1]), float(rot[2]))
            if axis_mode == "flip_xz":
                q_vrm1 = np.array([-q[0], q[1], -q[2], q[3]], dtype=np.float32)
            elif axis_mode == "flip_yz":
                q_vrm1 = np.array([q[0], -q[1], -q[2], q[3]], dtype=np.float32)
            elif axis_mode == "flip_xy":
                q_vrm1 = np.array([-q[0], -q[1], q[2], q[3]], dtype=np.float32)
            else:
                q_vrm1 = q
        else:
            q_vrm1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if parent is None:
            w = q_vrm1
        else:
            w = qmul(world_delta[parent], q_vrm1).astype(np.float32)
        world_delta[name] = w
        D[BONE_IDX[name]] = w

    # Propagate Chest, Hand, and LowerLeg deltas to all remaining 37 humanoid bones (upperChest, shoulders, fingers, feet, toes)
    D[51] = D[19] = D[43] = D[BONE_IDX["Chest"]]
    for fidx in (5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22):
        D[fidx] = D[BONE_IDX["LeftHand"]]
    for fidx in (29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 46):
        D[fidx] = D[BONE_IDX["RightHand"]]
    D[BONE_IDX["LeftFoot"]] = D[23] = D[BONE_IDX["LeftLowerLeg"]]
    D[BONE_IDX["RightFoot"]] = D[47] = D[BONE_IDX["RightLowerLeg"]]

    hp_raw = pose_data.get("HipsPos")
    if hp_raw is not None:
        # Kalidokit Hips.position is relative offset around 0
        hp = np.array([-float(hp_raw[0]), float(hp_raw[1]), -float(hp_raw[2])], dtype=np.float32)
        hp = np.clip(hp, -0.5, 0.5)
    else:
        hp = np.zeros(3, dtype=np.float32)

    return D, hp


class Ground(Node):
    """Checkerboard floor at y = 0.0 using Pyxel palette colors 0..15."""

    def __init__(self, half_tiles=4, tile_size=0.5):
        super().__init__()
        p_a, i_a = [], []
        p_b, i_b = [], []
        for gx in range(-half_tiles, half_tiles):
            for gz in range(-half_tiles, half_tiles):
                x0, x1 = gx * tile_size, (gx + 1) * tile_size
                z0, z1 = gz * tile_size, (gz + 1) * tile_size
                pts, idxs = (p_a, i_a) if ((gx + gz) & 1) == 0 else (p_b, i_b)
                base = len(pts) // 3
                pts.extend([x0, 0.0, z0, x1, 0.0, z0, x1, 0.0, z1, x0, 0.0, z1])
                idxs.extend([base, base + 2, base + 1, base, base + 3, base + 2])
        self.qa = Primitive(Primitive.MODE_TRIANGLES, p_a, i_a, cull=Primitive.CULL_NONE)
        self.qb = Primitive(Primitive.MODE_TRIANGLES, p_b, i_b, cull=Primitive.CULL_NONE)
        self.qa.compute_normals()
        self.qb.compute_normals()

    def on_draw(self):
        m = Mat4()
        self.prim(m, self.qa, 5)
        self.prim(m, self.qb, 13)


class FootShadow(Node):
    """Simple contact shadow discs at y = 0.004 under LeftFoot and RightFoot."""

    def __init__(self, fig, radius=0.09):
        super().__init__()
        self.fig = fig
        pts = [0.0, 0.0, 0.0]
        idxs = []
        segs = 8
        for k in range(segs):
            a = 2.0 * math.pi * k / segs
            pts.extend([math.cos(a) * radius, 0.0, math.sin(a) * radius])
            idxs.extend([0, 1 + ((k + 1) % segs), 1 + k])
        self.disc = Primitive(Primitive.MODE_TRIANGLES, pts, idxs, cull=Primitive.CULL_NONE)
        self.disc.compute_normals()

    def on_draw(self):
        if getattr(self.fig, "world_mats", None) is None:
            return
        for bkey in ("LeftFoot", "RightFoot"):
            nidx = self.fig.bone_nodes[BONE_IDX[bkey]]
            if nidx >= 0:
                p = self.fig.world_pos(nidx)
                self.prim(Mat4.from_translation(Vec3(float(p[0]), 0.004, float(p[2]))), self.disc, 1)


class App:
    def __init__(self):
        pyxel.init(160, 120, title="Pyxel 3 VRM Motion Studio", fps=30)
        pyxel.mouse(True)

        self.stage = Node()
        self.stage.add_child(Ground())
        self.fig = Figure("")
        self.foot_shadow = FootShadow(self.fig)
        self.stage.add_child(self.foot_shadow)
        self.stage.add_child(self.fig)

        self.cam = Camera()
        self.cam.fov = 42.0
        self.cam.near = 0.1
        self.cam.far = 50.0
        self.cam.clear_color = 1
        self.stage.camera = self.cam
        self.yaw = 0.0
        self.pitch = 0.12
        self.dist = 2.75
        self.drag_last = None

        self.cur_D = np.zeros((52, 4), dtype=np.float32)
        self.cur_D[:, 3] = 1.0
        self.cur_hp = np.zeros(3, dtype=np.float32)
        self.has_pose = False
        self.smooth = 0.45
        self.last_seq = -1
        self.show_markers = False
        self.rec_active = False
        self.rec_D = []
        self.rec_hp = []
        self.gif_frames = []
        self.gif_tick = 0
        self.pending_export_png = False
        pyxel._rec_active = False
        pyxel._rec_count = 0
        pyxel._gif_count = 0
        pyxel._vrm_info_json = json.dumps(getattr(self.fig, "info", {}))
        self.plant_l = None
        self.plant_r = None
        self.prev_hips_fwd = None
        self.prev_hx = None
        self.prev_sx = None
        self.lr_swap_streak = 0
        self.outlier_streak = 0
        self.est_cam_pitch = 0.0
        self.foot_locked = [False, False]
        self.last_dbg_meta = {}
        self.prev_q_hips = None
        self.prev_q_chest = None
        self.prev_lm3d = {}
        self.vel_lm3d = {}
        self.prev_video_time = None
        self.prev_track_idx = None

        pyxel.run(self.update, self.draw)

    def _predict_filter_lm3d(self, lm3d, video_time):
        """Predict next landmark positions from prior velocity and gate sudden spikes."""
        dt = None if (video_time is None or self.prev_video_time is None) else float(video_time) - self.prev_video_time
        self.prev_video_time = None if video_time is None else float(video_time)
        if dt is None or dt <= 0.0 or dt > 0.25:
            self.prev_lm3d = {str(k): np.array(v, dtype=np.float64) for k, v in lm3d.items()}
            self.vel_lm3d = {str(k): np.zeros(3, dtype=np.float64) for k in lm3d}
            return lm3d, 0

        max_step = max(0.18, 5.5 * dt)
        filtered = {}
        gated = 0
        for k_raw, raw_v in lm3d.items():
            k = str(k_raw)
            obs = np.array(raw_v, dtype=np.float64)
            if k in self.prev_lm3d:
                prev = self.prev_lm3d[k]
                vel = self.vel_lm3d.get(k, np.zeros(3, dtype=np.float64))
                pred = prev + vel * (0.45 * dt)
                err = float(np.linalg.norm(obs - pred))
                if err > max_step:
                    w_obs = max(0.12, (max_step / err) * 0.45)
                    filt = pred + (obs - pred) * w_obs
                    self.vel_lm3d[k] = vel * 0.5
                    gated += 1
                else:
                    filt = pred + (obs - pred) * 0.72
                    inst_v = (filt - prev) / max(dt, 1e-3)
                    self.vel_lm3d[k] = 0.5 * vel + 0.5 * inst_v
                self.prev_lm3d[k] = filt
                filtered[k_raw] = filt.tolist()
            else:
                self.prev_lm3d[k] = obs
                self.vel_lm3d[k] = np.zeros(3, dtype=np.float64)
                filtered[k_raw] = obs.tolist()
        return filtered, gated

    def _consume_js_pose(self):
        raw = getattr(pyxel, "_pose", None)
        if not raw:
            return
        if raw == "reset":
            pyxel._pose = None
            self.has_pose = False
            self.plant_l = None
            self.plant_r = None
            self.prev_hips_fwd = None
            self.prev_hx = None
            self.prev_sx = None
            self.lr_swap_streak = 0
            self.outlier_streak = 0
            self.est_cam_pitch = 0.0
            self.prev_lm3d = {}
            self.vel_lm3d = {}
            self.prev_video_time = None
            self.prev_track_idx = None
            self.cur_hp = np.zeros(3, dtype=np.float32)
            self.fig.set_custom_pose(None)
            return
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                return
        elif isinstance(raw, dict):
            data = raw
        else:
            return

        seq = data.get("seq", 0)
        if seq == self.last_seq:
            return
        self.last_seq = seq

        if "smooth" in data:
            self.smooth = max(0.05, min(1.0, float(data["smooth"])))

        is_outlier = False
        if "lm3d" in data and data.get("axis", "world3d") == "world3d":
            trk = data.get("tracking") or {}
            trk_idx = trk.get("idx")
            if trk_idx is not None and self.prev_track_idx is not None and trk_idx != self.prev_track_idx:
                self.prev_lm3d = {}
                self.vel_lm3d = {}
            if trk_idx is not None:
                self.prev_track_idx = trk_idx
            lm_raw = data["lm3d"]
            # Reject physically impossible upside-down / collapsed-skeleton MediaPipe frames
            if all(k in lm_raw for k in ("11", "12", "23", "24", "25", "26", "27", "28")):
                p11 = np.array(lm_raw["11"], dtype=np.float64)
                p12 = np.array(lm_raw["12"], dtype=np.float64)
                p23 = np.array(lm_raw["23"], dtype=np.float64)
                p24 = np.array(lm_raw["24"], dtype=np.float64)
                p25 = np.array(lm_raw["25"], dtype=np.float64)
                p26 = np.array(lm_raw["26"], dtype=np.float64)
                p27 = np.array(lm_raw["27"], dtype=np.float64)
                p28 = np.array(lm_raw["28"], dtype=np.float64)
                hip_y_mp = 0.5 * (float(p23[1]) + float(p24[1]))
                ank_y_mp = 0.5 * (float(p27[1]) + float(p28[1]))
                hip_vec_mp = p23 - p24
                hip_span_mp = float(np.linalg.norm(hip_vec_mp))
                hip_vert_ratio = abs(float(hip_vec_mp[1])) / max(hip_span_mp, 1e-6)
                l_leg_mp = float(np.linalg.norm(p25 - p23) + np.linalg.norm(p27 - p25))
                r_leg_mp = float(np.linalg.norm(p26 - p24) + np.linalg.norm(p28 - p26))
                torso_mp = float(np.linalg.norm(0.5 * (p11 + p12) - 0.5 * (p23 + p24)))
                if (
                    (ank_y_mp - hip_y_mp) < 0.15
                    or min(l_leg_mp, r_leg_mp) < 0.48
                    or torso_mp < 0.26
                    or (hip_span_mp > 0.06 and hip_vert_ratio > 0.78)
                ) and self.has_pose:
                    self.last_dbg_meta = {
                        "derived": {
                            "corrupt_pose_skipped": True,
                            "ank_hip_dy": round(ank_y_mp - hip_y_mp, 4),
                            "min_leg_mp": round(min(l_leg_mp, r_leg_mp), 4),
                            "torso_mp": round(torso_mp, 4),
                            "hip_vert_ratio": round(hip_vert_ratio, 4),
                        }
                    }
                    self.prev_video_time = None if data.get("videoTime") is None else float(data.get("videoTime"))
                    for k in self.vel_lm3d:
                        self.vel_lm3d[k] = self.vel_lm3d[k] * 0.5
                    return
            filt_lm3d, lm_gated = self._predict_filter_lm3d(lm_raw, data.get("videoTime"))
            target_D, target_hp, self.last_dbg_meta = world_landmarks_to_D(
                filt_lm3d,
                self.prev_hips_fwd,
                self.est_cam_pitch,
                self.prev_hx,
                self.prev_sx,
                lm2d=data.get("lm2d"),
                leg_l1=self.fig.leg_l1,
                leg_l2=self.fig.leg_l2,
            )
            if lm_gated:
                self.last_dbg_meta.setdefault("derived", {})["lm_gated"] = lm_gated
            obs_p = self.last_dbg_meta.pop("_obs_cam_pitch", None)
            new_hx = self.last_dbg_meta.pop("_hx", None)
            new_sx = self.last_dbg_meta.pop("_sx", None)
            raw_jump_deg = self._qangle_deg(target_D[BONE_IDX["Hips"]], self.cur_D[BONE_IDX["Hips"]]) if self.has_pose else 0.0
            if self.has_pose and raw_jump_deg > 65.0 and self.outlier_streak < 4:
                is_outlier = True
                self.outlier_streak += 1
                self.last_dbg_meta.setdefault("derived", {})["outlier_damped_deg"] = raw_jump_deg
            else:
                self.outlier_streak = 0
                if self.last_dbg_meta.get("derived", {}).get("lr_swapped"):
                    self.lr_swap_streak += 1
                    if self.lr_swap_streak > 45:
                        self.prev_hx = None
                        self.prev_sx = None
                        self.lr_swap_streak = 0
                else:
                    self.lr_swap_streak = 0
                    if new_hx is not None:
                        self.prev_hx = new_hx if self.prev_hx is None else _norm(0.4 * new_hx + 0.6 * self.prev_hx)
                    if new_sx is not None:
                        self.prev_sx = new_sx if self.prev_sx is None else _norm(0.4 * new_sx + 0.6 * self.prev_sx)
                if obs_p is not None:
                    prev_p = self.est_cam_pitch
                    self.est_cam_pitch = obs_p if not self.has_pose else prev_p + (obs_p - prev_p) * 0.15
                    if abs(self.est_cam_pitch - prev_p) > 0.005:
                        self.plant_l = None
                        self.plant_r = None
                self.prev_hips_fwd = qrot(target_D[BONE_IDX["Hips"]].astype(np.float64), np.array([0.0, 0.0, 1.0]))
        else:
            target_D, target_hp = kalidokit_to_D(data)
            self.last_dbg_meta = {}
        if not self.has_pose:
            self.cur_D = target_D
            self.cur_hp = target_hp
            self.has_pose = True
        else:
            dots = np.sum(self.cur_D * target_D, axis=1, keepdims=True)
            tD = np.where(dots < 0, -target_D, target_D)
            if is_outlier:
                a_vec = np.full((52, 1), 0.03, dtype=np.float32)
                a_hp = 0.03
            else:
                angs = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
                t_spd = np.clip((angs - 0.03) / 0.35, 0.0, 1.0)
                a_min = 0.55 * self.smooth
                a_max = min(0.92, self.smooth + 0.35)
                a_vec = (a_min + (a_max - a_min) * t_spd).astype(np.float32)
                a_vec[BONE_IDX["Hips"], 0] = min(float(a_vec[BONE_IDX["Hips"], 0]), 0.42)
                a_hp = self.smooth
            blended = self.cur_D + (tD - self.cur_D) * a_vec
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            self.cur_D = blended / np.maximum(norms, 1e-6)
            self.cur_hp = self.cur_hp + (target_hp - self.cur_hp) * a_hp

        self.fig.set_custom_pose(self.cur_D, self.cur_hp)
        if "lm3d" in data and data.get("axis", "world3d") == "world3d":
            self._auto_ground()

    def _auto_ground(self):
        """Keep the lowest foot planted at rest_foot_y and anchor stance center (X, Z)."""
        self.fig.pose()
        self.fig.skin()
        if getattr(self.fig, "world_mats", None) is None:
            return
        lf = self.fig.bone_nodes[BONE_IDX["LeftFoot"]]
        rf = self.fig.bone_nodes[BONE_IDX["RightFoot"]]
        ys = [float(self.fig.world_pos(n)[1]) for n in (lf, rf) if n >= 0]
        if ys:
            min_y = min(ys)
            foot_y = float(getattr(self.fig, "rest_foot_y", 0.08))
            inv_h = 1.0 / max(self.fig.hip_h, 0.1)
            self.cur_hp[1] = max(-0.7, min(0.5, self.cur_hp[1] + (foot_y - min_y) * inv_h))
            if not any(self.foot_locked):
                self.cur_hp[0] *= 0.85
                self.cur_hp[2] *= 0.85
        self.fig.set_custom_pose(self.cur_D, self.cur_hp)
        self._lock_feet()

    def _lock_feet(self):
        """Pin each foot's (X, Z) on the floor only when stationary (< 0.045m jitter)."""
        self.fig.pose()
        self.fig.skin()
        if getattr(self.fig, "world_mats", None) is None:
            return
        q_hips = self.cur_D[BONE_IDX["Hips"]]
        fwd_hips = _norm(qrot(q_hips.astype(np.float64), np.array([0.0, 0.0, 1.0])))
        foot_y = float(getattr(self.fig, "rest_foot_y", 0.08))
        lock_y_max = foot_y + 0.06
        L1 = float(getattr(self.fig, "leg_l1", 0.353))
        L2 = float(getattr(self.fig, "leg_l2", 0.415))
        q_flat = _flat_foot_quat(fwd_hips).astype(np.float32)

        specs = [
            ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", 23, "plant_l"),
            ("RightUpperLeg", "RightLowerLeg", "RightFoot", 47, "plant_r"),
        ]
        locked_shifts = []
        locked_flags = [False, False]
        for idx_leg, (u_name, l_name, f_name, toe_idx, attr) in enumerate(specs):
            un = self.fig.bone_nodes[BONE_IDX[u_name]]
            ln = self.fig.bone_nodes[BONE_IDX[l_name]]
            fn = self.fig.bone_nodes[BONE_IDX[f_name]]
            if un < 0 or ln < 0 or fn < 0:
                continue
            hip_p = self.fig.world_pos(un)
            knee_p = self.fig.world_pos(ln)
            foot_p = self.fig.world_pos(fn)
            cur_xz = np.array([foot_p[0], foot_p[2]], dtype=np.float64)
            planted = getattr(self, attr)
            if planted is None:
                setattr(self, attr, cur_xz)
                continue
            if foot_p[1] <= lock_y_max and np.linalg.norm(cur_xz - planted) < 0.045:
                locked_flags[idx_leg] = True
                locked_shifts.append(cur_xz - planted)
                target_foot = np.array([planted[0], foot_y, planted[1]], dtype=np.float64)
                tdir, sdir = solve_leg_ik(target_foot - hip_p, knee_p - hip_p, fwd_hips, L1=L1, L2=L2)
                q_u = qmul(quat_from_two_vectors(qrot(q_hips.astype(np.float64), np.array([0.0, -1.0, 0.0])), tdir), q_hips).astype(np.float32)
                q_l = qmul(quat_from_two_vectors(qrot(q_u.astype(np.float64), np.array([0.0, -1.0, 0.0])), sdir), q_u).astype(np.float32)
                self.cur_D[BONE_IDX[u_name]] = q_u
                self.cur_D[BONE_IDX[l_name]] = q_l
                self.cur_D[BONE_IDX[f_name]] = q_flat
                self.cur_D[toe_idx] = q_flat
            else:
                setattr(self, attr, planted + (cur_xz - planted) * 0.35 if foot_p[1] <= lock_y_max else cur_xz)

        if len(locked_shifts) == 2:
            avg_shift = 0.5 * (locked_shifts[0] + locked_shifts[1])
            inv_h = 1.0 / max(self.fig.hip_h, 0.1)
            self.cur_hp[0] = max(-0.5, min(0.5, self.cur_hp[0] - float(avg_shift[0]) * inv_h * 0.5))
            self.cur_hp[2] = max(-0.5, min(0.5, self.cur_hp[2] - float(avg_shift[1]) * inv_h * 0.5))

        self.foot_locked = locked_flags
        self.fig.set_custom_pose(self.cur_D, self.cur_hp)

    def update(self):
        # Camera controls
        if pyxel.btn(pyxel.KEY_LEFT):
            self.yaw -= 0.05
        if pyxel.btn(pyxel.KEY_RIGHT):
            self.yaw += 0.05
        if pyxel.btn(pyxel.KEY_UP):
            self.pitch = min(0.8, self.pitch + 0.04)
        if pyxel.btn(pyxel.KEY_DOWN):
            self.pitch = max(-0.5, self.pitch - 0.04)
        if pyxel.btnp(pyxel.KEY_R):
            self.yaw, self.pitch, self.dist = 0.0, 0.12, 2.75
        if pyxel.btnp(pyxel.KEY_M):
            self.show_markers = not self.show_markers

        if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
            pos = (pyxel.mouse_x, pyxel.mouse_y)
            if self.drag_last is not None:
                dx = pos[0] - self.drag_last[0]
                dy = pos[1] - self.drag_last[1]
                self.yaw += dx * 0.02
                self.pitch = max(-0.6, min(0.9, self.pitch + dy * 0.02))
            self.drag_last = pos
        else:
            self.drag_last = None

        if pyxel.mouse_wheel:
            self.dist = max(1.4, min(5.5, self.dist - pyxel.mouse_wheel * 0.2))

        self._consume_vrm_upload()
        self._consume_rec_cmd()
        self._consume_js_pose()
        self.fig.animate(1.0 / 30.0)
        if self.rec_active:
            self.rec_D.append(self.cur_D.copy())
            self.rec_hp.append(self.cur_hp.copy())
            if len(self.rec_D) >= 900:
                self.rec_active = False
                pyxel._rec_active = False
            pyxel._rec_count = len(self.rec_D)
        self._update_debug_json()

    def _screen_to_pil(self, scale=2):
        from PIL import Image as PilImage

        arr = np.frombuffer(pyxel.screen.data_ptr(), dtype=np.uint8, count=160 * 120).reshape(120, 160)
        cols = list(pyxel.colors)
        pal_rgb = []
        for c in cols[:256]:
            pal_rgb.extend([(c >> 16) & 255, (c >> 8) & 255, c & 255])
        if len(pal_rgb) < 256 * 3:
            pal_rgb.extend([0] * (256 * 3 - len(pal_rgb)))
        im = PilImage.fromarray(arr.copy(), mode="P")
        im.putpalette(pal_rgb)
        if scale > 1:
            im = im.resize((160 * scale, 120 * scale), PilImage.NEAREST)
        return im

    def _consume_vrm_upload(self):
        path = getattr(pyxel, "_pending_vrm", None)
        if not path:
            return
        pyxel._pending_vrm = None
        pyxel._vrm_error = ""
        try:
            with open(path, "rb") as f:
                data = f.read()
            new_fig = Figure.from_vrm(data)
            self.stage.remove_child(self.fig)
            self.fig = new_fig
            self.stage.add_child(self.fig)
            self.foot_shadow.fig = self.fig
            self.plant_l = None
            self.plant_r = None
            self.foot_locked = [False, False]
            self.dist = max(2.2, float(self.fig.height) * 1.9)
            self.fig.set_custom_pose(self.cur_D, self.cur_hp)
            if self.has_pose:
                self._auto_ground()
            pyxel._vrm_info_json = json.dumps(self.fig.info)
        except Exception as e:
            pyxel._vrm_error = str(e)

    def _consume_rec_cmd(self):
        cmd = getattr(pyxel, "_rec_cmd", None)
        if not cmd:
            return
        pyxel._rec_cmd = None
        if cmd == "toggle_markers":
            self.show_markers = not self.show_markers
        elif cmd == "start":
            self.rec_D = []
            self.rec_hp = []
            self.gif_frames = []
            self.gif_tick = 0
            self.rec_active = True
            pyxel._rec_active = True
            pyxel._rec_count = 0
            pyxel._gif_count = 0
        elif cmd == "stop":
            self.rec_active = False
            pyxel._rec_active = False
        elif cmd == "export_vrma":
            import vrma_export
            pyxel._export_error = ""
            try:
                blob = vrma_export.build_vrma_bytes(self.rec_D, self.rec_hp, 30.0)
                with open("/tmp/export_motion.vrma", "wb") as f:
                    f.write(blob)
                pyxel._export_ready = "vrma"
            except Exception as e:
                pyxel._export_error = str(e)
        elif cmd == "export_png":
            self.pending_export_png = True
        elif cmd == "export_gif":
            pyxel._export_error = ""
            try:
                if len(self.gif_frames) < 2:
                    raise ValueError("Need at least 2 recorded frames for GIF")
                frames_rgba = [im.convert("RGBA") for im in self.gif_frames]
                frames_rgba[0].save(
                    "/tmp/export_anim.gif",
                    save_all=True,
                    append_images=frames_rgba[1:],
                    duration=100,
                    loop=0,
                    disposal=2,
                )
                pyxel._export_ready = "gif"
            except Exception as e:
                pyxel._export_error = str(e)

    def _qangle_deg(self, a, b):
        if a is None or b is None:
            return 0.0
        d = min(1.0, abs(float(np.dot(a, b))))
        return round(math.degrees(2.0 * math.acos(d)), 2)

    def _update_debug_json(self):
        W = getattr(self.fig, "world_mats", None)
        if W is None:
            return
        bones = {}
        for name, bidx in BONE_IDX.items():
            nidx = self.fig.bone_nodes[bidx]
            pos = [round(float(v), 4) for v in self.fig.world_pos(nidx)] if nidx >= 0 else None
            quat = [round(float(v), 4) for v in self.cur_D[bidx]]
            bones[name] = {"pos": pos, "D": quat}

        qh = self.cur_D[BONE_IDX["Hips"]].copy()
        qc = self.cur_D[BONE_IDX["Chest"]].copy()
        hips_jump = self._qangle_deg(self.prev_q_hips, qh)
        chest_jump = self._qangle_deg(self.prev_q_chest, qc)
        self.prev_q_hips, self.prev_q_chest = qh, qc

        anomalies = []
        if hips_jump > 25.0:
            anomalies.append(f"hips_jump:{hips_jump}deg")
        if chest_jump > 25.0:
            anomalies.append(f"chest_jump:{chest_jump}deg")
        if self.last_dbg_meta.get("derived", {}).get("hip_flip_guarded"):
            anomalies.append("hip_flip_guarded")
        lf_y = bones["LeftFoot"]["pos"][1] if bones["LeftFoot"]["pos"] else 0.0
        rf_y = bones["RightFoot"]["pos"][1] if bones["RightFoot"]["pos"] else 0.0
        if abs(lf_y - rf_y) > 0.15:
            anomalies.append(f"foot_height_diff:{round(abs(lf_y - rf_y), 3)}m")

        pyxel._debug_json = json.dumps({
            "pose_seq": self.last_seq,
            "has_pose": self.has_pose,
            "hp": [round(float(v), 4) for v in self.cur_hp],
            "state": {
                "foot_locked": self.foot_locked,
                "plant_l": [round(float(v), 4) for v in self.plant_l] if self.plant_l is not None else None,
                "plant_r": [round(float(v), 4) for v in self.plant_r] if self.plant_r is not None else None,
                "hips_jump_deg": hips_jump,
                "chest_jump_deg": chest_jump,
            },
            "derived": self.last_dbg_meta.get("derived"),
            "mp_lengths": self.last_dbg_meta.get("mp_lengths"),
            "anomalies": anomalies,
            "bones": bones,
        })

    def draw(self):
        pyxel.cls(1)
        ty = self.fig.height * 0.52
        cx = math.sin(self.yaw) * math.cos(self.pitch) * self.dist
        cy = ty + math.sin(self.pitch) * self.dist
        cz = math.cos(self.yaw) * math.cos(self.pitch) * self.dist
        self.cam.transform = Mat4.look_at(Vec3(cx, cy, cz), Vec3(0.0, ty, 0.0), Vec3.UP)

        self.stage.draw(0, 0, 160, 120)

        if self.show_markers:
            self._draw_markers(np.array([cx, cy, cz]), np.array([0.0, ty, 0.0]))

        if not self.has_pose:
            pyxel.text(4, 4, "READY - PLAY VIDEO", 7)

        if self.rec_active:
            if self.gif_tick % 3 == 0 and len(self.gif_frames) < 80:
                self.gif_frames.append(self._screen_to_pil(scale=2))
                pyxel._gif_count = len(self.gif_frames)
            self.gif_tick += 1
            if (self.gif_tick // 10) % 2 == 0:
                pyxel.circ(152, 7, 3, 8)
        if self.pending_export_png:
            self.pending_export_png = False
            try:
                self._screen_to_pil(scale=2).save("/tmp/export_shot.png")
                pyxel._export_ready = "png"
            except Exception as e:
                pyxel._export_error = str(e)

    def _project(self, p, eye, target):
        f = target - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        d = p - eye
        z = float(np.dot(d, f))
        if z <= 0.05:
            return None
        x = float(np.dot(d, r))
        y = float(np.dot(d, u))
        tan_half = math.tan(math.radians(self.cam.fov * 0.5))
        sx = (0.5 + 0.5 * (x / (z * tan_half * (160.0 / 120.0)))) * 160.0
        sy = (0.5 - 0.5 * (y / (z * tan_half))) * 120.0
        return int(round(sx)), int(round(sy))

    def _draw_markers(self, eye, target):
        W = getattr(self.fig, "world_mats", None)
        if W is None:
            return
        pts = {}
        for name, bidx in BONE_IDX.items():
            node_idx = self.fig.bone_nodes[bidx]
            if node_idx >= 0:
                p = self.fig.world_pos(node_idx)
                pts[name] = self._project(p, eye, target)

        for a, b, col in SKELETON_LINKS:
            pa, pb = pts.get(a), pts.get(b)
            if pa and pb:
                pyxel.line(pa[0], pa[1], pb[0], pb[1], col)

        for name, pt in pts.items():
            if not pt:
                continue
            col = 12 if name.startswith("Left") else (8 if name.startswith("Right") else 10)
            pyxel.circ(pt[0], pt[1], 1, col)

        for name, lbl, col in [("LeftHand", "LH", 12), ("RightHand", "RH", 8), ("LeftFoot", "LF", 12), ("RightFoot", "RF", 8), ("Head", "HD", 10)]:
            pt = pts.get(name)
            if pt:
                pyxel.text(pt[0] + 3, pt[1] - 2, lbl, col)

        # Forward (+Z) indicator from Hips
        hips_node = self.fig.bone_nodes[BONE_IDX["Hips"]]
        if hips_node >= 0:
            hp = self.fig.world_pos(hips_node)
            fwd = _norm(qrot(self.cur_D[BONE_IDX["Hips"]].astype(np.float64), np.array([0.0, 0.0, 1.0])))
            p0 = self._project(hp, eye, target)
            p1 = self._project(hp + fwd * 0.35, eye, target)
            if p0 and p1:
                pyxel.line(p0[0], p0[1], p1[0], p1[1], 11)
                pyxel.text(p1[0] + 2, p1[1] - 2, "FWD", 11)


App()

