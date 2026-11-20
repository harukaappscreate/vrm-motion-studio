"""vrma_export.py - export recorded pose frames (D, hp) as a VRM Animation (.vrma / glTF 2.0 GLB)."""
# SPDX-License-Identifier: MIT
import json
import struct

import numpy as np

REST_HIP_Y = 0.908

# Canonical 52-bone VRM 1.0 humanoid hierarchy for VRMC_vrm_animation (all rest rotations = identity [0,0,0,1])
VRMA_BONES = [
    ("hips", None, [0.0, REST_HIP_Y, 0.0]),
    ("spine", "hips", [0.0, 0.052, 0.012]),
    ("chest", "spine", [0.0, 0.113, 0.003]),
    ("upperChest", "chest", [0.0, 0.108, -0.015]),
    ("neck", "upperChest", [0.0, 0.132, -0.029]),
    ("head", "neck", [0.0, 0.073, 0.009]),
    ("leftShoulder", "upperChest", [0.022, 0.105, -0.029]),
    ("leftUpperArm", "leftShoulder", [0.087, -0.012, 0.0]),
    ("leftLowerArm", "leftUpperArm", [0.219, 0.0, 0.0]),
    ("leftHand", "leftLowerArm", [0.215, 0.0, 0.0]),
    ("leftThumbMetacarpal", "leftHand", [0.025, -0.012, 0.025]),
    ("leftThumbProximal", "leftThumbMetacarpal", [0.025, 0.0, 0.020]),
    ("leftThumbDistal", "leftThumbProximal", [0.022, 0.0, 0.018]),
    ("leftIndexProximal", "leftHand", [0.065, 0.0, 0.024]),
    ("leftIndexIntermediate", "leftIndexProximal", [0.030, 0.0, 0.0]),
    ("leftIndexDistal", "leftIndexIntermediate", [0.022, 0.0, 0.0]),
    ("leftMiddleProximal", "leftHand", [0.068, 0.0, 0.004]),
    ("leftMiddleIntermediate", "leftMiddleProximal", [0.032, 0.0, 0.0]),
    ("leftMiddleDistal", "leftMiddleIntermediate", [0.024, 0.0, 0.0]),
    ("leftRingProximal", "leftHand", [0.065, 0.0, -0.014]),
    ("leftRingIntermediate", "leftRingProximal", [0.030, 0.0, 0.0]),
    ("leftRingDistal", "leftRingIntermediate", [0.022, 0.0, 0.0]),
    ("leftLittleProximal", "leftHand", [0.058, 0.0, -0.030]),
    ("leftLittleIntermediate", "leftLittleProximal", [0.026, 0.0, 0.0]),
    ("leftLittleDistal", "leftLittleIntermediate", [0.019, 0.0, 0.0]),
    ("rightShoulder", "upperChest", [-0.022, 0.105, -0.029]),
    ("rightUpperArm", "rightShoulder", [-0.087, -0.012, 0.0]),
    ("rightLowerArm", "rightUpperArm", [-0.219, 0.0, 0.0]),
    ("rightHand", "rightLowerArm", [-0.215, 0.0, 0.0]),
    ("rightThumbMetacarpal", "rightHand", [-0.025, -0.012, 0.025]),
    ("rightThumbProximal", "rightThumbMetacarpal", [-0.025, 0.0, 0.020]),
    ("rightThumbDistal", "rightThumbProximal", [-0.022, 0.0, 0.018]),
    ("rightIndexProximal", "rightHand", [-0.065, 0.0, 0.024]),
    ("rightIndexIntermediate", "rightIndexProximal", [-0.030, 0.0, 0.0]),
    ("rightIndexDistal", "rightIndexIntermediate", [-0.022, 0.0, 0.0]),
    ("rightMiddleProximal", "rightHand", [-0.068, 0.0, 0.004]),
    ("rightMiddleIntermediate", "rightMiddleProximal", [-0.032, 0.0, 0.0]),
    ("rightMiddleDistal", "rightMiddleIntermediate", [-0.024, 0.0, 0.0]),
    ("rightRingProximal", "rightHand", [-0.065, 0.0, -0.014]),
    ("rightRingIntermediate", "rightRingProximal", [-0.030, 0.0, 0.0]),
    ("rightRingDistal", "rightRingIntermediate", [-0.022, 0.0, 0.0]),
    ("rightLittleProximal", "rightHand", [-0.058, 0.0, -0.030]),
    ("rightLittleIntermediate", "rightLittleProximal", [-0.026, 0.0, 0.0]),
    ("rightLittleDistal", "rightLittleIntermediate", [-0.019, 0.0, 0.0]),
    ("leftUpperLeg", "hips", [0.077, -0.040, 0.0]),
    ("leftLowerLeg", "leftUpperLeg", [0.0, -0.353, 0.0]),
    ("leftFoot", "leftLowerLeg", [0.0, -0.415, 0.0]),
    ("leftToes", "leftFoot", [0.0, -0.040, 0.120]),
    ("rightUpperLeg", "hips", [-0.077, -0.040, 0.0]),
    ("rightLowerLeg", "rightUpperLeg", [0.0, -0.353, 0.0]),
    ("rightFoot", "rightLowerLeg", [0.0, -0.415, 0.0]),
    ("rightToes", "rightFoot", [0.0, -0.040, 0.120]),
]

_BONE_NAMES_52 = (
    "chest", "head", "hips", "leftFoot", "leftHand", "leftIndexDistal",
    "leftIndexIntermediate", "leftIndexProximal", "leftLittleDistal",
    "leftLittleIntermediate", "leftLittleProximal", "leftLowerArm",
    "leftLowerLeg", "leftMiddleDistal", "leftMiddleIntermediate",
    "leftMiddleProximal", "leftRingDistal", "leftRingIntermediate",
    "leftRingProximal", "leftShoulder", "leftThumbDistal",
    "leftThumbMetacarpal", "leftThumbProximal", "leftToes", "leftUpperArm",
    "leftUpperLeg", "neck", "rightFoot", "rightHand", "rightIndexDistal",
    "rightIndexIntermediate", "rightIndexProximal", "rightLittleDistal",
    "rightLittleIntermediate", "rightLittleProximal", "rightLowerArm",
    "rightLowerLeg", "rightMiddleDistal", "rightMiddleIntermediate",
    "rightMiddleProximal", "rightRingDistal", "rightRingIntermediate",
    "rightRingProximal", "rightShoulder", "rightThumbDistal",
    "rightThumbMetacarpal", "rightThumbProximal", "rightToes",
    "rightUpperArm", "rightUpperLeg", "spine", "upperChest",
)

# Mapping from VRMA bone name -> index in 52-bone D array
VRMA_TO_BIDX = {name: idx for idx, name in enumerate(_BONE_NAMES_52)}


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def build_vrma_bytes(frames_D, frames_hp, fps=30.0, name="PyxelMotion"):
    """Build a valid .vrma (GLB + VRMC_vrm_animation) byte buffer from frame arrays."""
    F = len(frames_D)
    if F < 2:
        raise ValueError("Need at least 2 recorded frames to export .vrma")

    times = (np.arange(F, dtype=np.float32) / float(fps)).astype(np.float32)
    name_to_node = {bname: idx for idx, (bname, _, _) in enumerate(VRMA_BONES)}
    children_map = {idx: [] for idx in range(len(VRMA_BONES))}
    for idx, (_bname, parent, _t) in enumerate(VRMA_BONES):
        if parent is not None:
            children_map[name_to_node[parent]].append(idx)

    nodes = []
    human_bones = {}
    for idx, (bname, _parent, rest_t) in enumerate(VRMA_BONES):
        nd = {
            "name": bname,
            "translation": [float(v) for v in rest_t],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        if children_map[idx]:
            nd["children"] = children_map[idx]
        nodes.append(nd)
        human_bones[bname] = {"node": idx}

    # Build binary buffer: times + hips translation + per-bone local rotations
    bin_chunks = []
    buffer_views = []
    accessors = []
    offset = 0

    def add_accessor(arr, acc_type, comp_type=5126, with_minmax=False):
        nonlocal offset
        raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        pad = (-len(raw)) % 4
        if pad:
            raw += b"\x00" * pad
        bv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        bin_chunks.append(raw)
        offset += len(raw)

        acc = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": int(arr.shape[0]),
            "type": acc_type,
        }
        if with_minmax:
            acc["min"] = [float(np.min(arr))]
            acc["max"] = [float(np.max(arr))]
        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    time_acc = add_accessor(times.reshape(-1, 1), "SCALAR", with_minmax=True)

    samplers = []
    channels = []

    # Hips translation track (rest hips_y = REST_HIP_Y)
    hips_trans = np.zeros((F, 3), dtype=np.float32)
    for k in range(F):
        hp = frames_hp[k]
        hips_trans[k] = np.array([0.0, REST_HIP_Y, 0.0], dtype=np.float32) + np.asarray(hp, dtype=np.float32) * REST_HIP_Y
    hips_t_acc = add_accessor(hips_trans, "VEC3")
    samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": hips_t_acc})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": name_to_node["hips"], "path": "translation"}})

    # Local rotation tracks for each bone
    for idx, (bname, parent, _rest_t) in enumerate(VRMA_BONES):
        bidx = VRMA_TO_BIDX[bname]
        pidx = VRMA_TO_BIDX[parent] if parent is not None else None
        rots = np.zeros((F, 4), dtype=np.float32)
        for k in range(F):
            D = frames_D[k]
            w = D[bidx]
            if pidx is not None:
                pw = D[pidx]
                q = _qmul(_qconj(pw), w)
            else:
                q = w.copy()
            n = float(np.linalg.norm(q))
            if n > 1e-6:
                q = q / n
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            # Hemisphere continuity
            if k > 0 and float(np.dot(rots[k - 1], q)) < 0.0:
                q = -q
            rots[k] = q
        rot_acc = add_accessor(rots, "VEC4")
        samplers.append({"input": time_acc, "interpolation": "LINEAR", "output": rot_acc})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": idx, "path": "rotation"}})

    bin_data = b"".join(bin_chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "Pyxel3 Video-to-VRMA"},
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": human_bones},
            }
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "animations": [{"name": name, "channels": channels, "samplers": samplers}],
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    j_pad = (-len(json_bytes)) % 4
    if j_pad:
        json_bytes += b" " * j_pad

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    j_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    b_chunk = struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
    return header + j_chunk + b_chunk
