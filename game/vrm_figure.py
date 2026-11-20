"""vrm_figure.py - load a figure exported by VRM RETRO VIEWER ("Pyxel 3D" zip) and play its motions in Pyxel 3.

    import pyxel
    from pyxel.cube import Camera, Node
    from vrm_figure import Figure

    fig = Figure("figure/")          # the folder with model.tmx, motions.tmx and tex_*.png
    fig.play("walk")
    ...
    fig.animate(1 / 30)                # every frame: advance the motion and skin the mesh (numpy)
    stage.add_child(fig)               # a pyxel.cube Node: move it with fig.transform

Needs Pyxel 3 and numpy.  No os / sys / file writing, so it also follows the Code & Magic import rules.
The data files are .tmx (base64 text), because Code & Magic accepts .py / .png / .tmx / .glb / .ogg / .pyxres.
The figure uses palette colours 16-255 (its own colours, 255 = magenta colour key); 0-15 stay yours.

Motions: Quaternius Universal Animation Library 1 & 2 and KayKit Character Animations (Kay Lousberg), all CC0,
converted to VRM Animation by the open-vrma project.
Code: MIT (haruka_apps).  Runs on Pyxel 3 by Takashi Kitao (MIT License) and NumPy (BSD-3-Clause).  The avatar data keeps the license of the original VRM: see README.txt.
"""
# SPDX-License-Identifier: MIT
import base64
import io
import json
import os
import tempfile

import numpy as np
import pyxel
from pyxel.cube import Mat4, Node, Primitive, Vec3

KEY = 255

BONE_NAMES_52 = [
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
]


def load_data(path):
    """Read model / motions data: a .tmx file (a 1x1 Tiled map whose "data" property holds the .npz as base64,
    so Code & Magic accepts it) or a plain .npz."""
    if path.endswith(".tmx"):
        with open(path, "r") as f:
            text = f.read()
        i = text.index('value="', text.index('<property name="data"')) + 7     # a Tiled map with the data in a property
        return np.load(io.BytesIO(base64.b64decode(text[i:text.index('"', i)])))
    return np.load(path)


def find_data(folder, name):
    """folder + name + ".tmx" if it is there, otherwise ".npz"."""
    for ext in (".tmx", ".npz"):
        try:
            open(folder + name + ext, "rb").close()
            return folder + name + ext
        except OSError:
            pass
    raise OSError("%s.tmx / %s.npz not found in '%s'" % (name, name, folder))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q, v):
    u = q[:3]
    return v + 2 * np.cross(u, np.cross(u, v) + q[3] * v)


def _qmul_batch(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


def _quat_from_two_vectors(u, v):
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    d = float(np.dot(u, v))
    if d > 0.99995:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if d < -0.99995:
        axis = np.cross(np.array([1.0, 0.0, 0.0]), u)
        if np.linalg.norm(axis) < 1e-4:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), u)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    c = np.cross(u, v)
    q = np.array([c[0], c[1], c[2], 1.0 + d], dtype=np.float64)
    return q / np.linalg.norm(q)


_LEFT_FINGERS = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22)
_RIGHT_FINGERS = (29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 46)


class Motions:
    """Per humanoid bone and frame (30 fps): the world rotation delta from the T-pose, plus the hips offset."""

    def __init__(self, path):
        d = load_data(path)
        self.bones = [str(b) for b in d["bones"]]
        self.meta = json.loads(bytes(d["meta"]).decode())
        self.by_id = {m["id"]: m for m in self.meta}
        self.D = {m["id"]: d["d_" + m["id"]].astype(np.float32) for m in self.meta}
        self.Hp = {m["id"]: d["h_" + m["id"]].astype(np.float32) for m in self.meta}

    def merge(self, path):
        """Add the motions of another file (my_motions.tmx: the ones you added to the viewer as FBX)."""
        d = load_data(path)
        if [str(b) for b in d["bones"]] != self.bones:
            raise ValueError("bones differ from motions.tmx")
        for m in json.loads(bytes(d["meta"]).decode()):
            if m["id"] not in self.by_id:
                self.meta.append(m)
                self.by_id[m["id"]] = m
                self.D[m["id"]] = d["d_" + m["id"]].astype(np.float32)
                self.Hp[m["id"]] = d["h_" + m["id"]].astype(np.float32)

    def frame(self, mid, t):
        D, Hp, m = self.D[mid], self.Hp[mid], self.by_id[mid]
        n = len(D)
        f = t * 30.0
        f = f % (n - 1) if m["loop"] else min(f, n - 1.001)
        i = int(f)
        u = f - i
        a, b = D[i], D[min(i + 1, n - 1)]
        b = np.where((np.sum(a * b, 1) < 0)[:, None], -b, b)
        q = a + (b - a) * u
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q, Hp[i] + (Hp[min(i + 1, n - 1)] - Hp[i]) * u


class Figure(Node):
    """The exported avatar: one textured Primitive per material, skinned on the CPU with numpy."""

    def __init__(self, folder="", motions=None):
        super().__init__()
        folder = folder if folder == "" or folder.endswith("/") else folder + "/"
        d = load_data(find_data(folder, "model"))
        info = json.loads(bytes(d["info"]).decode())
        self.info = info
        self.height = info["height"]
        self.face_yaw = info["face_yaw"]
        self.flip = info["flip"]
        self.bone_parent = d["parent"]
        self.order = d["order"]
        self.rest_t = d["rest_t"].astype(np.float64)
        self.rest_q = d["rest_q"].astype(np.float64)
        self.rest_s = d["rest_s"].astype(np.float64)
        self.rest_wq = d["rest_wq"].astype(np.float64)
        self.bone_nodes = [int(n) for n in d["bone_nodes"]]
        self.node_bone = {n: k for k, n in enumerate(self.bone_nodes) if n >= 0}
        self.hips = info["hips"]
        self.hip_h = info["hip_h"]
        # palette: 16..254 = the avatar's colours, 255 = magenta (colour key); 0..15 are left alone
        pal = [int(c) for c in d["palette"]]
        cols = list(pyxel.colors)
        cols += [0] * (256 - len(cols))
        cols[16:16 + len(pal)] = pal
        cols[KEY] = 0xFF00FF
        pyxel.colors[:] = cols[:256]
        self.textures = [pyxel.Image.from_image(folder + "tex_%d.png" % k) for k in range(info["textures"])]
        self.parts, self.prims, self.p1 = [], [], []
        for k, pinfo in enumerate(info["parts"]):
            p = dict(pinfo)
            for key in ("pos", "uv", "idx", "slot", "ww", "skin_nodes", "ibm"):
                p[key] = d["p%d_%s" % (k, key)]
            p["pos"] = p["pos"].astype(np.float32)
            p["ww"] = p["ww"].astype(np.float32)
            p["slot"] = p["slot"].astype(np.int64)
            p["ibm"] = p["ibm"].astype(np.float32)
            pr = Primitive(Primitive.MODE_TRIANGLES, p["pos"].reshape(-1).tolist(), p["idx"].astype(np.int64).tolist(), [],
                           p["uv"].astype(np.float32).reshape(-1).tolist(),
                           cull=Primitive.CULL_NONE if p["double"] else Primitive.CULL_BACK)
            pr.compute_normals()
            self.parts.append(p)
            self.prims.append(pr)
            self.p1.append(np.hstack([p["pos"], np.ones((len(p["pos"]), 1), np.float32)]))
        self.draw_order = sorted(range(len(self.parts)), key=lambda i: {"OPAQUE": 0, "MASK": 1}.get(self.parts[i]["alpha"], 2))
        try:
            self.motions = motions or Motions(find_data(folder, "motions"))
        except OSError:
            self.motions = None
        if self.motions is not None and motions is None:
            try:
                self.motions.merge(find_data(folder, "my_motions"))     # motions you added (FBX), if any
            except OSError:
                pass
        self.custom_D = None
        self.custom_hp = None
        self.motion = "idle" if (self.motions and "idle" in self.motions.by_id) else "tpose"
        self.t = 0.0
        self.q, self.tr = self.rest_q.copy(), self.rest_t.copy()
        self._init_rest_metrics()
        self.animate(0.0)

    @classmethod
    def from_vrm(cls, data):
        """Load any VRM 0.x or 1.0 binary directly in memory via vrm_io.VrmModel."""
        from PIL import Image as PilImage
        import vrm_io as V

        def _quat_of(M):
            sc = np.maximum(np.linalg.norm(M[:3, :3], axis=0), 1e-8)
            q = V.mat_to_quat(M[:3, :3] / sc)
            n = np.linalg.norm(q)
            return q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        model = V.VrmModel(data)
        hips = model.human.get("hips")
        if hips is None:
            raise ValueError("No humanoid 'hips' bone found in this VRM")
        fig = cls.__new__(cls)
        Node.__init__(fig)
        fig.info = {
            "title": model.title,
            "author": model.author,
            "version": model.version,
            "license": model.license,
            "warnings": model.warnings,
            "height": model.height,
        }
        fig.height = model.height
        fig.face_yaw = model.face_yaw
        fig.flip = (model.version == "VRM 0.x")
        fig.bone_parent = np.asarray(model.parent, dtype=np.int64)
        fig.order = np.asarray(model.order, dtype=np.int64)
        fig.rest_t = np.asarray(model.rest_t, dtype=np.float64)
        fig.rest_q = np.asarray(model.rest_q, dtype=np.float64)
        q_norms = np.maximum(np.linalg.norm(fig.rest_q, axis=1, keepdims=True), 1e-8)
        fig.rest_q = fig.rest_q / q_norms
        fig.rest_s = np.asarray(model.rest_s, dtype=np.float64)
        fig.rest_wq = np.array([_quat_of(M) for M in model.rest_world], dtype=np.float64)
        fig.bone_nodes = [int(model.human.get(b, -1)) for b in BONE_NAMES_52]
        fig.node_bone = {n: k for k, n in enumerate(fig.bone_nodes) if n >= 0}
        fig.hips = int(hips)
        fig.hip_h = abs(float(model.rest_world[hips][1, 3])) or 1.0

        pal = [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in model.palette]
        cols = list(pyxel.colors)
        cols += [0] * (256 - len(cols))
        cols[16:16 + len(pal)] = pal[:239]
        cols[KEY] = 0xFF00FF
        pyxel.colors[:] = cols[:256]

        # Build RGB PNGs matching vrm_retro_viewer png_from_index so Pyxel maps every index cleanly
        cols_rgb = np.zeros((256, 3), dtype=np.uint8)
        n_pal = min(len(model.palette), 239)
        if n_pal > 0:
            cols_rgb[16:16 + n_pal] = np.asarray(model.palette[:n_pal], dtype=np.uint8)
        cols_rgb[KEY] = (255, 0, 255)

        tmp_dir = tempfile.gettempdir()
        fig.textures = []
        for k, idx_arr in enumerate(model.tex_index):
            im = PilImage.fromarray(cols_rgb[idx_arr.astype(np.uint8)], mode="RGB")
            path = os.path.join(tmp_dir, "vrm_itch_t%d.png" % k)
            im.save(path)
            fig.textures.append(pyxel.Image.from_image(path))

        fig.parts, fig.prims, fig.p1 = [], [], []
        for pinfo in model.parts:
            p = dict(pinfo)
            p["pos"] = np.asarray(p["pos"], dtype=np.float32)
            p["ww"] = np.asarray(p["ww"], dtype=np.float32)
            p["slot"] = np.asarray(p["slot"], dtype=np.int64)
            p["ibm"] = np.asarray(p["ibm"], dtype=np.float32)
            pr = Primitive(
                Primitive.MODE_TRIANGLES,
                p["pos"].reshape(-1).tolist(),
                np.asarray(p["idx"], dtype=np.int64).tolist(),
                [],
                np.asarray(p["uv"], dtype=np.float32).reshape(-1).tolist(),
                cull=Primitive.CULL_NONE if p["double"] else Primitive.CULL_BACK,
            )
            pr.compute_normals()
            fig.parts.append(p)
            fig.prims.append(pr)
            fig.p1.append(np.hstack([p["pos"], np.ones((len(p["pos"]), 1), np.float32)]))
        fig.draw_order = sorted(range(len(fig.parts)), key=lambda i: {"OPAQUE": 0, "MASK": 1}.get(fig.parts[i]["alpha"], 2))
        fig.motions = None
        fig.custom_D = None
        fig.custom_hp = None
        fig.motion = "tpose"
        fig.t = 0.0
        fig.q, fig.tr = fig.rest_q.copy(), fig.rest_t.copy()
        fig._init_rest_metrics()
        fig.animate(0.0)
        return fig

    def _init_rest_metrics(self):
        """Measure rest leg lengths, foot height, local hips scale, and canonical T-pose alignment."""
        W0 = self._compute_world_mats(self.rest_q, self.rest_t)

        def _pos_vrm1(n):
            p = W0[n, :3, 3].astype(np.float64).copy()
            if self.flip:
                p[0] = -p[0]
                p[2] = -p[2]
            return p

        if self.hips >= 0:
            p_hips = self.bone_parent[self.hips]
            if p_hips >= 0:
                pscale = float(np.mean(np.linalg.norm(W0[p_hips, :3, :3], axis=0)))
            else:
                pscale = 1.0
            self.local_hip_h = (self.hip_h / pscale) if pscale > 1e-6 else self.hip_h
        else:
            self.local_hip_h = 1.0

        l1_vals, l2_vals = [], []
        for u_b, l_b, f_b in ((25, 12, 3), (49, 36, 27)):
            un, ln, fn = self.bone_nodes[u_b], self.bone_nodes[l_b], self.bone_nodes[f_b]
            if un >= 0 and ln >= 0:
                d1 = float(np.linalg.norm(_pos_vrm1(ln) - _pos_vrm1(un)))
                if d1 > 1e-3:
                    l1_vals.append(d1)
            if ln >= 0 and fn >= 0:
                d2 = float(np.linalg.norm(_pos_vrm1(fn) - _pos_vrm1(ln)))
                if d2 > 1e-3:
                    l2_vals.append(d2)
        self.leg_l1 = float(np.mean(l1_vals)) if l1_vals else 0.353
        self.leg_l2 = float(np.mean(l2_vals)) if l2_vals else 0.415

        lf, rf = self.bone_nodes[3], self.bone_nodes[27]
        foot_ys = [float(W0[n, 1, 3]) for n in (lf, rf) if n >= 0]
        self.rest_foot_y = min(foot_ys) if foot_ys else 0.06

        tpose_fix = np.zeros((52, 4), dtype=np.float64)
        tpose_fix[:, 3] = 1.0
        limb_pairs = (
            (24, 11, np.array([1.0, 0.0, 0.0])),
            (11, 4, np.array([1.0, 0.0, 0.0])),
            (48, 35, np.array([-1.0, 0.0, 0.0])),
            (35, 28, np.array([-1.0, 0.0, 0.0])),
            (25, 12, np.array([0.0, -1.0, 0.0])),
            (12, 3, np.array([0.0, -1.0, 0.0])),
            (49, 36, np.array([0.0, -1.0, 0.0])),
            (36, 27, np.array([0.0, -1.0, 0.0])),
        )
        for pb, cb, canon_dir in limb_pairs:
            pn, cn = self.bone_nodes[pb], self.bone_nodes[cb]
            if pn >= 0 and cn >= 0:
                vec = _pos_vrm1(cn) - _pos_vrm1(pn)
                if np.linalg.norm(vec) > 1e-4:
                    tpose_fix[pb] = _quat_from_two_vectors(vec, canon_dir)

        tpose_fix[4] = tpose_fix[11]
        for fidx in _LEFT_FINGERS:
            tpose_fix[fidx] = tpose_fix[11]
        tpose_fix[28] = tpose_fix[35]
        for fidx in _RIGHT_FINGERS:
            tpose_fix[fidx] = tpose_fix[35]
        self.tpose_fix = tpose_fix

    def world_pos(self, node_idx):
        """World-space position of node_idx in viewer (+Z facing) coordinates."""
        p = self.world_mats[node_idx, :3, 3].astype(np.float64).copy()
        if self.flip:
            p[0] = -p[0]
            p[2] = -p[2]
        return p

    def world_axis(self, node_idx, col=2):
        """Unit world-space basis axis (default +Z col=2) of node_idx in viewer (+Z facing) coordinates."""
        a = self.world_mats[node_idx, :3, col].astype(np.float64).copy()
        if self.flip:
            a[0] = -a[0]
            a[2] = -a[2]
        n = float(np.linalg.norm(a))
        return a / n if n > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ------------------------------------------------------------------ motions
    def play(self, mid, restart=True):
        """mid: an id from motion_ids() or "tpose"."""
        if mid != self.motion or restart:
            if mid == "tpose" or (self.motions is not None and mid in self.motions.by_id):
                self.motion = mid
                self.t = 0.0

    def motion_ids(self):
        return [m["id"] for m in self.motions.meta] if self.motions is not None else []

    def finished(self):
        """True when a one-shot motion (jump, punch, death ...) has reached its last frame."""
        if self.motions is None:
            return False
        m = self.motions.by_id.get(self.motion)
        return m is not None and not m["loop"] and self.t >= m["seconds"]

    def set_custom_pose(self, D, hp=None):
        """Set per-bone world rotation delta D (52, 4) and optional hips offset hp (3,)."""
        self.custom_D = D.astype(np.float32) if D is not None else None
        self.custom_hp = (np.zeros(3, np.float32) if hp is None else np.asarray(hp, np.float32))

    def animate(self, dt):
        self.t += dt
        self.pose()
        self.skin()

    def pose(self):
        q = self.rest_q.copy()
        tr = self.rest_t.copy()
        has_custom = self.custom_D is not None
        has_motion = (self.motion != "tpose" and self.motions is not None)
        if (has_custom or has_motion) and self.hips >= 0:
            if has_custom:
                D = _qmul_batch(self.custom_D.astype(np.float64), self.tpose_fix).astype(np.float32)
                hp = self.custom_hp
            else:
                D, hp = self.motions.frame(self.motion, self.t)
            if self.flip:                                     # VRM 0.x faces -Z
                D = D * np.array([-1.0, 1.0, -1.0, 1.0], np.float32)
                hp = hp * np.array([-1.0, 1.0, -1.0], np.float32)
            one = np.array([0.0, 0.0, 0.0, 1.0])
            Wq = {}
            for i in self.order:
                p = self.bone_parent[i]
                pw = Wq.get(p) if p >= 0 else None
                if pw is None:
                    pw = self.rest_wq[p] if p >= 0 else one
                k = self.node_bone.get(int(i))
                if k is not None:
                    w = qmul(D[k], self.rest_wq[i])
                    q[i] = qmul(qconj(pw), w)
                    Wq[i] = w
                else:
                    Wq[i] = qmul(pw, q[i])
            p = self.bone_parent[self.hips]
            pw = Wq.get(p, one) if p >= 0 else one
            tr[self.hips] = self.rest_t[self.hips] + qrot(qconj(pw), hp * self.local_hip_h)
        self.q, self.tr = q, tr

    # ------------------------------------------------------------------ skinning
    def _compute_world_mats(self, q, tr):
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        M = np.zeros((len(q), 4, 4), np.float64)
        M[:, 0, 0] = 1 - 2 * (y * y + z * z)
        M[:, 0, 1] = 2 * (x * y - z * w)
        M[:, 0, 2] = 2 * (x * z + y * w)
        M[:, 1, 0] = 2 * (x * y + z * w)
        M[:, 1, 1] = 1 - 2 * (x * x + z * z)
        M[:, 1, 2] = 2 * (y * z - x * w)
        M[:, 2, 0] = 2 * (x * z - y * w)
        M[:, 2, 1] = 2 * (y * z + x * w)
        M[:, 2, 2] = 1 - 2 * (x * x + y * y)
        M[:, :3, :3] *= self.rest_s[:, None, :]
        M[:, :3, 3] = tr
        M[:, 3, 3] = 1.0
        W = np.empty_like(M)
        for i in self.order:
            p = self.bone_parent[i]
            W[i] = M[i] if p < 0 else W[p] @ M[i]
        return W

    def skin(self):
        W = self._compute_world_mats(self.q, self.tr).astype(np.float32)
        self.world_mats = W
        for p, pr, p1 in zip(self.parts, self.prims, self.p1):
            mats = (W[p["skin_nodes"]] @ p["ibm"])[:, :3, :]
            pos = np.zeros((len(p1), 3), np.float32)
            for k in range(4):
                wk = p["ww"][:, k]
                if not wk.any():
                    continue
                pos += wk[:, None] * np.einsum("nij,nj->ni", mats[p["slot"][:, k]], p1)
            pr.positions[:] = pos.reshape(-1).tolist()
            pr.compute_normals()

    def on_draw(self):
        m = Mat4.from_euler(Vec3(0.0, float(self.face_yaw), 0.0))   # every figure faces +Z (toward the camera at +Z)
        for i in self.draw_order:
            p = self.parts[i]
            if p["keyed"]:
                self.prim(m, self.prims[i], self.textures[p["tex"]], colkey=KEY)
            else:
                self.prim(m, self.prims[i], self.textures[p["tex"]])
