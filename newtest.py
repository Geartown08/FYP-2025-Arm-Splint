# arm_splint_wrap_buttons_tex_lock.py
# OBJ materials (texture + Kd) with Lambert shading; LOCK SHAPE button; SAFE/FAST detect button

import os, sys, cv2, numpy as np
from collections import deque

# ===================== SPEED / CAMERA SETTINGS =====================
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, cv2.getNumberOfCPUs() - 1))
except Exception:
    pass

CAM_INDEX   = 1
FRAME_W, FRAME_H, FRAME_FPS = 960, 540, 30

# FAST detection params (SAFE ignores these)
DETECT_SCALE = 0.5
DETECT_EVERY = 3
ROI_MARGIN   = 1.40

# ========================= ARUCO / GEOMETRY =========================
CALIB_PATH  = "camcalib.npz"
DICT        = cv2.aruco.DICT_4X4_50

WRIST_IDS = set(range(0, 14))
FORE_IDS  = set(range(14, 33))

MARKER_SIZE = 0.010
GAP_SIZE    = 0.002
WIN = 5
EMA_POS  = 0.65
EMA_AXIS = 0.65
EMA_RAD  = 0.65
MAX_CENTER_STEP = 0.05
MAX_RADIUS_STEP = 0.005

# =========================== MESH INPUT ============================
MESH_PATH   = "splint.obj"  # OBJ with .mtl next to it if textured
MESH_UNITS  = "mm"          # "m" or "mm"

# ============================ UI STATE =============================
UI_H = 50
UI_ROWS = 2
BAR_H = UI_H * UI_ROWS
DRAW_GUIDES = False

state = {
    'alpha': 1.0,        # 0.5 or 1.0
    't_max': 1.0,        # 0.5, 0.75, 1.0
    'wrist_cm': 16.0,    # overrides wrist radius
    'fore_cm':  24.0,    # overrides forearm radius
    'decim_ratio': 1.0,  # 1.0, 0.75, 0.5
    'needs_mesh_reload': False,
    'cull': False,
    'use_calib': True,
    'draw_mesh': True,
    'detect_fast': False,   # SAFE/FAST via button
    'locked': False,        # LOCK SHAPE via button
    'quit': False
}

# =========================== HELPERS ===============================
def ema(prev, new, a): return new if prev is None else (a*new + (1-a)*prev)

def clamp_step(prev, new, max_step):
    if prev is None: return new
    d = new - prev; n = float(np.linalg.norm(d))
    if n <= max_step or n == 0.0: return new
    return prev + d * (max_step / n)

def project_points(P3, K, dist):
    img,_ = cv2.projectPoints(P3.astype(np.float32), np.zeros(3), np.zeros(3), K, dist)
    return img.reshape(-1,2).astype(int)

def project_points_with_z(P3, K, dist):
    pts2 = project_points(P3, K, dist)
    z = P3[:,2].astype(np.float32).copy()
    return pts2, z

def circle_radius_from_ids(min_id, max_id):
    n = max_id - min_id + 1
    C = n*MARKER_SIZE + (n-1)*GAP_SIZE
    return C / (2*np.pi)

def solve_tag_pose(corners, K, dist):
    half = MARKER_SIZE/2.0
    obj = np.float32([[-half, half, 0],[ half, half, 0],[ half,-half, 0],[-half,-half, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1,2).astype(np.float32),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return ok, rvec, tvec

def fit_plane(points, prev_n=None):
    P = np.asarray(points); C = P.mean(axis=0)
    _,_,Vt = np.linalg.svd(P - C)
    n = Vt[-1]
    if prev_n is not None and float(np.dot(n, prev_n)) < 0.0: n = -n
    ref = np.array([1,0,0], float)
    if abs(float(np.dot(ref, n))) > 0.9: ref = np.array([0,1,0], float)
    u = ref - (ref @ n)*n; u/= (np.linalg.norm(u)+1e-9)
    v = np.cross(n, u);     v/= (np.linalg.norm(v)+1e-9)
    return C, n/np.linalg.norm(n), u, v

def ring_segments(center, n, r, samples=64):
    ref = np.array([1,0,0], float)
    if abs(float(np.dot(ref, n))) > 0.9: ref = np.array([0,1,0], float)
    u = ref - (ref @ n)*n; u/= (np.linalg.norm(u)+1e-9)
    v = np.cross(n, u);     v/= (np.linalg.norm(v)+1e-9)
    th = np.linspace(0, 2*np.pi, samples, endpoint=False)
    pts = [center + r*np.cos(t)*u + r*np.sin(t)*v for t in th]
    return [(pts[i], pts[(i+1)%samples]) for i in range(samples)]

# ====================== MESH LOADING & UV PARAMS ====================
try:
    import trimesh
except ImportError:
    trimesh = None

def fit_model_axis(V):
    C = V.mean(axis=0); _,_,Vt = np.linalg.svd(V - C)
    zhat = Vt[0]; xhat = Vt[1]
    yhat = np.cross(zhat, xhat); yhat /= (np.linalg.norm(yhat)+1e-9)
    xhat = np.cross(yhat, zhat); xhat /= (np.linalg.norm(xhat)+1e-9)
    zhat /= (np.linalg.norm(zhat)+1e-9)
    zcoord = (V - C) @ zhat
    zmin, zmax = float(zcoord.min()), float(zcoord.max())
    return C, xhat, yhat, zhat, zmin, zmax

def model_uv_from_vertices(V, C, xhat, yhat, zhat, zmin, zmax):
    rel = V - C
    x = rel @ xhat; y = rel @ yhat; z = rel @ zhat
    t = (z - zmin) / max(1e-6, (zmax - zmin))
    theta = np.arctan2(y, x)
    return t.astype(np.float32), theta.astype(np.float32)

# ---- Decimation backends: pyfqmr -> open3d -> pymeshlab -> voxel fallback ----
def _voxel_cluster_decimate(V, F, target_faces):
    if target_faces >= len(F) - 10:
        return V.astype(np.float32), F.astype(np.int32)
    bb_min = V.min(axis=0); bb_max = V.max(axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    ratio = target_faces / max(1, len(F))
    voxel = max(1e-6, diag * (1.0 - ratio) * 0.02)
    q = np.floor((V - bb_min) / voxel).astype(np.int32)
    key = q[:,0].astype(np.int64) + (q[:,1].astype(np.int64) << 21) + (q[:,2].astype(np.int64) << 42)
    uniq, inv = np.unique(key, return_inverse=True)
    Vd = np.zeros((len(uniq), 3), np.float64); np.add.at(Vd, inv, V)
    counts = np.bincount(inv); Vd /= counts[:,None]
    Fd = inv[F]
    keep = (Fd[:,0] != Fd[:,1]) & (Fd[:,1] != Fd[:,2]) & (Fd[:,2] != Fd[:,0])
    Fd = Fd[keep]
    return Vd.astype(np.float32), Fd.astype(np.int32)

def decimate_mesh_np(V, F, ratio: float):
    ratio = float(max(0.01, min(1.0, ratio)))
    target_faces = int(max(1000, len(F) * ratio))
    try:
        import pyfqmr
        simp = pyfqmr.Simplify()
        simp.setMesh(V.astype(np.float32), F.astype(np.int32))
        simp.simplify_mesh(target_count=target_faces, preserve_border=True, verbose=False)
        Vd, Fd = simp.getMesh()
        return Vd.astype(np.float32), Fd.astype(np.int32), "pyfqmr"
    except Exception:
        pass
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V.astype(np.float64)),
            o3d.utility.Vector3iVector(F.astype(np.int32))
        )
        m = m.simplify_quadric_decimation(target_faces)
        Vd = np.asarray(m.vertices, dtype=np.float32)
        Fd = np.asarray(m.triangles, dtype=np.int32)
        if len(Fd) >= 3:
            return Vd, Fd, "open3d"
    except Exception:
        pass
    try:
        import pymeshlab as pml
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(V, F))
        ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                        targetfacenum=target_faces, preservenormal=True)
        m = ms.current_mesh()
        Vd = m.vertex_matrix().astype(np.float32)
        Fd = m.face_matrix().astype(np.int32)
        if len(Fd) >= 3:
            return Vd, Fd, "pymeshlab"
    except Exception:
        pass
    Vd, Fd = _voxel_cluster_decimate(V, F, target_faces)
    return Vd, Fd, "voxel"

def decim_cache_path(mesh_path, ratio):
    base, _ = os.path.splitext(mesh_path)
    return f"{base}.decR{ratio:.2f}.npz"

def build_t_bins(t, T_BINS=64):
    bins = [[] for _ in range(T_BINS)]
    tb = np.clip((t * T_BINS).astype(int), 0, T_BINS-1)
    for i, bi in enumerate(tb): bins[bi].append(i)
    return [np.array(b, np.int32) for b in bins]

def load_mesh_fast(mesh_path, units="m", decim_ratio=1.0):
    cache_path = decim_cache_path(mesh_path, decim_ratio)
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        V = data["V"].astype(np.float32); F = data["F"].astype(np.int32)
        t = data["t"].astype(np.float32); theta = data["theta"].astype(np.float32)
        t_bins = [arr for arr in data["t_bins"]]
        backend = str(data["backend"]) if "backend" in data.files else "cache"
        print(f"[MESH] Loaded cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
        return V, F, t, theta, t_bins

    if trimesh is None:
        raise RuntimeError("trimesh not installed and no cache present.")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm": m.apply_scale(0.001)

    V0 = m.vertices.astype(np.float32); F0 = m.faces.astype(np.int32)
    if decim_ratio < 0.999:
        V, F, backend = decimate_mesh_np(V0, F0, decim_ratio)
        print(f"[MESH] Decimated via {backend}: {len(F0)} -> {len(F)} faces")
    else:
        V, F, backend = V0, F0, "none"

    C0, xhat0, yhat0, zhat0, zmin0, zmax0 = fit_model_axis(V)
    t, theta = model_uv_from_vertices(V, C0, xhat0, yhat0, zhat0, zmin0, zmax0)
    t_bins = build_t_bins(t, T_BINS=64)

    np.savez_compressed(cache_path, V=V, F=F, t=t, theta=theta,
                        t_bins=np.array(t_bins, dtype=object),
                        backend=np.array(backend))
    print(f"[MESH] Built cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
    return V, F, t, theta, t_bins

# ---- Textured loader (OBJ+MTL) ----
def load_mesh_textured(mesh_path, units="m"):
    """
    Returns: V, F, UV, tex_bgr, kd_bgr
    """
    if trimesh is None:
        raise RuntimeError("trimesh required for OBJ/MTL (pip install trimesh)")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm":
        m.apply_scale(0.001)

    V = m.vertices.astype(np.float32)
    F = m.faces.astype(np.int32)
    UV = None; tex_bgr = None; kd_bgr = (180,180,180)
    vis = getattr(m, "visual", None)
    if vis is not None:
        uv = getattr(vis, "uv", None)
        if uv is not None and len(uv) == len(V):
            UV = uv.astype(np.float32).copy()
        try:
            mat = getattr(vis, "material", None)
            if mat is not None:
                kd = getattr(mat, "diffuse", None) or getattr(mat, "kd", None)
                if kd is not None:
                    kd_rgb = tuple(float(x) for x in kd[:3])  # 0..1
                    kd_bgr = (int(np.clip(kd_rgb[2]*255,0,255)),
                              int(np.clip(kd_rgb[1]*255,0,255)),
                              int(np.clip(kd_rgb[0]*255,0,255)))
                tex = getattr(mat, "image", None)
                if tex is not None:
                    tex_rgb = np.array(tex.convert("RGB"))
                    tex_bgr = tex_rgb[:, :, ::-1].copy()
        except Exception:
            pass
    return V, F, UV, tex_bgr, kd_bgr

# ==================== FAST WRAP (t-BINNED VECTOR) ===================
T_BINS = 64

def compute_slice_frames(cw, nw, rw, cf, nf, rf):
    ts = np.linspace(0, 1, T_BINS, dtype=np.float32)
    C  = np.empty((T_BINS,3), np.float32)
    U  = np.empty_like(C); V = np.empty_like(C); R = np.empty((T_BINS,), np.float32)
    for k, t in enumerate(ts):
        c = cw*(1-t) + cf*t
        n = (nw*(1-t) + nf*t); n /= (np.linalg.norm(n)+1e-9)
        ref = np.array([1,0,0], float)
        if abs(float(np.dot(ref,n))) > 0.9: ref = np.array([0,1,0], float)
        u = ref - (ref @ n)*n; u /= (np.linalg.norm(u)+1e-9)
        v = np.cross(n, u);   v /= (np.linalg.norm(v)+1e-9)
        r = rw*(1-t) + rf*t
        C[k], U[k], V[k], R[k] = c, u, v, r
    return C, U, V, R

def wrap_vertices_to_arm_binned(V_out, theta, t_bins, C, U, V, R, t_max=1.0):
    max_bin = min(T_BINS-1, int(t_max * (T_BINS-1) + 1e-6))
    for bin_idx in range(max_bin+1):
        idx = t_bins[bin_idx]
        if idx.size == 0: continue
        c = C[bin_idx]; u = U[bin_idx]; v = V[bin_idx]; r = R[bin_idx]
        th = theta[idx]; ct = np.cos(th); st = np.sin(th)
        V_out[idx] = c + (r*ct)[:,None]*u + (r*st)[:,None]*v
    for bin_idx in range(max_bin+1, T_BINS):
        idx = t_bins[bin_idx]
        if idx.size == 0: continue
        c = C[max_bin]; u = U[max_bin]; v = V[max_bin]; r = R[max_bin]
        th = theta[idx]; ct = np.cos(th); st = np.sin(th)
        V_out[idx] = c + (r*ct)[:,None]*u + (r*st)[:,None]*v

# ======================= RENDERERS (Lambert) ========================
def painter_fill_mesh_textured(frame, K, dist, V, F, UV, tex_bgr,
                               alpha=1.0, ambient=0.22, headlamp=True, cull=True):
    if UV is None or tex_bgr is None:
        return False
    Ht, Wt = tex_bgr.shape[:2]
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]; tris3 = V[F]; tri_uv = UV[F]
    area2 = ((tris2[:,1,0]-tris2[:,0,0])*(tris2[:,2,1]-tris2[:,0,1])
           - (tris2[:,1,1]-tris2[:,0,1])*(tris2[:,2,0]-tris2[:,0,0]))
    vis = (area2 > 0) if cull else np.ones(len(F), dtype=bool)
    if not np.any(vis): return False
    t2v, t3v, tuv, Fv = tris2[vis], tris3[vis], tri_uv[vis], F[vis]
    z_tri = z[Fv].mean(axis=1); order = np.argsort(z_tri)
    overlay = frame.copy()
    for fi in order:
        dst = t2v[fi].astype(np.float32)
        x0 = max(0, int(np.floor(dst[:,0].min())))
        y0 = max(0, int(np.floor(dst[:,1].min())))
        x1 = min(frame.shape[1], int(np.ceil(dst[:,0].max())))
        y1 = min(frame.shape[0], int(np.ceil(dst[:,1].max())))
        if x1 <= x0 or y1 <= y0: continue
        dst_local = (dst - np.array([x0, y0], np.float32)).astype(np.float32)
        uv = tuv[fi].astype(np.float32)
        src = np.stack([uv[:,0]*Wt, (1.0-uv[:,1])*Ht], axis=1).astype(np.float32)
        M = cv2.getAffineTransform(src[:3], dst_local[:3])
        rect_w = max(1, x1 - x0); rect_h = max(1, y1 - y0)
        patch = cv2.warpAffine(tex_bgr, M, (rect_w, rect_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((rect_h, rect_w), np.uint8)
        cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255, cv2.LINE_AA)
        a,b,c = t3v[fi][0], t3v[fi][1], t3v[fi][2]
        n = np.cross(b-a, c-a); ln = np.linalg.norm(n)
        if ln < 1e-12: continue
        n /= ln
        if headlamp:
            centroid = (a+b+c)/3.0; l = -centroid; l/= (np.linalg.norm(l)+1e-12)
        else:
            l = np.array([0,0,1], np.float32)
        lam = max(ambient, float(np.dot(n, l)))
        patch_f = np.clip(patch.astype(np.float32) * lam, 0, 255).astype(np.uint8)
        roi = overlay[y0:y1, x0:x1]
        patch_masked = cv2.bitwise_and(patch_f, patch_f, mask=mask)
        bg_masked    = cv2.bitwise_and(roi,      roi,      mask=cv2.bitwise_not(mask))
        roi[:] = patch_masked + bg_masked
    cv2.addWeighted(overlay, float(alpha), frame, 1.0-float(alpha), 0.0, dst=frame)
    return True

def painter_fill_mesh_kd(frame, K, dist, V, F, kd_bgr=(180,180,180),
                         alpha=1.0, ambient=0.22, headlamp=True, cull=True):
    # Solid color Lambert shading using material Kd (no texture)
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]; tris3 = V[F]
    area2 = ((tris2[:,1,0]-tris2[:,0,0])*(tris2[:,2,1]-tris2[:,0,1])
           - (tris2[:,1,1]-tris2[:,0,1])*(tris2[:,2,0]-tris2[:,0,0]))
    vis = (area2 > 0) if cull else np.ones(len(F), dtype=bool)
    if not np.any(vis): return False
    t2v, t3v, Fv = tris2[vis], tris3[vis], F[vis]
    z_tri = z[Fv].mean(axis=1); order = np.argsort(z_tri)
    overlay = frame.copy()
    kb,kg,kr = map(float, kd_bgr)
    for fi in order:
        tri2 = t2v[fi].astype(np.int32)
        a,b,c = t3v[fi][0], t3v[fi][1], t3v[fi][2]
        n = np.cross(b-a, c-a); ln = np.linalg.norm(n)
        if ln < 1e-12: continue
        n /= ln
        if headlamp:
            centroid = (a+b+c)/3.0; l = -centroid; l/= (np.linalg.norm(l)+1e-12)
        else:
            l = np.array([0,0,1], np.float32)
        lam = max(ambient, float(np.dot(n, l)))
        col = (int(np.clip(kb*lam,0,255)), int(np.clip(kg*lam,0,255)), int(np.clip(kr*lam,0,255)))
        cv2.fillConvexPoly(overlay, tri2, col, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0-float(alpha), 0.0, dst=frame)
    return True

# ============================ EXPORT ===============================
def save_obj_surface(path, V_world, F, units='mm'):
    scale = 1000.0 if units == 'mm' else 1.0
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# exported from arm_splint_wrap_buttons_tex_lock.py\n")
        for v in (V_world * scale): f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in (F + 1):         f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

def _vertex_normals(V, F):
    vnorm = np.zeros_like(V, dtype=np.float64)
    tri = V[F]; n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    for i, face in enumerate(F): vnorm[face] += n[i]
    nn = np.linalg.norm(vnorm, axis=1) + 1e-12
    vnorm = (vnorm.T / nn).T
    return vnorm.astype(np.float32)

def export_surface_obj(V_arm, F_model, filename="export_surface.obj", units='mm'):
    if V_arm is None or F_model is None:
        print("[EXPORT] Nothing to export (mesh not ready)."); return
    save_obj_surface(filename, V_arm, F_model, units=units)
    print(f"[EXPORT] Surface OBJ written: {filename} ({units})")

def export_thickened_obj(V_arm, F_model, thickness_mm=3.0, filename="export_solid.obj", units='mm'):
    if V_arm is None or F_model is None:
        print("[EXPORT] Nothing to export (mesh not ready)."); return
    try:
        import trimesh
        surf = trimesh.Trimesh(vertices=V_arm.copy(), faces=F_model.copy(), process=False)
        N = surf.vertex_normals.astype(np.float32)
    except Exception:
        N = _vertex_normals(V_arm, F_model)
    t = float(thickness_mm) / 1000.0
    V_out = V_arm + t * N; V_in = V_arm - t * N
    F_out = F_model.copy(); F_in = F_model[:, ::-1] + len(V_out)
    E = np.vstack([F_model[:,[0,1]], F_model[:,[1,2]], F_model[:,[2,0]]]).astype(np.int32)
    E_sort = np.sort(E, axis=1); uniq, counts = np.unique(E_sort, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    side = []; offset = len(V_out)
    for e in boundary:
        v0, v1 = int(e[0]), int(e[1])
        side.append([v0, v1, v1 + offset])
        side.append([v0, v1 + offset, v0 + offset])
    side = np.array(side, dtype=np.int32)
    V_all = np.vstack([V_out, V_in]); F_all = np.vstack([F_out, F_in, side])
    save_obj_surface(filename, V_all, F_all, units=units)
    print(f"[EXPORT] Solid OBJ written: {filename} (thickness {thickness_mm} mm, {units})")

# ============================== UI =================================
class Button:
    def __init__(self, x, y, w, h, label, cb, toggle_group=None, get_label=None):
        self.r = (x,y,w,h); self.label = label; self.cb = cb
        self.toggle_group = toggle_group; self.active = False
        self.get_label = get_label
    def contains(self, px, py):
        x,y,w,h = self.r; return (x <= px < x+w) and (y <= py < y+h)
    def draw(self, img):
        x,y,w,h = self.r
        txt = self.label if self.get_label is None else self.get_label()
        col = (60,60,60) if not self.active else (90,90,90)
        cv2.rectangle(img, (x,y), (x+w,y+h), (200,200,200), -1)
        cv2.rectangle(img, (x+2,y+2), (x+w-2,y+h-2), col, -1)
        cv2.rectangle(img, (x,y), (x+w,y+h), (40,40,40), 1)
        cv2.putText(img, txt, (x+6, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255,255,255), 1, cv2.LINE_AA)

def layout_buttons():
    btns = []; w, h, pad = 100, 34, 8

    # Row 1: opacity, length, decimation
    x, y = 8, 8
    def set_alpha(a):
        def _():
            state['alpha'] = a
            for b in btns:
                if b.toggle_group=='alpha': b.active = False
            btn_a50.active = (a==0.5); btn_a100.active=(a==1.0)
        return _
    btn_a50  = Button(x,y,w,h,'Mesh 50%', set_alpha(0.5), toggle_group='alpha'); x+=w+pad
    btn_a100 = Button(x,y,w,h,'Mesh 100%', set_alpha(1.0), toggle_group='alpha'); x+=w+pad

    def set_len(frac):
        def _():
            state['t_max'] = frac
            for b in btns:
                if b.toggle_group=='length': b.active = False
            btn_l50.active=(frac==0.5); btn_l75.active=(frac==0.75); btn_l100.active=(frac==1.0)
        return _
    btn_l50  = Button(x,y,w,h,'Length 50%', set_len(0.5),  toggle_group='length'); x+=w+pad
    btn_l75  = Button(x,y,w,h,'Length 75%', set_len(0.75), toggle_group='length'); x+=w+pad
    btn_l100 = Button(x,y,w,h,'Length 100%',set_len(1.0),  toggle_group='length'); x+=w+pad

    def set_decim(ratio):
        def _():
            state['decim_ratio'] = ratio
            for b in btns:
                if b.toggle_group=='decim': b.active = False
            btn_d100.active = (ratio==1.0)
            btn_d75.active  = (ratio==0.75)
            btn_d50.active  = (ratio==0.5)
            state['needs_mesh_reload'] = True
        return _
    btn_d100 = Button(x,y,w,h,'Decim 100', set_decim(1.0),  toggle_group='decim'); x+=w+pad
    btn_d75  = Button(x,y,w,h,'Decim 75',  set_decim(0.75), toggle_group='decim'); x+=w+pad
    btn_d50  = Button(x,y,w,h,'Decim 50',  set_decim(0.5),  toggle_group='decim')

    # Row 2: wrist/fore +/-/value, Detect, Lock, Quit
    x, y = 8, UI_H + 8
    def w_plus():  state['wrist_cm'] = round(state['wrist_cm']+0.5,1)
    def w_minus(): state['wrist_cm'] = round(max(8.0, state['wrist_cm']-0.5),1)
    def f_plus():  state['fore_cm']  = round(state['fore_cm'] +0.5,1)
    def f_minus(): state['fore_cm']  = round(max(10.0, state['fore_cm'] -0.5),1)

    b_wm = Button(x,y,w,h,'Wrist −', w_minus); x+=w+pad
    b_wp = Button(x,y,w,h,'Wrist +', w_plus);  x+=w+pad
    b_wv = Button(x,y,w,h,'Wrist cm', lambda: None,
                  get_label=lambda: f"Wrist {state['wrist_cm']:.1f} cm"); x+=w+pad

    b_fm = Button(x,y,w,h,'Fore −', f_minus); x+=w+pad
    b_fp = Button(x,y,w,h,'Fore +', f_plus);  x+=w+pad
    b_fv = Button(x,y,w,h,'Fore cm', lambda: None,
                  get_label=lambda: f"Fore {state['fore_cm']:.1f} cm"); x+=w+pad

    def toggle_detect(): state['detect_fast'] = not state['detect_fast']
    b_detect = Button(x,y,130,h,'Detect', toggle_detect,
                      get_label=lambda: f"Detect {'FAST' if state['detect_fast'] else 'SAFE'}"); x+=130+pad

    def toggle_lock():
        # toggles in the mouse callback; actual mesh freezing handled in main loop
        state['locked'] = not state['locked']
    b_lock = Button(x,y,120,h,'Lock', toggle_lock,
                    get_label=lambda: f"{'UNLOCK' if state['locked'] else 'LOCK'} SHAPE"); x+=120+pad

    b_quit = Button(x,y,w,h,'Quit', lambda: state.__setitem__('quit', True))

    btns = [btn_a50,btn_a100, btn_l50,btn_l75,btn_l100, btn_d100,btn_d75,btn_d50,
            b_wm,b_wp,b_wv,b_fm,b_fp,b_fv,b_detect,b_lock,b_quit]
    (btn_a50 if state['alpha']==0.5 else btn_a100).active = True
    {0.5:btn_l50, 0.75:btn_l75, 1.0:btn_l100}[state['t_max']].active = True
    {1.0:btn_d100, 0.75:btn_d75, 0.5:btn_d50}[state['decim_ratio']].active = True
    return btns

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and y < BAR_H:
        for b in param['buttons']:
            if b.contains(x,y): b.cb()

# ========================== CAMERA OPEN ============================
def quiet_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

def warmup(cap, n=8):
    for _ in range(n):
        ok, _ = cap.read()
        if not ok: break

def open_camera(cam_index=0, w=960, h=540, fps=30):
    quiet_opencv()
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        fourcc_list = [cv2.VideoWriter_fourcc(*"MJPG"),
                       cv2.VideoWriter_fourcc(*"YUY2"), 0]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]; fourcc_list = [0]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]; fourcc_list = [cv2.VideoWriter_fourcc(*"MJPG"), 0]
    last_err = "unknown"
    for be in backends:
        cap = cv2.VideoCapture(cam_index, be)
        if not cap.isOpened(): last_err = f"backend {be} failed"; continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        ok = False
        for fcc in fourcc_list:
            if fcc: cap.set(cv2.CAP_PROP_FOURCC, fcc)
            ok, _ = cap.read()
            if ok: break
        if ok:
            print(f"[CAM] Opened index {cam_index} via backend {be}, {w}x{h}@{fps}")
            warmup(cap, n=8); return cap
        cap.release(); last_err = f"backend {be} opened but no frames"
    raise RuntimeError(f"Camera open failed: {last_err}")

# ============================== MAIN ===============================
def main():
    # Calibration (toggle with 'k')
    cal = np.load(CALIB_PATH)
    K_base, dist_base = cal['K'].astype(np.float32), cal['dist'].astype(np.float32)

    cap = open_camera(cam_index=CAM_INDEX, w=FRAME_W, h=FRAME_H, fps=FRAME_FPS)

    # ArUco detector (compat)
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(aruco, "CORNER_REFINE_SUBPIX", 1)
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 31
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 6.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.33
    params.minCornerDistanceRate = 0.01

    try:
        _detector = aruco.ArucoDetector(dictionary, params)
        def detect_markers(img): return _detector.detectMarkers(img)
    except Exception:
        def detect_markers(img):
            corners, ids, rej = aruco.detectMarkers(img, dictionary=dictionary, parameters=params)
            return corners, ids, rej

    # Mesh load (textured if decim=1.0; else fast decimated, UV-less)
    V_model = F_model = t_model = th_model = None
    t_bins = []; UV_model = None; TEX = None; KD_BGR = (180,180,180)
    if os.path.exists(MESH_PATH):
        try:
            if state['decim_ratio'] >= 0.999:
                V_model, F_model, UV_model, TEX, KD_BGR = load_mesh_textured(MESH_PATH, units=MESH_UNITS)
                print(f"[MESH] Textured load: V={len(V_model)} F={len(F_model)} "
                      f"UV={'yes' if UV_model is not None else 'no'} TEX={'yes' if TEX is not None else 'no'}")
            else:
                V_model, F_model, t_model, th_model, t_bins = load_mesh_fast(
                    MESH_PATH, units=MESH_UNITS, decim_ratio=state['decim_ratio'])
                UV_model = None; TEX = None
                print("[WARN] Decimation used → texture disabled (UVs not preserved)")
            C0, xhat0, yhat0, zhat0, zmin0, zmax0 = fit_model_axis(V_model)
            t_model, th_model = model_uv_from_vertices(V_model, C0, xhat0, yhat0, zhat0, zmin0, zmax0)
            t_bins = build_t_bins(t_model, T_BINS=64)
        except Exception as e:
            print(f"[MESH] Failed: {e}")
    else:
        print(f"[MESH] Missing {MESH_PATH}")

    # Smoothing state
    wrist_ids_seen, fore_ids_seen = set(), set()
    cw_win, nw_win, rw_win = deque(maxlen=WIN), deque(maxlen=WIN), deque(maxlen=WIN)
    cf_win, nf_win, rf_win = deque(maxlen=WIN), deque(maxlen=WIN), deque(maxlen=WIN)
    sm_cw=sm_nw=sm_rw=None; sm_cf=sm_nf=sm_rf=None

    # UI
    buttons = layout_buttons()
    cv2.namedWindow("Arm Splint", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Arm Splint", on_mouse, param={'buttons': buttons})

    frame_idx = 0
    _last_ids, _last_corners = None, None
    _last_bbox_small = None
    last_wrapped = None          # last computed (unlocked)
    locked_mesh = None           # frozen mesh in camera coords when locked

    print("Buttons: Mesh 50/100, Length 50/75/100, Decim 100/75/50, Wrist/Fore +/-/value, Detect SAFE/FAST, LOCK/UNLOCK. "
          "Keys: R reset, C clear IDs, B cull, K calib, M mesh, E surf OBJ, X solid OBJ, ESC quit")

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Mesh reload after decim change
        if state.get('needs_mesh_reload', False) and os.path.exists(MESH_PATH):
            try:
                if state['decim_ratio'] >= 0.999:
                    V_model, F_model, UV_model, TEX, KD_BGR = load_mesh_textured(MESH_PATH, units=MESH_UNITS)
                    print(f"[MESH] Textured reload: V={len(V_model)} F={len(F_model)} "
                          f"UV={'yes' if UV_model is not None else 'no'} TEX={'yes' if TEX is not None else 'no'}")
                else:
                    V_model, F_model, t_model, th_model, t_bins = load_mesh_fast(
                        MESH_PATH, units=MESH_UNITS, decim_ratio=state['decim_ratio'])
                    UV_model = None; TEX = None
                    print("[WARN] Decimation used → texture disabled (UVs not preserved)")
                C0, xhat0, yhat0, zhat0, zmin0, zmax0 = fit_model_axis(V_model)
                t_model, th_model = model_uv_from_vertices(V_model, C0, xhat0, yhat0, zhat0, zmin0, zmax0)
                t_bins = build_t_bins(t_model, T_BINS)
                # clearing lock on reload keeps things consistent
                locked_mesh = None; state['locked'] = False
            except Exception as e:
                print(f"[MESH] Reload failed: {e}")
            state['needs_mesh_reload'] = False

        # UI bar
        bar = np.full((BAR_H, frame.shape[1], 3), 30, np.uint8)
        for b in buttons: b.draw(bar)

        # Intrinsics choice
        if state['use_calib']:
            K_use, dist_use = K_base, dist_base
        else:
            fx = fy = 800.0; cx, cy = FRAME_W/2.0, FRAME_H/2.0
            K_use = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
            dist_use = np.zeros((5,1), np.float32)

        # ---------- Detection (button controlled) ----------
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = None; ids = None

        if not state['detect_fast']:
            corners, ids, _ = detect_markers(gray)
            _last_corners, _last_ids = corners, ids
            _last_bbox_small = None
        else:
            if frame_idx % DETECT_EVERY == 1:
                gsmall = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                    interpolation=cv2.INTER_AREA)
                def detect_full(): return detect_markers(gsmall)
                def detect_roi(bbox):
                    x0,y0,x1,y1 = bbox; roi = gsmall[y0:y1, x0:x1]
                    c_s, i_s, _ = detect_markers(roi)
                    if i_s is None: return None, None, None
                    c_s = [c + np.array([[[x0,y0]]], dtype=c.dtype) for c in c_s]
                    return c_s, i_s, None
                if _last_bbox_small is not None:
                    x0,y0,x1,y1 = _last_bbox_small
                    cx2, cy2 = (x0+x1)//2, (y0+y1)//2
                    w, h = (x1-x0), (y1-y0)
                    w2 = int(w*ROI_MARGIN/2); h2 = int(h*ROI_MARGIN/2)
                    xs0 = max(0, cx2 - w2); ys0 = max(0, cy2 - h2)
                    xs1 = min(gsmall.shape[1], cx2 + w2); ys1 = min(gsmall.shape[0], cy2 + h2)
                    if xs1>xs0 and ys1>ys0:
                        corners_s, ids, _ = detect_roi((xs0,ys0,xs1,ys1))
                    else:
                        corners_s, ids, _ = detect_full()
                else:
                    corners_s, ids, _ = detect_full()
                if ids is not None:
                    corners = [c / DETECT_SCALE for c in corners_s]
                    _last_corners, _last_ids = corners, ids
                    pts = np.vstack([c.reshape(-1,2) for c in corners_s])
                    x0,y0 = np.maximum(pts.min(axis=0).astype(int) - 4, 0)
                    x1 = min(int(pts[:,0].max()) + 4, gsmall.shape[1])
                    y1 = min(int(pts[:,1].max()) + 4, gsmall.shape[0])
                    _last_bbox_small = (x0,y0,x1,y1)
                else:
                    corners, ids = _last_corners, _last_ids
            else:
                corners, ids = _last_corners, _last_ids

        # Optional debug overlay
        if ids is not None and corners is not None:
            try: cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception: pass
            cv2.putText(frame, f"Detected IDs: {list(ids.flatten())[:10]}...",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            cv2.putText(frame, "No markers detected", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Pose smoothing (only meaningful when not locked)
        wrist_pts, fore_pts = [], []
        if ids is not None and corners is not None and not state['locked']:
            for cid, cs in zip(ids.flatten(), corners):
                okp, rvec, tvec = solve_tag_pose(cs, K_use, dist_use)
                if not okp: continue
                pos = tvec.reshape(3)
                if int(cid) in WRIST_IDS:
                    wrist_pts.append(pos); wrist_ids_seen.add(int(cid))
                elif int(cid) in FORE_IDS:
                    fore_pts.append(pos); fore_ids_seen.add(int(cid))

        if not state['locked']:
            if len(wrist_pts) >= 3:
                cw_raw, nw_raw, _, _ = fit_plane(wrist_pts, prev_n=sm_nw)
                cw_win.append(cw_raw)
                if sm_nw is not None and float(np.dot(nw_raw, sm_nw)) < 0.0: nw_raw = -nw_raw
                nw_win.append(nw_raw)
                if wrist_ids_seen:
                    rw_raw = circle_radius_from_ids(min(wrist_ids_seen), max(wrist_ids_seen)); rw_win.append(rw_raw)
                cw_w = np.mean(cw_win, axis=0)
                n_sum = np.sum(np.array(nw_win), axis=0); nw_w = n_sum / (np.linalg.norm(n_sum)+1e-9)
                rw_w = np.mean(rw_win) if len(rw_win)>0 else None
                conf = min(1.0, len(wrist_pts)/8.0)
                sm_cw = ema(sm_cw, cw_w, EMA_POS*conf); sm_cw = clamp_step(sm_cw, cw_w, MAX_CENTER_STEP)
                sm_nw = ema(sm_nw, nw_w, EMA_AXIS*conf); sm_nw /= (np.linalg.norm(sm_nw)+1e-9)
                if rw_w is not None: sm_rw = ema(sm_rw, rw_w, EMA_RAD*conf)

            if len(fore_pts) >= 3:
                cf_raw, nf_raw, _, _ = fit_plane(fore_pts, prev_n=sm_nf)
                cf_win.append(cf_raw)
                if sm_nf is not None and float(np.dot(nf_raw, sm_nf)) < 0.0: nf_raw = -nf_raw
                nf_win.append(nf_raw)
                if fore_ids_seen:
                    rf_raw = circle_radius_from_ids(min(fore_ids_seen), max(fore_ids_seen)); rf_win.append(rf_raw)
                cf_w = np.mean(cf_win, axis=0)
                n_sum = np.sum(np.array(nf_win), axis=0); nf_w = n_sum / (np.linalg.norm(n_sum)+1e-9)
                rf_w = np.mean(rf_win) if len(rf_win)>0 else None
                conf = min(1.0, len(fore_pts)/8.0)
                sm_cf = ema(sm_cf, cf_w, EMA_POS*conf); sm_cf = clamp_step(sm_cf, cf_w, MAX_CENTER_STEP)
                sm_nf = ema(sm_nf, nf_w, EMA_AXIS*conf); sm_nf /= (np.linalg.norm(sm_nf)+1e-9)
                if rf_w is not None: sm_rf = ema(sm_rf, rf_w, EMA_RAD*conf)

        # Draw mesh (locked uses frozen copy; unlocked recomputes)
        V_to_draw = None
        if state['locked'] and locked_mesh is not None:
            V_to_draw = locked_mesh
        else:
            if all(x is not None for x in (sm_cw, sm_nw, sm_rw, sm_cf, sm_nf, sm_rf)) and V_model is not None:
                rw_use = (state['wrist_cm'] / 100.0) / (2*np.pi)
                rf_use = (state['fore_cm']  / 100.0) / (2*np.pi)
                C_s, U_s, V_s, R_s = compute_slice_frames(sm_cw, sm_nw, rw_use, sm_cf, sm_nf, rf_use)
                V_arm = np.empty_like(V_model, dtype=np.float32)
                wrap_vertices_to_arm_binned(V_arm, th_model, t_bins, C_s, U_s, V_s, R_s, t_max=state['t_max'])
                last_wrapped = V_arm.copy()
                V_to_draw = V_arm

        # If user just pressed LOCK and we have a wrapped mesh, freeze it
        if state['locked'] and locked_mesh is None and last_wrapped is not None:
            locked_mesh = last_wrapped.copy()
            print("[LOCK] Shape frozen. Further tracking changes won't affect mesh until UNLOCK.")

        if state['draw_mesh'] and V_to_draw is not None and not (np.isnan(V_to_draw).any() or np.isinf(V_to_draw).any()):
            # Prefer textured Lambert; else solid Kd Lambert
            drawn = False
            if UV_model is not None and TEX is not None and state['decim_ratio'] >= 0.999:
                drawn = painter_fill_mesh_textured(
                    frame, K_use, dist_use, V_to_draw, F_model, UV_model, TEX,
                    alpha=state['alpha'], ambient=0.22, headlamp=True, cull=state['cull'])
            if not drawn:
                painter_fill_mesh_kd(frame, K_use, dist_use, V_to_draw, F_model, kd_bgr=KD_BGR,
                                     alpha=state['alpha'], ambient=0.22, headlamp=True, cull=state['cull'])

        # Compose & HUD
        out = np.vstack([bar, frame])
        cv2.putText(out, f"Wrist IDs: {sorted(wrist_ids_seen)}", (12, BAR_H+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(out,  f"Fore IDs: {sorted(fore_ids_seen)}", (12, BAR_H+42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        hud_y = BAR_H + 64
        cv2.putText(out,
            f"Mesh:{'OK' if V_model is not None else 'MISS'}  "
            f"V={0 if V_model is None else len(V_model)}  "
            f"F={0 if F_model is None else len(F_model)}  "
            f"Decim={state.get('decim_ratio',1.0):.2f}  "
            f"Cull={'ON' if state.get('cull',False) else 'OFF'}  "
            f"Calib={'OK' if state.get('use_calib',True) else 'FALLBACK'}  "
            f"Detect={'FAST' if state.get('detect_fast',False) else 'SAFE'}  "
            f"Tex={'YES' if (UV_model is not None and TEX is not None and state['decim_ratio']>=0.999) else 'NO'}  "
            f"Lock={'ON' if state.get('locked',False) else 'OFF'}",
            (12, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        cv2.imshow("Arm Splint", out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or state['quit']: break
        if k == ord('r'):
            wrist_ids_seen.clear(); fore_ids_seen.clear()
            cw_win.clear(); nw_win.clear(); rw_win.clear()
            cf_win.clear(); nf_win.clear(); rf_win.clear()
            sm_cw=sm_nw=sm_rw=None; sm_cf=sm_nf=sm_rf=None
            locked_mesh = None; state['locked'] = False
            print("[RESET] cleared IDs + windows + smoothing; UNLOCKED.")
        if k == ord('c'):
            wrist_ids_seen.clear(); fore_ids_seen.clear()
            rw_win.clear(); rf_win.clear()
            print("[CLEAR] IDs cleared (pose smoothing kept)")
        if k == ord('e'):
            mesh_to_export = locked_mesh if (state['locked'] and locked_mesh is not None) else last_wrapped
            export_surface_obj(mesh_to_export, F_model, filename="export_surface.obj", units='mm')
        if k == ord('x'):
            mesh_to_export = locked_mesh if (state['locked'] and locked_mesh is not None) else last_wrapped
            export_thickened_obj(mesh_to_export, F_model, thickness_mm=3.0, filename="export_solid.obj", units='mm')
        if k == ord('b'): state['cull'] = not state['cull']; print(f"[RENDER] Cull {'ON' if state['cull'] else 'OFF'}")
        if k == ord('k'): state['use_calib'] = not state['use_calib']; print(f"[CALIB] {'calibration' if state['use_calib'] else 'fallback intrinsics'}")
        if k == ord('m'): state['draw_mesh'] = not state['draw_mesh']; print(f"[RENDER] Mesh {'ON' if state['draw_mesh'] else 'OFF'}")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
