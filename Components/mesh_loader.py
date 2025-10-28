import os
import numpy as np
try:
    import trimesh
except ImportError:
    trimesh = None
from config import T_BINS

# ====================== MESH LOADING & UV PARAMS ====================
try:
    import trimesh
except ImportError:
    trimesh = None


def fit_model_axis(V):
    C = V.mean(axis=0)
    _, _, Vt = np.linalg.svd(V - C)
    zhat = Vt[0]
    xhat = Vt[1]
    yhat = np.cross(zhat, xhat)
    yhat /= (np.linalg.norm(yhat)+1e-9)
    xhat = np.cross(yhat, zhat)
    xhat /= (np.linalg.norm(xhat)+1e-9)
    zhat /= (np.linalg.norm(zhat)+1e-9)
    zcoord = (V - C) @ zhat
    zmin, zmax = float(zcoord.min()), float(zcoord.max())
    return C, xhat, yhat, zhat, zmin, zmax


def model_uv_from_vertices(V, C, xhat, yhat, zhat, zmin, zmax):
    rel = V - C
    x = rel @ xhat
    y = rel @ yhat
    z = rel @ zhat
    t = (z - zmin) / max(1e-6, (zmax - zmin))
    theta = np.arctan2(y, x)
    return t.astype(np.float32), theta.astype(np.float32)

# ---- Decimation backends: pyfqmr -> open3d -> pymeshlab -> voxel fallback ----


def _voxel_cluster_decimate(V, F, target_faces):
    if target_faces >= len(F) - 10:
        return V.astype(np.float32), F.astype(np.int32)
    bb_min = V.min(axis=0)
    bb_max = V.max(axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    ratio = target_faces / max(1, len(F))
    voxel = max(1e-6, diag * (1.0 - ratio) * 0.02)
    q = np.floor((V - bb_min) / voxel).astype(np.int32)
    key = q[:, 0].astype(np.int64) + (q[:, 1].astype(np.int64)
                                      << 21) + (q[:, 2].astype(np.int64) << 42)
    uniq, inv = np.unique(key, return_inverse=True)
    Vd = np.zeros((len(uniq), 3), np.float64)
    np.add.at(Vd, inv, V)
    counts = np.bincount(inv)
    Vd /= counts[:, None]
    Fd = inv[F]
    keep = (Fd[:, 0] != Fd[:, 1]) & (
        Fd[:, 1] != Fd[:, 2]) & (Fd[:, 2] != Fd[:, 0])
    Fd = Fd[keep]
    return Vd.astype(np.float32), Fd.astype(np.int32)


def decimate_mesh_np(V, F, ratio: float):
    ratio = float(max(0.01, min(1.0, ratio)))
    target_faces = int(max(1000, len(F) * ratio))
    try:
        import pyfqmr
        simp = pyfqmr.Simplify()
        simp.setMesh(V.astype(np.float32), F.astype(np.int32))
        simp.simplify_mesh(target_count=target_faces,
                           preserve_border=True, verbose=False)
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
    for i, bi in enumerate(tb):
        bins[bi].append(i)
    return [np.array(b, np.int32) for b in bins]


def load_mesh_fast(mesh_path, units="m", decim_ratio=1.0):
    cache_path = decim_cache_path(mesh_path, decim_ratio)
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        V = data["V"].astype(np.float32)
        F = data["F"].astype(np.int32)
        t = data["t"].astype(np.float32)
        theta = data["theta"].astype(np.float32)
        t_bins = [arr for arr in data["t_bins"]]
        backend = str(data["backend"]) if "backend" in data.files else "cache"
        print(
            f"[MESH] Loaded cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
        return V, F, t, theta, t_bins

    if trimesh is None:
        raise RuntimeError("trimesh not installed and no cache present.")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm":
        m.apply_scale(0.001)

    V0 = m.vertices.astype(np.float32)
    F0 = m.faces.astype(np.int32)
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
    print(
        f"[MESH] Built cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
    return V, F, t, theta, t_bins

# ---- Textured loader (OBJ+MTL) ----


def load_mesh_textured(mesh_path, units="m"):
    """
    Returns: V, F, UV, tex_bgr, kd_bgr
    """
    if trimesh is None:
        raise RuntimeError(
            "trimesh required for OBJ/MTL (pip install trimesh)")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm":
        m.apply_scale(0.001)

    V = m.vertices.astype(np.float32)
    F = m.faces.astype(np.int32)
    UV = None
    tex_bgr = None
    kd_bgr = (180, 180, 180)
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
                    kd_bgr = (int(np.clip(kd_rgb[2]*255, 0, 255)),
                              int(np.clip(kd_rgb[1]*255, 0, 255)),
                              int(np.clip(kd_rgb[0]*255, 0, 255)))
                tex = getattr(mat, "image", None)
                if tex is not None:
                    tex_rgb = np.array(tex.convert("RGB"))
                    tex_bgr = tex_rgb[:, :, ::-1].copy()
        except Exception:
            pass
    return V, F, UV, tex_bgr, kd_bgr


def decim_cache_path(mesh_path, ratio):
    base, _ = os.path.splitext(mesh_path)
    return f"{base}.decR{ratio:.2f}.npz"


def load_mesh_fast(mesh_path, units="m", decim_ratio=1.0):
    cache_path = decim_cache_path(mesh_path, decim_ratio)
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        V = data["V"].astype(np.float32)
        F = data["F"].astype(np.int32)
        t = data["t"].astype(np.float32)
        theta = data["theta"].astype(np.float32)
        t_bins = [arr for arr in data["t_bins"]]
        backend = str(data["backend"]) if "backend" in data.files else "cache"
        print(
            f"[MESH] Loaded cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
        return V, F, t, theta, t_bins

    if trimesh is None:
        raise RuntimeError("trimesh not installed and no cache present.")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm":
        m.apply_scale(0.001)

    V0 = m.vertices.astype(np.float32)
    F0 = m.faces.astype(np.int32)
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
    print(
        f"[MESH] Built cache {cache_path} ({len(V)} verts, {len(F)} tris) [decim={backend}]")
    return V, F, t, theta, t_bins


def load_mesh_textured(mesh_path, units="m"):
    """
    Returns: V, F, UV, tex_bgr, kd_bgr
    """
    if trimesh is None:
        raise RuntimeError(
            "trimesh required for OBJ/MTL (pip install trimesh)")
    m = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    if units.lower() == "mm":
        m.apply_scale(0.001)

    V = m.vertices.astype(np.float32)
    F = m.faces.astype(np.int32)
    UV = None
    tex_bgr = None
    kd_bgr = (180, 180, 180)
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
                    kd_bgr = (int(np.clip(kd_rgb[2]*255, 0, 255)),
                              int(np.clip(kd_rgb[1]*255, 0, 255)),
                              int(np.clip(kd_rgb[0]*255, 0, 255)))
                tex = getattr(mat, "image", None)
                if tex is not None:
                    tex_rgb = np.array(tex.convert("RGB"))
                    tex_bgr = tex_rgb[:, :, ::-1].copy()
        except Exception:
            pass
    return V, F, UV, tex_bgr, kd_bgr


def compute_slice_frames(cw, nw, rw, cf, nf, rf):
    ts = np.linspace(0, 1, T_BINS, dtype=np.float32)
    C = np.empty((T_BINS, 3), np.float32)
    U = np.empty_like(C)
    V = np.empty_like(C)
    R = np.empty((T_BINS,), np.float32)
    for k, t in enumerate(ts):
        c = cw*(1-t) + cf*t
        n = (nw*(1-t) + nf*t)
        n /= (np.linalg.norm(n)+1e-9)
        ref = np.array([1, 0, 0], float)
        if abs(float(np.dot(ref, n))) > 0.9:
            ref = np.array([0, 1, 0], float)
        u = ref - (ref @ n)*n
        u /= (np.linalg.norm(u)+1e-9)
        v = np.cross(n, u)
        v /= (np.linalg.norm(v)+1e-9)
        r = rw*(1-t) + rf*t
        C[k], U[k], V[k], R[k] = c, u, v, r
    return C, U, V, R


def wrap_vertices_to_arm_binned(V_out, theta, t_bins, C, U, V, R, t_max=1.0):
    max_bin = min(T_BINS-1, int(t_max * (T_BINS-1) + 1e-6))
    for bin_idx in range(max_bin+1):
        idx = t_bins[bin_idx]
        if idx.size == 0:
            continue
        c = C[bin_idx]
        u = U[bin_idx]
        v = V[bin_idx]
        r = R[bin_idx]
        th = theta[idx]
        ct = np.cos(th)
        st = np.sin(th)
        V_out[idx] = c + (r*ct)[:, None]*u + (r*st)[:, None]*v
    for bin_idx in range(max_bin+1, T_BINS):
        idx = t_bins[bin_idx]
        if idx.size == 0:
            continue
        c = C[max_bin]
        u = U[max_bin]
        v = V[max_bin]
        r = R[max_bin]
        th = theta[idx]
        ct = np.cos(th)
        st = np.sin(th)
        V_out[idx] = c + (r*ct)[:, None]*u + (r*st)[:, None]*v
