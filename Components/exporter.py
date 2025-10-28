import numpy as np


def save_obj_surface(path, V_world, F, units='mm'):
    scale = 1000.0 if units == 'mm' else 1.0
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# exported from arm_splint_wrap_buttons_tex_lock.py\n")
        for v in (V_world * scale):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in (F + 1):
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def _vertex_normals(V, F):
    vnorm = np.zeros_like(V, dtype=np.float64)
    tri = V[F]
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    for i, face in enumerate(F):
        vnorm[face] += n[i]
    nn = np.linalg.norm(vnorm, axis=1) + 1e-12
    vnorm = (vnorm.T / nn).T
    return vnorm.astype(np.float32)


def export_surface_obj(V_arm, F_model, filename="export_surface.obj", units='mm'):
    if V_arm is None or F_model is None:
        print("[EXPORT] Nothing to export (mesh not ready).")
        return
    save_obj_surface(filename, V_arm, F_model, units=units)
    print(f"[EXPORT] Surface OBJ written: {filename} ({units})")


def export_thickened_obj(V_arm, F_model, thickness_mm=3.0, filename="export_solid.obj", units='mm'):
    if V_arm is None or F_model is None:
        print("[EXPORT] Nothing to export (mesh not ready).")
        return
    try:
        import trimesh
        surf = trimesh.Trimesh(vertices=V_arm.copy(),
                               faces=F_model.copy(), process=False)
        N = surf.vertex_normals.astype(np.float32)
    except Exception:
        N = _vertex_normals(V_arm, F_model)
    t = float(thickness_mm) / 1000.0
    V_out = V_arm + t * N
    V_in = V_arm - t * N
    F_out = F_model.copy()
    F_in = F_model[:, ::-1] + len(V_out)
    E = np.vstack([F_model[:, [0, 1]], F_model[:, [1, 2]],
                  F_model[:, [2, 0]]]).astype(np.int32)
    E_sort = np.sort(E, axis=1)
    uniq, counts = np.unique(E_sort, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    side = []
    offset = len(V_out)
    for e in boundary:
        v0, v1 = int(e[0]), int(e[1])
        side.append([v0, v1, v1 + offset])
        side.append([v0, v1 + offset, v0 + offset])
    side = np.array(side, dtype=np.int32)
    V_all = np.vstack([V_out, V_in])
    F_all = np.vstack([F_out, F_in, side])
    save_obj_surface(filename, V_all, F_all, units=units)
    print(
        f"[EXPORT] Solid OBJ written: {filename} (thickness {thickness_mm} mm, {units})")
