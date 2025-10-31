import cv2
import numpy as np
from numba import njit, prange
from components.utils import project_points_with_z

# --------------------------
# Numba helpers
# --------------------------

@njit(parallel=True)
def compute_normals_and_lambert(V_tri, ambient=0.22, headlamp=True):
    n_tri = V_tri.shape[0]
    lambert = np.empty(n_tri, dtype=np.float32)
    normals = np.empty((n_tri, 3), dtype=np.float32)
    for i in prange(n_tri):
        a, b, c = V_tri[i]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            normals[i] = np.array([0,0,1], dtype=np.float32)
            lambert[i] = ambient
            continue
        n /= ln
        normals[i] = n
        if headlamp:
            centroid = (a + b + c) / 3.0
            l = -centroid
            l /= np.linalg.norm(l) + 1e-12
        else:
            l = np.array([0,0,1], dtype=np.float32)
        lambert[i] = max(ambient, np.dot(n,l))
    return normals, lambert

@njit
def fill_convex_poly_numba(frame, tri2, color):
    H, W, _ = frame.shape
    xmin = max(0, np.min(tri2[:,0]))
    xmax = min(W-1, np.max(tri2[:,0]))
    ymin = max(0, np.min(tri2[:,1]))
    ymax = min(H-1, np.max(tri2[:,1]))
    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            x0, y0 = tri2[0]
            x1, y1 = tri2[1]
            x2, y2 = tri2[2]
            den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
            if abs(den) < 1e-6:
                continue
            w0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / den
            w1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / den
            w2 = 1 - w0 - w1
            if w0 >=0 and w1 >=0 and w2 >=0:
                frame[y,x,0] = color[0]
                frame[y,x,1] = color[1]
                frame[y,x,2] = color[2]

# --------------------------
# Solid color (Kd) renderer
# --------------------------

def painter_fill_mesh_kd(frame, K, dist, V, F, kd_bgr=(180,180,180),
                              alpha=1.0, ambient=0.22, headlamp=True, cull=True):
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]
    tris3 = V[F]
    area2 = (tris2[:,1,0]-tris2[:,0,0])*(tris2[:,2,1]-tris2[:,0,1]) - (tris2[:,1,1]-tris2[:,0,1])*(tris2[:,2,0]-tris2[:,0,0])
    vis = (area2>0) if cull else np.ones(len(F),dtype=bool)
    if not np.any(vis):
        return False
    t2v, t3v, Fv = tris2[vis], tris3[vis], F[vis]
    _, lam = compute_normals_and_lambert(t3v, ambient=ambient, headlamp=headlamp)
    overlay = frame.copy()
    kb, kg, kr = map(float, kd_bgr)
    colors = np.stack([np.clip(kb*lam,0,255), np.clip(kg*lam,0,255), np.clip(kr*lam,0,255)],axis=1).astype(np.uint8)
    for fi in range(len(t2v)):
        tri2 = t2v[fi].astype(np.int32)
        fill_convex_poly_numba(overlay, tri2, colors[fi])
    cv2.addWeighted(overlay, float(alpha), frame, 1.0-float(alpha), 0.0, dst=frame)
    return True

# --------------------------
# Textured mesh renderer
# --------------------------

def painter_fill_mesh_textured(frame, K, dist, V, F, UV, tex_bgr,
                                    alpha=1.0, ambient=0.22, headlamp=True, cull=True):
    if UV is None or tex_bgr is None:
        return False
    Ht, Wt = tex_bgr.shape[:2]
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]
    tris3 = V[F]
    tri_uv = UV[F]
    area2 = (tris2[:,1,0]-tris2[:,0,0])*(tris2[:,2,1]-tris2[:,0,1]) - (tris2[:,1,1]-tris2[:,0,1])*(tris2[:,2,0]-tris2[:,0,0])
    vis = (area2>0) if cull else np.ones(len(F),dtype=bool)
    if not np.any(vis):
        return False
    t2v, t3v, tuv, Fv = tris2[vis], tris3[vis], tri_uv[vis], F[vis]
    z_tri = z[Fv].mean(axis=1)
    order = np.argsort(z_tri)
    overlay = frame.copy()
    normals, lam = compute_normals_and_lambert(t3v, ambient=ambient, headlamp=headlamp)

    for idx in order:
        dst = t2v[idx].astype(np.float32)
        x0 = max(0, int(np.floor(dst[:,0].min())))
        y0 = max(0, int(np.floor(dst[:,1].min())))
        x1 = min(frame.shape[1], int(np.ceil(dst[:,0].max())))
        y1 = min(frame.shape[0], int(np.ceil(dst[:,1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        dst_local = (dst - np.array([x0,y0], np.float32)).astype(np.float32)
        uv = tuv[idx].astype(np.float32)
        src = np.stack([uv[:,0]*Wt, (1.0-uv[:,1])*Ht],axis=1).astype(np.float32)
        M = cv2.getAffineTransform(src[:3], dst_local[:3])
        rect_w = max(1, x1-x0)
        rect_h = max(1, y1-y0)
        patch = cv2.warpAffine(tex_bgr, M, (rect_w, rect_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Lambert shading
        patch_f = np.clip(patch.astype(np.float32) * lam[idx], 0, 255).astype(np.uint8)
        # Mask & blend
        mask = np.zeros((rect_h, rect_w), np.uint8)
        cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255, cv2.LINE_AA)
        roi = overlay[y0:y1, x0:x1]
        patch_masked = cv2.bitwise_and(patch_f, patch_f, mask=mask)
        bg_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        roi[:] = patch_masked + bg_masked

    cv2.addWeighted(overlay, float(alpha), frame, 1.0-float(alpha), 0.0, dst=frame)
    return True
