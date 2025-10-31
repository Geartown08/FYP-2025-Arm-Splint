import cv2
import numpy as np
from components.utils import project_points_with_z


def painter_fill_mesh_textured(
    frame, K, dist, V, F, UV, tex_bgr, alpha=1.0, ambient=0.22, headlamp=True, cull=True
):
    if UV is None or tex_bgr is None:
        return False
    Ht, Wt = tex_bgr.shape[:2]
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]
    tris3 = V[F]
    tri_uv = UV[F]
    area2 = (tris2[:, 1, 0] - tris2[:, 0, 0]) * (tris2[:, 2, 1] - tris2[:, 0, 1]) - (
        tris2[:, 1, 1] - tris2[:, 0, 1]
    ) * (tris2[:, 2, 0] - tris2[:, 0, 0])
    vis = (area2 > 0) if cull else np.ones(len(F), dtype=bool)
    if not np.any(vis):
        return False
    t2v, t3v, tuv, Fv = tris2[vis], tris3[vis], tri_uv[vis], F[vis]
    z_tri = z[Fv].mean(axis=1)
    order = np.argsort(z_tri)
    overlay = frame.copy()
    for fi in order:
        dst = t2v[fi].astype(np.float32)
        x0 = max(0, int(np.floor(dst[:, 0].min())))
        y0 = max(0, int(np.floor(dst[:, 1].min())))
        x1 = min(frame.shape[1], int(np.ceil(dst[:, 0].max())))
        y1 = min(frame.shape[0], int(np.ceil(dst[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        dst_local = (dst - np.array([x0, y0], np.float32)).astype(np.float32)
        uv = tuv[fi].astype(np.float32)
        src = np.stack([uv[:, 0] * Wt, (1.0 - uv[:, 1]) * Ht], axis=1).astype(
            np.float32
        )
        M = cv2.getAffineTransform(src[:3], dst_local[:3])
        rect_w = max(1, x1 - x0)
        rect_h = max(1, y1 - y0)
        patch = cv2.warpAffine(
            tex_bgr,
            M,
            (rect_w, rect_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = np.zeros((rect_h, rect_w), np.uint8)
        cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255, cv2.LINE_AA)
        a, b, c = t3v[fi][0], t3v[fi][1], t3v[fi][2]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        if headlamp:
            centroid = (a + b + c) / 3.0
            l = -centroid
            l /= np.linalg.norm(l) + 1e-12
        else:
            l = np.array([0, 0, 1], np.float32)
        lam = max(ambient, float(np.dot(n, l)))
        patch_f = np.clip(patch.astype(np.float32) * lam, 0, 255).astype(np.uint8)
        roi = overlay[y0:y1, x0:x1]
        patch_masked = cv2.bitwise_and(patch_f, patch_f, mask=mask)
        bg_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        roi[:] = patch_masked + bg_masked
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, dst=frame)
    return True


def painter_fill_mesh_kd(
    frame,
    K,
    dist,
    V,
    F,
    kd_bgr=(180, 180, 180),
    alpha=1.0,
    ambient=0.22,
    headlamp=True,
    cull=True,
):
    # Solid color Lambert shading using material Kd (no texture)
    pts2, z = project_points_with_z(V, K, dist)
    tris2 = pts2[F]
    tris3 = V[F]
    area2 = (tris2[:, 1, 0] - tris2[:, 0, 0]) * (tris2[:, 2, 1] - tris2[:, 0, 1]) - (
        tris2[:, 1, 1] - tris2[:, 0, 1]
    ) * (tris2[:, 2, 0] - tris2[:, 0, 0])
    vis = (area2 > 0) if cull else np.ones(len(F), dtype=bool)
    if not np.any(vis):
        return False
    t2v, t3v, Fv = tris2[vis], tris3[vis], F[vis]
    z_tri = z[Fv].mean(axis=1)
    order = np.argsort(z_tri)
    overlay = frame.copy()
    kb, kg, kr = map(float, kd_bgr)
    for fi in order:
        tri2 = t2v[fi].astype(np.int32)
        a, b, c = t3v[fi][0], t3v[fi][1], t3v[fi][2]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        if headlamp:
            centroid = (a + b + c) / 3.0
            l = -centroid
            l /= np.linalg.norm(l) + 1e-12
        else:
            l = np.array([0, 0, 1], np.float32)
        lam = max(ambient, float(np.dot(n, l)))
        col = (
            int(np.clip(kb * lam, 0, 255)),
            int(np.clip(kg * lam, 0, 255)),
            int(np.clip(kr * lam, 0, 255)),
        )
        cv2.fillConvexPoly(overlay, tri2, col, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, dst=frame)
    return True