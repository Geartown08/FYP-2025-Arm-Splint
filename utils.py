import cv2
import numpy as np

from config import MARKER_SIZE


def ema(prev, new, a):
    return new if prev is None else (a*new + (1-a)*prev)


def clamp_step(prev, new, max_step):
    if prev is None:
        return new
    d = new - prev
    n = float(np.linalg.norm(d))
    if n <= max_step or n == 0.0:
        return new
    return prev + d * (max_step / n)


def project_points(P3, K, dist):
    img, _ = cv2.projectPoints(
        P3.astype(np.float32), np.zeros(3), np.zeros(3), K, dist)
    return img.reshape(-1, 2).astype(int)


def project_points_with_z(P3, K, dist):
    pts2 = project_points(P3, K, dist)
    z = P3[:, 2].astype(np.float32).copy()
    return pts2, z


def circle_radius_from_ids(min_id, max_id):
    n = max_id - min_id + 1
    C = n*0.01 + (n-1)*0.002
    return C / (2*np.pi)


def fit_plane(points, prev_n=None):
    P = np.asarray(points)
    C = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - C)
    n = Vt[-1]
    if prev_n is not None and float(np.dot(n, prev_n)) < 0.0:
        n = -n
    ref = np.array([1, 0, 0], float)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0, 1, 0], float)
    u = ref - (ref @ n)*n
    u /= (np.linalg.norm(u)+1e-9)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v)+1e-9)
    return C, n/np.linalg.norm(n), u, v


def ring_segments(center, n, r, samples=64):
    ref = np.array([1, 0, 0], float)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0, 1, 0], float)
    u = ref - (ref @ n)*n
    u /= (np.linalg.norm(u)+1e-9)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v)+1e-9)
    th = np.linspace(0, 2*np.pi, samples, endpoint=False)
    pts = [center + r*np.cos(t)*u + r*np.sin(t)*v for t in th]
    return [(pts[i], pts[(i+1) % samples]) for i in range(samples)]


def solve_tag_pose(corners, K, dist):
    half = MARKER_SIZE/2.0
    obj = np.float32([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 2).astype(np.float32),
                                  K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return ok, rvec, tvec
