# arm_splint_wrap_buttons_tex_lock.py
# OBJ materials (texture + Kd) with Lambert shading; LOCK SHAPE button; SAFE/FAST detect button

import os
import sys
import cv2
import numpy as np
from collections import deque
from components.camera import auto_open_camera, open_camera
from config import FRAME_W, FRAME_H, FRAME_FPS, CALIB_PATH, DICT, WRIST_IDS, FORE_IDS, T_BINS, MESH_PATH, MESH_UNITS, state, UI_H, BAR_H, DETECT_SCALE, DETECT_EVERY, ROI_MARGIN, WIN, EMA_POS, EMA_AXIS, EMA_RAD, MAX_CENTER_STEP
from components.exporter import export_surface_obj, export_thickened_obj
from components.renderer import painter_fill_mesh_kd, painter_fill_mesh_textured
from Ui.settings import layout_buttons, on_mouse, settings_panel
from components.utils import circle_radius_from_ids, ema, clamp_step, fit_plane, solve_tag_pose
from components.mesh_loader import build_t_bins, compute_slice_frames, fit_model_axis, load_mesh_fast, load_mesh_textured, model_uv_from_vertices, wrap_vertices_to_arm_binned

# ============================== MAIN ===============================


def main():
    # Calibration (toggle with 'k')
    cal = np.load(CALIB_PATH)
    K_base, dist_base = cal['K'].astype(
        np.float32), cal['dist'].astype(np.float32)

    cap, CAM_INDEX = auto_open_camera(
        max_devices=5,
        w=FRAME_W,
        h=FRAME_H,
        fps=FRAME_FPS
    )

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
            corners, ids, rej = aruco.detectMarkers(
                img, dictionary=dictionary, parameters=params)
            return corners, ids, rej

    # Mesh load (textured if decim=1.0; else fast decimated, UV-less)
    V_model = F_model = t_model = th_model = None
    t_bins = []
    UV_model = None
    TEX = None
    KD_BGR = (180, 180, 180)
    if os.path.exists(MESH_PATH):
        try:
            if state['decim_ratio'] >= 0.999:
                V_model, F_model, UV_model, TEX, KD_BGR = load_mesh_textured(
                    MESH_PATH, units=MESH_UNITS)
                print(f"[MESH] Textured load: V={len(V_model)} F={len(F_model)} "
                      f"UV={'yes' if UV_model is not None else 'no'} TEX={'yes' if TEX is not None else 'no'}")
            else:
                V_model, F_model, t_model, th_model, t_bins = load_mesh_fast(
                    MESH_PATH, units=MESH_UNITS, decim_ratio=state['decim_ratio'])
                UV_model = None
                TEX = None
                print("[WARN] Decimation used → texture disabled (UVs not preserved)")
            C0, xhat0, yhat0, zhat0, zmin0, zmax0 = fit_model_axis(V_model)
            t_model, th_model = model_uv_from_vertices(
                V_model, C0, xhat0, yhat0, zhat0, zmin0, zmax0)
            t_bins = build_t_bins(t_model, T_BINS=64)
        except Exception as e:
            print(f"[MESH] Failed: {e}")
    else:
        print(f"[MESH] Missing {MESH_PATH}")

    # Smoothing state
    wrist_ids_seen, fore_ids_seen = set(), set()
    cw_win, nw_win, rw_win = deque(maxlen=WIN), deque(
        maxlen=WIN), deque(maxlen=WIN)
    cf_win, nf_win, rf_win = deque(maxlen=WIN), deque(
        maxlen=WIN), deque(maxlen=WIN)
    sm_cw = sm_nw = sm_rw = None
    sm_cf = sm_nf = sm_rf = None

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
        if not ok:
            break

        # Mesh reload after decim change
        if state.get('needs_mesh_reload', False) and os.path.exists(MESH_PATH):
            try:
                if state['decim_ratio'] >= 0.999:
                    V_model, F_model, UV_model, TEX, KD_BGR = load_mesh_textured(
                        MESH_PATH, units=MESH_UNITS)
                    print(f"[MESH] Textured reload: V={len(V_model)} F={len(F_model)} "
                          f"UV={'yes' if UV_model is not None else 'no'} TEX={'yes' if TEX is not None else 'no'}")
                else:
                    V_model, F_model, t_model, th_model, t_bins = load_mesh_fast(
                        MESH_PATH, units=MESH_UNITS, decim_ratio=state['decim_ratio'])
                    UV_model = None
                    TEX = None
                    print(
                        "[WARN] Decimation used → texture disabled (UVs not preserved)")
                C0, xhat0, yhat0, zhat0, zmin0, zmax0 = fit_model_axis(V_model)
                t_model, th_model = model_uv_from_vertices(
                    V_model, C0, xhat0, yhat0, zhat0, zmin0, zmax0)
                t_bins = build_t_bins(t_model, T_BINS)
                # clearing lock on reload keeps things consistent
                locked_mesh = None
                state['locked'] = False
            except Exception as e:
                print(f"[MESH] Reload failed: {e}")
            state['needs_mesh_reload'] = False

        # UI bar
        bar = np.full((BAR_H, frame.shape[1], 3), 30, np.uint8)
        for b in buttons:
            b.draw(bar)

        # --- Draw Settings Panel if open ---
        settings_panel.draw(frame)

        # Intrinsics choice
        if state['use_calib']:
            K_use, dist_use = K_base, dist_base
        else:
            fx = fy = 800.0
            cx, cy = FRAME_W/2.0, FRAME_H/2.0
            K_use = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
            dist_use = np.zeros((5, 1), np.float32)

        # ---------- Detection (button controlled) ----------
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = None
        ids = None

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
                    x0, y0, x1, y1 = bbox
                    roi = gsmall[y0:y1, x0:x1]
                    c_s, i_s, _ = detect_markers(roi)
                    if i_s is None:
                        return None, None, None
                    c_s = [c + np.array([[[x0, y0]]], dtype=c.dtype)
                           for c in c_s]
                    return c_s, i_s, None
                if _last_bbox_small is not None:
                    x0, y0, x1, y1 = _last_bbox_small
                    cx2, cy2 = (x0+x1)//2, (y0+y1)//2
                    w, h = (x1-x0), (y1-y0)
                    w2 = int(w*ROI_MARGIN/2)
                    h2 = int(h*ROI_MARGIN/2)
                    xs0 = max(0, cx2 - w2)
                    ys0 = max(0, cy2 - h2)
                    xs1 = min(gsmall.shape[1], cx2 + w2)
                    ys1 = min(gsmall.shape[0], cy2 + h2)
                    if xs1 > xs0 and ys1 > ys0:
                        corners_s, ids, _ = detect_roi((xs0, ys0, xs1, ys1))
                    else:
                        corners_s, ids, _ = detect_full()
                else:
                    corners_s, ids, _ = detect_full()
                if ids is not None:
                    corners = [c / DETECT_SCALE for c in corners_s]
                    _last_corners, _last_ids = corners, ids
                    pts = np.vstack([c.reshape(-1, 2) for c in corners_s])
                    x0, y0 = np.maximum(pts.min(axis=0).astype(int) - 4, 0)
                    x1 = min(int(pts[:, 0].max()) + 4, gsmall.shape[1])
                    y1 = min(int(pts[:, 1].max()) + 4, gsmall.shape[0])
                    _last_bbox_small = (x0, y0, x1, y1)
                else:
                    corners, ids = _last_corners, _last_ids
            else:
                corners, ids = _last_corners, _last_ids

        # Optional debug overlay
        if ids is not None and corners is not None:
            try:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception:
                pass
            cv2.putText(frame, f"Detected IDs: {list(ids.flatten())[:10]}...",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No markers detected", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Pose smoothing (only meaningful when not locked)
        wrist_pts, fore_pts = [], []
        if ids is not None and corners is not None and not state['locked']:
            for cid, cs in zip(ids.flatten(), corners):
                okp, rvec, tvec = solve_tag_pose(cs, K_use, dist_use)
                if not okp:
                    continue
                pos = tvec.reshape(3)
                if int(cid) in WRIST_IDS:
                    wrist_pts.append(pos)
                    wrist_ids_seen.add(int(cid))
                elif int(cid) in FORE_IDS:
                    fore_pts.append(pos)
                    fore_ids_seen.add(int(cid))

        if not state['locked']:
            if len(wrist_pts) >= 3:
                cw_raw, nw_raw, _, _ = fit_plane(wrist_pts, prev_n=sm_nw)
                cw_win.append(cw_raw)
                if sm_nw is not None and float(np.dot(nw_raw, sm_nw)) < 0.0:
                    nw_raw = -nw_raw
                nw_win.append(nw_raw)
                if wrist_ids_seen:
                    rw_raw = circle_radius_from_ids(
                        min(wrist_ids_seen), max(wrist_ids_seen))
                    rw_win.append(rw_raw)
                cw_w = np.mean(cw_win, axis=0)
                n_sum = np.sum(np.array(nw_win), axis=0)
                nw_w = n_sum / (np.linalg.norm(n_sum)+1e-9)
                rw_w = np.mean(rw_win) if len(rw_win) > 0 else None
                conf = min(1.0, len(wrist_pts)/8.0)
                sm_cw = ema(sm_cw, cw_w, EMA_POS*conf)
                sm_cw = clamp_step(sm_cw, cw_w, MAX_CENTER_STEP)
                sm_nw = ema(sm_nw, nw_w, EMA_AXIS*conf)
                sm_nw /= (np.linalg.norm(sm_nw)+1e-9)
                if rw_w is not None:
                    sm_rw = ema(sm_rw, rw_w, EMA_RAD*conf)

            if len(fore_pts) >= 3:
                cf_raw, nf_raw, _, _ = fit_plane(fore_pts, prev_n=sm_nf)
                cf_win.append(cf_raw)
                if sm_nf is not None and float(np.dot(nf_raw, sm_nf)) < 0.0:
                    nf_raw = -nf_raw
                nf_win.append(nf_raw)
                if fore_ids_seen:
                    rf_raw = circle_radius_from_ids(
                        min(fore_ids_seen), max(fore_ids_seen))
                    rf_win.append(rf_raw)
                cf_w = np.mean(cf_win, axis=0)
                n_sum = np.sum(np.array(nf_win), axis=0)
                nf_w = n_sum / (np.linalg.norm(n_sum)+1e-9)
                rf_w = np.mean(rf_win) if len(rf_win) > 0 else None
                conf = min(1.0, len(fore_pts)/8.0)
                sm_cf = ema(sm_cf, cf_w, EMA_POS*conf)
                sm_cf = clamp_step(sm_cf, cf_w, MAX_CENTER_STEP)
                sm_nf = ema(sm_nf, nf_w, EMA_AXIS*conf)
                sm_nf /= (np.linalg.norm(sm_nf)+1e-9)
                if rf_w is not None:
                    sm_rf = ema(sm_rf, rf_w, EMA_RAD*conf)

        # Draw mesh (locked uses frozen copy; unlocked recomputes)
        V_to_draw = None
        if state['locked'] and locked_mesh is not None:
            V_to_draw = locked_mesh
        else:
            if all(x is not None for x in (sm_cw, sm_nw, sm_rw, sm_cf, sm_nf, sm_rf)) and V_model is not None:
                rw_use = (state['wrist_cm'] / 100.0) / (2*np.pi)
                rf_use = (state['fore_cm'] / 100.0) / (2*np.pi)
                C_s, U_s, V_s, R_s = compute_slice_frames(
                    sm_cw, sm_nw, rw_use, sm_cf, sm_nf, rf_use)
                V_arm = np.empty_like(V_model, dtype=np.float32)
                wrap_vertices_to_arm_binned(
                    V_arm, th_model, t_bins, C_s, U_s, V_s, R_s, t_max=state['t_max'])
                last_wrapped = V_arm.copy()
                V_to_draw = V_arm

        # If user just pressed LOCK and we have a wrapped mesh, freeze it
        if state['locked'] and locked_mesh is None and last_wrapped is not None:
            locked_mesh = last_wrapped.copy()
            print(
                "[LOCK] Shape frozen. Further tracking changes won't affect mesh until UNLOCK.")

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out,  f"Fore IDs: {sorted(fore_ids_seen)}", (12, BAR_H+42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        hud_y = BAR_H + 64
        cv2.putText(out,
                    f"Mesh:{'OK' if V_model is not None else 'MISS'}  "
                    f"V={0 if V_model is None else len(V_model)}  "
                    f"F={0 if F_model is None else len(F_model)}  "
                    f"Decim={state.get('decim_ratio', 1.0):.2f}  "
                    f"Cull={'ON' if state.get('cull', False) else 'OFF'}  "
                    f"Calib={'OK' if state.get('use_calib', True) else 'FALLBACK'}  "
                    f"Detect={'FAST' if state.get('detect_fast', False) else 'SAFE'}  "
                    f"Tex={'YES' if (UV_model is not None and TEX is not None and state['decim_ratio'] >= 0.999) else 'NO'}  "
                    f"Lock={'ON' if state.get('locked', False) else 'OFF'}",
                    (12, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Arm Splint", out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or state['quit']:
            break
        if k == ord('r'):
            wrist_ids_seen.clear()
            fore_ids_seen.clear()
            cw_win.clear()
            nw_win.clear()
            rw_win.clear()
            cf_win.clear()
            nf_win.clear()
            rf_win.clear()
            sm_cw = sm_nw = sm_rw = None
            sm_cf = sm_nf = sm_rf = None
            locked_mesh = None
            state['locked'] = False
            print("[RESET] cleared IDs + windows + smoothing; UNLOCKED.")
        if k == ord('c'):
            wrist_ids_seen.clear()
            fore_ids_seen.clear()
            rw_win.clear()
            rf_win.clear()
            print("[CLEAR] IDs cleared (pose smoothing kept)")
        if k == ord('e'):
            mesh_to_export = locked_mesh if (
                state['locked'] and locked_mesh is not None) else last_wrapped
            export_surface_obj(mesh_to_export, F_model,
                               filename="export_surface.obj", units='mm')
        if k == ord('x'):
            mesh_to_export = locked_mesh if (
                state['locked'] and locked_mesh is not None) else last_wrapped
            export_thickened_obj(
                mesh_to_export, F_model, thickness_mm=3.0, filename="export_solid.obj", units='mm')
        if k == ord('b'):
            state['cull'] = not state['cull']
            print(f"[RENDER] Cull {'ON' if state['cull'] else 'OFF'}")
        if k == ord('k'):
            state['use_calib'] = not state['use_calib']
            print(
                f"[CALIB] {'calibration' if state['use_calib'] else 'fallback intrinsics'}")
        if k == ord('m'):
            state['draw_mesh'] = not state['draw_mesh']
            print(f"[RENDER] Mesh {'ON' if state['draw_mesh'] else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
