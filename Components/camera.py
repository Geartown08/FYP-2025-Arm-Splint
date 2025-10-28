import cv2
import sys


def quiet_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def warmup(cap, n=8):
    for _ in range(n):
        ok, _ = cap.read()
        if not ok:
            break


def open_camera(cam_index=0, w=960, h=540, fps=30):
    quiet_opencv()
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        fourcc_list = [
            cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.VideoWriter_fourcc(*"YUY2"),
            0,
        ]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        fourcc_list = [0]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        fourcc_list = [cv2.VideoWriter_fourcc(*"MJPG"), 0]
    last_err = "unknown"
    for be in backends:
        cap = cv2.VideoCapture(cam_index, be)
        if not cap.isOpened():
            last_err = f"backend {be} failed"
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok = False
        for fcc in fourcc_list:
            if fcc:
                cap.set(cv2.CAP_PROP_FOURCC, fcc)
            ok, _ = cap.read()
            if ok:
                break
        if ok:
            print(
                f"[CAM] Opened index {cam_index} via backend {be}, {w}x{h}@{fps}")
            warmup(cap, n=8)
            return cap
        cap.release()
        last_err = f"backend {be} opened but no frames"
    raise RuntimeError(f"Camera open failed: {last_err}")

def auto_open_camera(max_devices=5, w=1920, h=1080, fps=30):
    print("[CAM] Auto-detecting camera...")
    for idx in range(max_devices):
        try:
            cap = open_camera(cam_index=idx, w=w, h=h, fps=fps)
            print(f"[CAM] ✅ Using camera index: {idx}")
            return cap, idx
        except:
            print(f"[CAM] ❌ Camera index {idx} failed")
    raise RuntimeError("[CAM] ❌ No working camera found")