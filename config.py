import numpy as np

# Camera settings
CAM_INDEX = 1
FRAME_W, FRAME_H, FRAME_FPS = 960, 540, 30

# FAST detection params
DETECT_SCALE = 0.5
DETECT_EVERY = 3
ROI_MARGIN = 1.40

# ArUco / Geometry
CALIB_PATH = "camcalib.npz"
DICT = 4  # cv2.aruco.DICT_4X4_50
WRIST_IDS = set(range(0, 14))
FORE_IDS = set(range(14, 33))
MARKER_SIZE = 0.010
GAP_SIZE = 0.002
WIN = 5
EMA_POS = 0.65
EMA_AXIS = 0.65
EMA_RAD = 0.65
MAX_CENTER_STEP = 0.05
MAX_RADIUS_STEP = 0.005

# Mesh input
MESH_PATH = "splint.obj"
MESH_UNITS = "mm"
T_BINS = 64

# UI
UI_H = 50
UI_ROWS = 2
BAR_H = UI_H * UI_ROWS

# Default state
state = {
    'alpha': 1.0,
    't_max': 1.0,
    'wrist_cm': 16.0,
    'fore_cm': 24.0,
    'decim_ratio': 1.0,
    'needs_mesh_reload': False,
    'cull': False,
    'use_calib': True,
    'draw_mesh': True,
    'detect_fast': False,
    'locked': False,
    'quit': False,
    'settings_open': False
}
