# Handles drawing of the on-screen HUD (status overlay)

import cv2
import numpy as np


def draw_hud(frame, bar, settings_panel, wrist_ids_seen, fore_ids_seen,
             state, V_model, F_model, UV_model, TEX, BAR_H):
    """
    Compose the main output with HUD text drawn below the settings panel.
    Returns the composited image (bar + frame + HUD text).
    """
    # Compose the base output image (bar on top of camera frame)
    out = np.vstack([bar, frame])

    # Calculate HUD base offset below the settings panel if open
    panel_offset = settings_panel.h + 50 if state.get('settings_open', False) else 0
    hud_start_y = BAR_H + panel_offset

    # Draw wrist/forearm IDs
    cv2.putText(out, f"Wrist IDs: {sorted(wrist_ids_seen)}", (12, hud_start_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(out, f"Fore IDs: {sorted(fore_ids_seen)}", (12, hud_start_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Compose the status string
    status_line = (
        f"Mesh:{'OK' if V_model is not None else 'MISS'}  "
        f"V={0 if V_model is None else len(V_model)}  "
        f"F={0 if F_model is None else len(F_model)}  "
        f"Decim={state.get('decim_ratio', 1.0):.2f}  "
        f"Cull={'ON' if state.get('cull', False) else 'OFF'}  "
        f"Calib={'OK' if state.get('use_calib', True) else 'FALLBACK'}  "
        f"Detect={'FAST' if state.get('detect_fast', False) else 'SAFE'}  "
        f"Tex={'YES' if (UV_model is not None and TEX is not None and state['decim_ratio'] >= 0.999) else 'NO'}  "
        f"Lock={'ON' if state.get('locked', False) else 'OFF'}"
    )

    # Auto-wrap long HUD text
    max_width = frame.shape[1] - 24  # padding from right edge
    words = status_line.split()
    lines = []
    current_line = ""

    font_scale = 0.6
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    for w in words:
        test_line = current_line + ("" if current_line == "" else " ") + w
        size, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if size[0] > max_width:
            lines.append(current_line)
            current_line = w
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    # Draw wrapped lines
    line_y = hud_start_y + 64
    for line in lines:
        cv2.putText(out, line, (12, line_y),
                    font, font_scale, (0, 255, 255), 1)
        line_y += 22

    return out
