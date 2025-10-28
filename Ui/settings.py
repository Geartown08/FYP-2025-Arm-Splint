import cv2
from config import BAR_H, state

# --- Drawing Utilities ---
def draw_gradient_rect(img, x, y, w, h, color1, color2, radius=8):
    overlay = img.copy()
    for i in range(h):
        ratio = i / h
        color = tuple(int(c1*(1-ratio) + c2*ratio) for c1, c2 in zip(color1, color2))
        cv2.line(overlay, (x+radius, y+i), (x+w-radius, y+i), color, 1)
    cv2.ellipse(overlay, (x+radius, y+radius), (radius, radius), 180, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+w-radius, y+radius), (radius, radius), 270, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+radius, y+h-radius), (radius, radius), 90, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+w-radius, y+h-radius), (radius, radius), 0, 0, 90, color2, -1)
    cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color2, -1)
    cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color2, -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), 2)

def draw_slider(img, x, y, w, h, value, min_val=0, max_val=1, label=""):
    cv2.rectangle(img, (x, y+h//3), (x+w, y+2*h//3), (80, 80, 80), -1)
    fill_w = int((value - min_val) / (max_val - min_val) * w)
    cv2.rectangle(img, (x, y+h//3), (x+fill_w, y+2*h//3), (0, 150, 0), -1)
    cv2.rectangle(img, (x, y+h//3), (x+w, y+2*h//3), (40, 40, 40), 2)
    cv2.putText(img, f"{label}: {value:.2f}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# --- Button Class ---
class Button:
    def __init__(self, x, y, w, h, label, cb=None, get_label=None):
        self.r = (x, y, w, h)
        self.label = label
        self.cb = cb
        self.get_label = get_label
        self.active = False

    def contains(self, px, py):
        x, y, w, h = self.r
        return x <= px < x + w and y <= py < y + h

    def draw(self, img):
        x, y, w, h = self.r
        txt = self.label if self.get_label is None else self.get_label()
        base_color = (60, 60, 60)
        gradient_start = (0, 150, 0) if self.active else (100, 100, 100)
        gradient_end = (40, 40, 40) if self.active else base_color
        draw_gradient_rect(img, x, y, w, h, gradient_start, gradient_end, radius=10)
        cv2.putText(img, txt, (x+7, y+h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, txt, (x+6, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

# --- Persistent UI Buttons ---
def layout_buttons():
    # Ensure key exists
    if 'settings_open' not in state:
        state['settings_open'] = False

    btns = []
    # Settings Tab
    b_settings = Button(8, 8, 120, 34, "Settings",
                        cb=lambda: state.__setitem__('settings_open', not state['settings_open']))
    btns.append(b_settings)

    # Quit Button
    b_quit = Button(140, 8, 100, 34, "Quit",
                    cb=lambda: state.__setitem__("quit", True))
    btns.append(b_quit)

    return btns

# --- Settings Panel ---
class SettingsPanel:
    def __init__(self):
        self.x, self.y = 8, 50
        self.w, self.h = 300, 250
        self.margin = 20
        self.slider_h = 30

        # Persistent toggle buttons inside panel
        self.b_detect = Button(
            self.x+self.margin, self.y+self.margin+200, 120, 30, "Detect",
            cb=lambda: state.__setitem__("detect_fast", not state['detect_fast']),
            get_label=lambda: f"Detect {'FAST' if state['detect_fast'] else 'SAFE'}"
        )
        self.b_lock = Button(
            self.x+self.margin+140, self.y+self.margin+200, 140, 30, "Lock",
            cb=lambda: state.__setitem__("locked", not state['locked']),
            get_label=lambda: "UNLOCK SHAPE" if state['locked'] else "LOCK SHAPE"
        )

    def draw(self, img, mouse_pos=None):
        if not state.get('settings_open', False):
            return

        # Draw background
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), (50, 50, 50), -1)

        # Draw sliders with handles
        sliders = [
            ("Mesh Opacity", state['alpha'], 0.0, 1.0),
            ("Length", state['t_max'], 0.5, 1.0),
            ("Decimation", state['decim_ratio'], 0.5, 1.0),
            ("Wrist cm", state['wrist_cm'], 0.0, 20.0),
            ("Forearm cm", state['fore_cm'], 0.0, 20.0)
        ]

        for i, (label, val, vmin, vmax) in enumerate(sliders):
            y_pos = self.y + self.margin + i*40
            self.draw_slider(img, self.x+self.margin, y_pos, self.w-2*self.margin,
                             self.slider_h, val, vmin, vmax, label, mouse_pos)

        # Draw toggle buttons
        self.b_detect.draw(img)
        self.b_lock.draw(img)

    def draw_slider(self, img, x, y, w, h, value, min_val, max_val, label="", mouse_pos=None):
        # Draw track
        cv2.rectangle(img, (x, y+h//3), (x+w, y+2*h//3), (80, 80, 80), -1)
        
        # Draw filled portion
        fill_w = int((value - min_val) / (max_val - min_val) * w)
        cv2.rectangle(img, (x, y+h//3), (x+fill_w, y+2*h//3), (0, 150, 0), -1)

        # Draw handle
        handle_x = x + fill_w
        handle_y = y + h//2
        cv2.circle(img, (handle_x, handle_y), h//2, (0, 200, 0), -1)
        cv2.circle(img, (handle_x, handle_y), h//2, (0, 0, 0), 2)

        # Hover effect
        if mouse_pos:
            mx, my = mouse_pos
            if (handle_x - h//2 <= mx <= handle_x + h//2) and (handle_y - h//2 <= my <= handle_y + h//2):
                cv2.circle(img, (handle_x, handle_y), h//2 + 3, (255,255,255), 2)

        # Border
        cv2.rectangle(img, (x, y+h//3), (x+w, y+2*h//3), (40, 40, 40), 2)

        # Label
        cv2.putText(img, f"{label}: {value:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    def handle_click(self, mx, my):
        # Check toggle buttons first
        if self.b_detect.contains(mx, my):
            self.b_detect.cb()
            return True
        if self.b_lock.contains(mx, my):
            self.b_lock.cb()
            return True

        # Sliders
        rel_y = my - self.y - self.margin
        slider_idx = int(rel_y // 40)
        slider_x, slider_w = self.x + self.margin, self.w - 2*self.margin
        val_ratio = (mx - slider_x) / slider_w
        val_ratio = max(0.0, min(1.0, val_ratio))  # clamp
        step = 0.01  # smooth step
        val_ratio = round(val_ratio / step) * step

        if slider_idx == 0:
            state['alpha'] = val_ratio
        elif slider_idx == 1:
            state['t_max'] = 0.5 + 0.5*val_ratio
        elif slider_idx == 2:
            state['decim_ratio'] = 0.5 + 0.5*val_ratio
            state['needs_mesh_reload'] = True
        elif slider_idx == 3:
            state['wrist_cm'] = val_ratio * 20
        elif slider_idx == 4:
            state['fore_cm'] = val_ratio * 20

        return True if 0 <= slider_idx <= 4 else False


# --- Global Panel ---
settings_panel = SettingsPanel()

# --- Mouse callback ---
def on_mouse(event, mx, my, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    buttons = param['buttons']

    # Check main buttons (bar)
    for b in buttons:
        if b.contains(mx, my):
            b.cb()
            return

    # Check settings panel if open
    if state.get('settings_open', False):
        # Subtract BAR_H from y
        settings_panel.handle_click(mx, my - BAR_H)
