import cv2
from Components.config import state, UI_H, BAR_H


def draw_gradient_rect(img, x, y, w, h, color1, color2, radius=8):
    """Draw a rounded rectangle with vertical gradient."""
    # create overlay for blending
    overlay = img.copy()

    # draw main rectangle with gradient
    for i in range(h):
        ratio = i / h
        color = tuple(int(c1*(1-ratio) + c2*ratio)
                      for c1, c2 in zip(color1, color2))
        cv2.line(overlay, (x+radius, y+i), (x+w-radius, y+i), color, 1)

    # fill corners
    cv2.ellipse(overlay, (x+radius, y+radius),
                (radius, radius), 180, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+w-radius, y+radius),
                (radius, radius), 270, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+radius, y+h-radius),
                (radius, radius), 90, 0, 90, color2, -1)
    cv2.ellipse(overlay, (x+w-radius, y+h-radius),
                (radius, radius), 0, 0, 90, color2, -1)

    # draw rectangle bodies for corners
    cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color2, -1)
    cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color2, -1)

    # blend with original image
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # draw border
    cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), 2)


class Button:
    def __init__(self, x, y, w, h, label, cb=None, toggle_group=None, get_label=None):
        self.r = (x, y, w, h)
        self.label = label
        self.cb = cb
        self.toggle_group = toggle_group
        self.active = False
        self.get_label = get_label

    def contains(self, px, py):
        x, y, w, h = self.r
        return x <= px < x + w and y <= py < y + h

    def draw(self, img):
        x, y, w, h = self.r
        txt = self.label if self.get_label is None else self.get_label()

        # Colors
        base_color = (60, 60, 60)
        active_color = (0, 150, 0)
        gradient_start = active_color if self.active else (100, 100, 100)
        gradient_end = (40, 40, 40) if self.active else base_color

        draw_gradient_rect(img, x, y, w, h, gradient_start,
                           gradient_end, radius=10)

        # Text shadow
        cv2.putText(img, txt, (x+7, y+h-7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, txt, (x+6, y+h-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)


def make_toggle_cb(label, group, buttons):
    def cb():
        for b in buttons:
            if b.toggle_group == group:
                b.active = False
            if b.label == label:
                b.active = True
        # Update state dynamically based on label
        if group == "alpha":
            state['alpha'] = 0.5 if "50%" in label else 1.0
        elif group == "length":
            state['t_max'] = 0.5 if "50%" in label else 0.75 if "75%" in label else 1.0
        elif group == "decim":
            state['decim_ratio'] = 0.5 if "50" in label else 0.75 if "75" in label else 1.0
            state['needs_mesh_reload'] = True
    return cb


def layout_buttons():
    btns = []
    w, h, pad = 100, 34, 8

    # Row 1: Mesh opacity, length, decimation
    x, y = 8, 8
    row1_labels = [
        ("Mesh 50%", "alpha"),
        ("Mesh 100%", "alpha"),
        ("Length 50%", "length"),
        ("Length 75%", "length"),
        ("Length 100%", "length"),
        ("Decim 100", "decim"),
        ("Decim 75", "decim"),
        ("Decim 50", "decim"),
    ]
    for label, group in row1_labels:
        b = Button(x, y, w, h, label, toggle_group=group)
        b.cb = make_toggle_cb(label, group, btns)
        # Pre-activate according to state
        if group == "alpha" and state['alpha'] == 0.5 and "50" in label:
            b.active = True
        if group == "alpha" and state['alpha'] == 1.0 and "100" in label:
            b.active = True
        if group == "length" and ((state['t_max'] == 0.5 and "50" in label) or
                                  (state['t_max'] == 0.75 and "75" in label) or
                                  (state['t_max'] == 1.0 and "100" in label)):
            b.active = True
        if group == "decim" and ((state['decim_ratio'] == 0.5 and "50" in label) or
                                 (state['decim_ratio'] == 0.75 and "75" in label) or
                                 (state['decim_ratio'] == 1.0 and "100" in label)):
            b.active = True
        btns.append(b)
        x += w + pad

    # Row 2: Wrist/Fore ±, Detect, Lock, Quit
    x, y = 8, UI_H + 8

    # Wrist/Fore adjustments
    def make_inc_cb(key, delta): return lambda: state.__setitem__(
        key, round(max(0, state[key]+delta), 1))

    def make_val_btn(key): return lambda: None

    adjustments = [
        ("Wrist −", make_inc_cb("wrist_cm", -0.5)),
        ("Wrist +", make_inc_cb("wrist_cm", 0.5)),
        ("Wrist cm", make_val_btn("wrist_cm"),
         lambda: f"Wrist {state['wrist_cm']:.1f} cm"),
        ("Fore −", make_inc_cb("fore_cm", -0.5)),
        ("Fore +", make_inc_cb("fore_cm", 0.5)),
        ("Fore cm", make_val_btn("fore_cm"),
         lambda: f"Fore {state['fore_cm']:.1f} cm"),
    ]
    for item in adjustments:
        label, cb = item[0], item[1]
        get_label = item[2] if len(item) > 2 else None
        b = Button(x, y, w, h, label, cb=cb, get_label=get_label)
        btns.append(b)
        x += w + pad

    # Detect toggle
    b_detect = Button(x, y, 130, h, "Detect", lambda: state.__setitem__("detect_fast", not state['detect_fast']),
                      get_label=lambda: f"Detect {'FAST' if state['detect_fast'] else 'SAFE'}")
    btns.append(b_detect)
    x += 130 + pad

    # Lock toggle
    b_lock = Button(x, y, 120, h, "Lock", lambda: state.__setitem__("locked", not state['locked']),
                    get_label=lambda: f"{'UNLOCK' if state['locked'] else 'LOCK'} SHAPE")
    btns.append(b_lock)
    x += 120 + pad

    # Quit
    b_quit = Button(x, y, w, h, "Quit",
                    lambda: state.__setitem__("quit", True))
    btns.append(b_quit)

    return btns


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and y < BAR_H:
        for b in param['buttons']:
            if b.contains(x, y):
                b.cb()
