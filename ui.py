import cv2
from config import state, UI_H, UI_ROWS, BAR_H


class Button:
    def __init__(self, x, y, w, h, label, cb, toggle_group=None, get_label=None):
        self.r = (x, y, w, h)
        self.label = label
        self.cb = cb
        self.toggle_group = toggle_group
        self.active = False
        self.get_label = get_label

    def contains(self, px, py):
        x, y, w, h = self.r
        return (x <= px < x+w) and (y <= py < y+h)

    def draw(self, img):
        x, y, w, h = self.r
        txt = self.label if self.get_label is None else self.get_label()
        col = (60, 60, 60) if not self.active else (90, 90, 90)
        cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), -1)
        cv2.rectangle(img, (x+2, y+2), (x+w-2, y+h-2), col, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), 1)
        cv2.putText(img, txt, (x+6, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)


def layout_buttons():
    btns = []
    w, h, pad = 100, 34, 8

    # Row 1: opacity, length, decimation
    x, y = 8, 8

    def set_alpha(a):
        def _():
            state['alpha'] = a
            for b in btns:
                if b.toggle_group == 'alpha':
                    b.active = False
            btn_a50.active = (a == 0.5)
            btn_a100.active = (a == 1.0)
        return _
    btn_a50 = Button(x, y, w, h, 'Mesh 50%',
                     set_alpha(0.5), toggle_group='alpha')
    x += w+pad
    btn_a100 = Button(x, y, w, h, 'Mesh 100%',
                      set_alpha(1.0), toggle_group='alpha')
    x += w+pad

    def set_len(frac):
        def _():
            state['t_max'] = frac
            for b in btns:
                if b.toggle_group == 'length':
                    b.active = False
            btn_l50.active = (frac == 0.5)
            btn_l75.active = (frac == 0.75)
            btn_l100.active = (frac == 1.0)
        return _
    btn_l50 = Button(x, y, w, h, 'Length 50%',
                     set_len(0.5),  toggle_group='length')
    x += w+pad
    btn_l75 = Button(x, y, w, h, 'Length 75%',
                     set_len(0.75), toggle_group='length')
    x += w+pad
    btn_l100 = Button(x, y, w, h, 'Length 100%',
                      set_len(1.0),  toggle_group='length')
    x += w+pad

    def set_decim(ratio):
        def _():
            state['decim_ratio'] = ratio
            for b in btns:
                if b.toggle_group == 'decim':
                    b.active = False
            btn_d100.active = (ratio == 1.0)
            btn_d75.active = (ratio == 0.75)
            btn_d50.active = (ratio == 0.5)
            state['needs_mesh_reload'] = True
        return _
    btn_d100 = Button(x, y, w, h, 'Decim 100',
                      set_decim(1.0),  toggle_group='decim')
    x += w+pad
    btn_d75 = Button(x, y, w, h, 'Decim 75',
                     set_decim(0.75), toggle_group='decim')
    x += w+pad
    btn_d50 = Button(x, y, w, h, 'Decim 50',
                     set_decim(0.5),  toggle_group='decim')

    # Row 2: wrist/fore +/-/value, Detect, Lock, Quit
    x, y = 8, UI_H + 8
    def w_plus():  state['wrist_cm'] = round(state['wrist_cm']+0.5, 1)
    def w_minus(): state['wrist_cm'] = round(
        max(8.0, state['wrist_cm']-0.5), 1)

    def f_plus():  state['fore_cm'] = round(state['fore_cm'] + 0.5, 1)
    def f_minus(): state['fore_cm'] = round(
        max(10.0, state['fore_cm'] - 0.5), 1)

    b_wm = Button(x, y, w, h, 'Wrist −', w_minus)
    x += w+pad
    b_wp = Button(x, y, w, h, 'Wrist +', w_plus)
    x += w+pad
    b_wv = Button(x, y, w, h, 'Wrist cm', lambda: None,
                  get_label=lambda: f"Wrist {state['wrist_cm']:.1f} cm")
    x += w+pad

    b_fm = Button(x, y, w, h, 'Fore −', f_minus)
    x += w+pad
    b_fp = Button(x, y, w, h, 'Fore +', f_plus)
    x += w+pad
    b_fv = Button(x, y, w, h, 'Fore cm', lambda: None,
                  get_label=lambda: f"Fore {state['fore_cm']:.1f} cm")
    x += w+pad

    def toggle_detect(): state['detect_fast'] = not state['detect_fast']
    b_detect = Button(x, y, 130, h, 'Detect', toggle_detect,
                      get_label=lambda: f"Detect {'FAST' if state['detect_fast'] else 'SAFE'}")
    x += 130+pad

    def toggle_lock():
        # toggles in the mouse callback; actual mesh freezing handled in main loop
        state['locked'] = not state['locked']
    b_lock = Button(x, y, 120, h, 'Lock', toggle_lock,
                    get_label=lambda: f"{'UNLOCK' if state['locked'] else 'LOCK'} SHAPE")
    x += 120+pad

    b_quit = Button(x, y, w, h, 'Quit',
                    lambda: state.__setitem__('quit', True))

    btns = [btn_a50, btn_a100, btn_l50, btn_l75, btn_l100, btn_d100, btn_d75, btn_d50,
            b_wm, b_wp, b_wv, b_fm, b_fp, b_fv, b_detect, b_lock, b_quit]
    (btn_a50 if state['alpha'] == 0.5 else btn_a100).active = True
    {0.5: btn_l50, 0.75: btn_l75, 1.0: btn_l100}[state['t_max']].active = True
    {1.0: btn_d100, 0.75: btn_d75, 0.5: btn_d50}[
        state['decim_ratio']].active = True
    return btns


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and y < BAR_H:
        for b in param['buttons']:
            if b.contains(x, y):
                b.cb()
