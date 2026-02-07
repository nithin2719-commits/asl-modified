# Sign Language to Text (Tkinter) - fast & more accurate
# Dependencies: opencv-python, cvzone, keras, enchant, pyttsx3, pillow

import math
import traceback
import cv2
import numpy as np
import pyttsx3
import enchant
from collections import deque, Counter
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk

def preprocess_crop(img):
    """Stabilize lighting for clearer predictions."""
    if img is None or img.size == 0:
        return img
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    balanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.bilateralFilter(balanced, 5, 35, 35)

# --------------- Config ----------------
MODEL_PATH = "cnn8grps_rad1_model.h5"
OFFSET = 30
MIN_HAND_DIAG = 80          # clearer shapes; still works at moderate distance
CONF_THRESH = 0.40
MARGIN_THRESH = 0.12        # top1-top2 gap
QUALITY_MIN = 0.22          # max_prob * size_score
SMOOTH_WINDOW = 3           # a bit more smoothing for accuracy
LOOP_DELAY_MS = 18          # balanced speed/precision
# ---------------------------------------

dct = enchant.Dict("en-US")
hd = HandDetector(maxHands=1, detectionCon=0.65, minTrackCon=0.55)
hd2 = HandDetector(maxHands=1, detectionCon=0.70, minTrackCon=0.55)

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.model = load_model(MODEL_PATH)
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 110)

        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self.sentence = " "
        self.suggest = [" "] * 4
        self.current_symbol = ""
        self.smoothing = deque(maxlen=SMOOTH_WINDOW)

        # UI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1300x720")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

        self.panel = tk.Label(self.root); self.panel.place(x=40, y=10, width=560, height=640)
        self.panel2 = tk.Label(self.root); self.panel2.place(x=660, y=120, width=400, height=400)

        tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 28, "bold")).place(x=40, y=660)
        tk.Label(self.root, text="Character :", font=("Courier", 22, "bold")).place(x=40, y=615)
        self.panel3 = tk.Label(self.root, font=("Courier", 26)); self.panel3.place(x=200, y=612)

        tk.Label(self.root, text="Sentence :", font=("Courier", 22, "bold")).place(x=40, y=690)
        self.panel5 = tk.Label(self.root, font=("Courier", 20), wraplength=1040, justify="left")
        self.panel5.place(x=200, y=688)

        tk.Label(self.root, text="Suggestions :", fg="red", font=("Courier", 22, "bold")).place(x=660, y=540)
        self.b1 = tk.Button(self.root, font=("Courier", 16), wraplength=320, command=self.action1)
        self.b2 = tk.Button(self.root, font=("Courier", 16), wraplength=320, command=self.action2)
        self.b3 = tk.Button(self.root, font=("Courier", 16), wraplength=320, command=self.action3)
        self.b4 = tk.Button(self.root, font=("Courier", 16), wraplength=320, command=self.action4)
        for i, btn in enumerate([self.b1, self.b2, self.b3, self.b4]):
            btn.place(x=660 + (i % 2) * 200, y=580 + (i // 2) * 60, width=180, height=50)

        tk.Button(self.root, text="Speak", font=("Courier", 16),
                  command=self.speak_fun).place(x=1100, y=580, width=120)
        tk.Button(self.root, text="Clear", font=("Courier", 16),
                  command=self.clear_fun).place(x=1100, y=630, width=120)

        self.root.after(1, self.video_loop)

    # ---------- Helpers ----------
    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def hand_diag(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def folded(self, pts, margin):
        return (pts[8][1] > pts[5][1] + margin and
                pts[12][1] > pts[9][1] + margin and
                pts[16][1] > pts[13][1] + margin and
                pts[20][1] > pts[17][1] + margin)

    def thumb_inside(self, pts, extra=8):
        palm_x_min = min(pts[5][0], pts[9][0], pts[13][0])
        palm_x_max = max(pts[5][0], pts[9][0], pts[13][0])
        return palm_x_min - extra <= pts[4][0] <= palm_x_max + extra

    def is_next(self, pts, diag):
        if diag < 100: return False
        margin = max(14, diag * 0.12)
        folded = self.folded(pts, margin)
        thumb_below = pts[4][1] > pts[2][1] + 6
        return folded and self.thumb_inside(pts, 6) and thumb_below

    def is_space(self, pts):
        return (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and
                pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])

    def is_backspace(self, pts):
        return (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0] and
                pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1] and
                pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1])

    def is_fist_A(self, pts, diag):
        margin = max(8, diag * 0.05)
        folded = self.folded(pts, margin)
        return folded and not self.thumb_inside(pts, 10)

    # ---------- Classification ----------
    def classify(self, img, diag):
        clean = preprocess_crop(img)
        white = cv2.resize(clean, (400, 400), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        prob = np.array(self.model.predict(white.reshape(1, 400, 400, 3), verbose=0)[0], dtype="float32")
        top1 = int(np.argmax(prob))
        top1p = float(prob[top1])
        prob[top1] = 0
        top2p = float(np.max(prob))
        margin = top1p - top2p
        size_score = min(1.2, diag / 150.0)  # give a boost to small-but-clear hands
        quality = top1p * size_score
        if top1p < CONF_THRESH or quality < QUALITY_MIN or margin < MARGIN_THRESH:
            return "", quality
        return top1, quality

    def group_to_letter(self, ch1):
        pts = self.pts
        if ch1 == 0:
            ch1 = 'S'
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch1 = 'A'
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                ch1 = 'T'
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch1 = 'E'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch1 = 'M'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch1 = 'N'
        elif ch1 == 2:
            ch1 = 'C' if self.distance(pts[12], pts[4]) > 42 else 'O'
        elif ch1 == 3:
            ch1 = 'G' if self.distance(pts[8], pts[12]) > 72 else 'H'
        elif ch1 == 7:
            ch1 = 'Y' if self.distance(pts[8], pts[4]) > 42 else 'J'
        elif ch1 == 4:
            ch1 = 'L'
        elif ch1 == 6:
            ch1 = 'X'
        elif ch1 == 5:
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                ch1 = 'Z' if pts[8][1] < pts[5][1] else 'Q'
            else:
                ch1 = 'P'
        elif ch1 == 1:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 'B'
            elif pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = 'D'
            elif pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 'F'
            elif pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 'I'
            elif pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = 'W'
            elif pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[4][1] < pts[9][1]:
                ch1 = 'K'
            elif (self.distance(pts[8], pts[12]) - self.distance(pts[6], pts[10])) < 8 and \
                    pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = 'U'
            elif (self.distance(pts[8], pts[12]) - self.distance(pts[6], pts[10])) >= 8 and \
                    pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[4][1] > pts[9][1]:
                ch1 = 'V'
            elif pts[8][0] > pts[12][0] and \
                    pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = 'R'
        return ch1

    def predict_letter(self, img, diag):
        ch_raw, quality = self.classify(img, diag)
        if ch_raw == "":
            return ""
        ch = self.group_to_letter(ch_raw)
        # Debounce J
        if ch == 'J' and (quality < 0.45 or diag < 120):
            return ""
        self.smoothing.append(ch)
        if len(self.smoothing) < 2:
            return ""
        return Counter(self.smoothing).most_common(1)[0][0]

    def update_sentence(self, ch):
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch
        if ch == "next" and self.prev_char != "next":
            prior = self.ten_prev_char[(self.count - 2) % 10]
            if prior == "Backspace":
                self.sentence = self.sentence[:-1]
            elif prior not in ["next", "Backspace", ""]:
                self.sentence += prior
        elif ch == " " and self.prev_char != " ":
            self.sentence += " "
        elif ch == "Backspace" and self.prev_char != "Backspace":
            self.sentence = self.sentence[:-1]
        self.prev_char = ch

    # ---------- Main loop ----------
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(40, self.video_loop)
                return
            frame = cv2.flip(frame, 1)
            hands, _ = hd.findHands(frame, draw=False, flipType=True)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.panel.config(image=imgtk); self.panel.imgtk = imgtk

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                x1, y1 = max(x - OFFSET, 0), max(y - OFFSET, 0)
                x2, y2 = min(x + w + OFFSET, frame.shape[1]), min(y + h + OFFSET, frame.shape[0])
                crop = frame[y1:y2, x1:x2]

                white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                ch = ""
                if crop.size > 0:
                    ch = ""
                    hands2, _ = hd2.findHands(crop, draw=False, flipType=True)
                    if hands2:
                        self.pts = hands2[0]['lmList']
                        diag = self.hand_diag(self.pts)
                        if diag >= MIN_HAND_DIAG:
                            if self.is_next(self.pts, diag):
                                ch = "next"; self.smoothing.clear()
                            elif self.is_backspace(self.pts):
                                ch = "Backspace"; self.smoothing.clear()
                            elif self.is_space(self.pts):
                                ch = " "; self.smoothing.clear()
                            elif self.is_fist_A(self.pts, diag):
                                ch = "A"; self.smoothing.clear()
                            else:
                                # Upscale small crops for clearer prediction at distance
                                h_c, w_c = crop.shape[:2]
                                if min(h_c, w_c) < 190:
                                    scale = 220 / max(1, min(h_c, w_c))
                                    new_w = max(220, int(w_c * scale))
                                    new_h = max(220, int(h_c * scale))
                                    crop_up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                                else:
                                    crop_up = crop
                                ch = self.predict_letter(crop_up, diag)

                        # draw skeleton preview centered to hand2 bbox
                        tmp = np.ones((400, 400, 3), dtype=np.uint8) * 255
                        pts = self.pts
                        hx, hy, hw, hh = hands2[0]['bbox']
                        scale = min(360 / hw, 360 / hh)
                        for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                                     (5,9),(9,10),(10,11),(11,12),
                                     (9,13),(13,14),(14,15),(15,16),
                                     (13,17),(17,18),(18,19),(19,20)]:
                            ax = int((pts[a][0] - hx) * scale) + 20
                            ay = int((pts[a][1] - hy) * scale) + 20
                            bx = int((pts[b][0] - hx) * scale) + 20
                            by = int((pts[b][1] - hy) * scale) + 20
                            cv2.line(tmp, (ax, ay), (bx, by), (0,255,0), 2)
                        for p in pts:
                            px = int((p[0] - hx) * scale) + 20
                            py = int((p[1] - hy) * scale) + 20
                            cv2.circle(tmp, (px, py), 2, (0,0,255), 1)
                        white = tmp

                self.current_symbol = ch or ""
                if ch:
                    self.update_sentence(ch)

                imgtk2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(white, cv2.COLOR_BGR2RGB)))
                self.panel2.config(image=imgtk2); self.panel2.imgtk = imgtk2
                self.panel3.config(text=self.current_symbol or "…")
                self.panel5.config(text=self.sentence)
                self.update_suggestions()

            self.root.after(LOOP_DELAY_MS, self.video_loop)
        except Exception:
            print(traceback.format_exc())
            self.root.after(40, self.video_loop)

    # ---------- Suggestions ----------
    def update_suggestions(self):
        word = self.sentence.split()[-1] if self.sentence.strip() else ""
        if word:
            sug = dct.suggest(word)[:4]
            arr = sug + [" "] * (4 - len(sug))
        else:
            arr = [" "] * 4
        self.suggest = arr
        for btn, txt in zip([self.b1, self.b2, self.b3, self.b4], arr):
            btn.config(text=txt)

    def apply_suggestion(self, w):
        if not w.strip():
            return
        parts = self.sentence.rstrip().split(" ")
        if parts:
            parts[-1] = w.upper()
            self.sentence = " ".join(parts) + " "

    def action1(self): self.apply_suggestion(self.suggest[0])
    def action2(self): self.apply_suggestion(self.suggest[1])
    def action3(self): self.apply_suggestion(self.suggest[2])
    def action4(self): self.apply_suggestion(self.suggest[3])

    # ---------- Misc ----------
    def speak_fun(self):
        self.speak_engine.say(self.sentence)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.sentence = " "
        self.prev_char = ""
        self.smoothing.clear()
        self.panel5.config(text=self.sentence)

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()


print("Starting Application...")
Application().root.mainloop()
