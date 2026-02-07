import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math
import sqlite3
import secrets
import hashlib
import hmac
import time
from typing import Dict, Optional
from string import ascii_uppercase

app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and initialize detectors
model = load_model('cnn8grps_rad1_model.h5')
# Lower thresholds slightly more to keep tracking stable at distance and close-up
hd = HandDetector(maxHands=1, detectionCon=0.50, minTrackCon=0.35)
hd2 = HandDetector(maxHands=1, detectionCon=0.50, minTrackCon=0.35)
# Slightly adjust crop margin to keep more of the hand visible
offset = 45
mirror_input = False  # keep original orientation for model consistency

# Auth / DB
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signflow.db")
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours

def _db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _init_db():
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    # Simple migration: add email column if DB was created before email support
    cur.execute("PRAGMA table_info(users)")
    cols = [row[1] for row in cur.fetchall()]
    if "email" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return dk.hex()

def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    test_hash = _hash_password(password, salt)
    return hmac.compare_digest(test_hash, stored_hash)

DEFAULT_ACCOUNTS = [
    {"username": "primary", "email": "primary@example.com", "password": "primary123"},
    {"username": "secondary", "email": "secondary@example.com", "password": "secondary123"},
]

def _seed_default_users():
    """Ensure the primary/secondary hardcoded users exist."""
    conn = _db()
    cur = conn.cursor()
    for acct in DEFAULT_ACCOUNTS:
        cur.execute("SELECT id FROM users WHERE username = ?", (acct["username"],))
        if cur.fetchone():
            continue
        salt = secrets.token_hex(16)
        pw_hash = _hash_password(acct["password"], salt)
        cur.execute(
            "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (acct["username"], acct["email"], pw_hash, salt, int(time.time())),
        )
    conn.commit()
    conn.close()

def _is_password_unique(password: str) -> bool:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users")
    rows = cur.fetchall()
    conn.close()
    for pw_hash, salt in rows:
        if _verify_password(password, salt, pw_hash):
            return False
    return True

def _get_user_from_session(token: str):
    if not token:
        return None
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    user_id, expires_at = row
    if expires_at < int(time.time()):
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    cur.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user

_init_db()
_seed_default_users()

# Simple in-memory signaling slots for a single primary-secondary pair
signals: Dict[str, Optional[WebSocket]] = {"primary": None, "secondary": None}
pending: Dict[str, list] = {"primary": [], "secondary": []}  # messages queued for this role
http_signal_queue: Dict[str, list] = {"primary": [], "secondary": []}  # for HTTP polling signaling


def _get_peer(role: str) -> Optional[WebSocket]:
    other = "secondary" if role == "primary" else "primary"
    return signals.get(other)


@app.websocket("/ws/signaling")
async def websocket_signaling(ws: WebSocket):
    await ws.accept()
    try:
        # Expect first message to declare role
        role_msg = await ws.receive_text()
        role = role_msg.strip().lower()
        if role not in ("primary", "secondary"):
            await ws.send_text("error:invalid-role")
            await ws.close(code=1008)
            return
        # Replace any existing connection for this role
        if signals[role]:
            try:
                await signals[role].close(code=1011)
            except Exception:
                pass
        signals[role] = ws
        await ws.send_text("ack:{}".format(role))

        # Flush any pending messages that were queued while peer was absent
        # Deliver any messages queued for this role
        for msg in pending[role]:
            try:
                await ws.send_text(msg)
            except Exception:
                pass
        pending[role].clear()

        # Relay loop
        while True:
            data = await ws.receive_text()
            peer = _get_peer(role)
            if peer:
                try:
                    await peer.send_text(data)
                except Exception:
                    pass
            else:
                # queue for the peer to receive when it connects
                other = "secondary" if role == "primary" else "primary"
                pending[other].append(data)
    except WebSocketDisconnect:
        pass
    finally:
        # cleanup on disconnect
        for r, sock in signals.items():
            if sock is ws:
                signals[r] = None


# ----------------- HTTP polling signaling (fallback when websockets not available) -----------------
from fastapi import Body

def _other(role: str) -> str:
    return "secondary" if role == "primary" else "primary"

@app.post("/signal/send")
async def signal_send(payload: dict = Body(...)):
    role = payload.get("role", "").lower()
    data = payload.get("data")
    if role not in ("primary", "secondary"):
        raise HTTPException(status_code=400, detail="invalid role")
    http_signal_queue[_other(role)].append(data)
    return {"ok": True}

@app.get("/signal/recv")
async def signal_recv(role: str):
    role = role.lower()
    if role not in ("primary", "secondary"):
        raise HTTPException(status_code=400, detail="invalid role")
    msgs = http_signal_queue[role][:]
    http_signal_queue[role].clear()
    return {"messages": msgs}

# Global variables for prediction smoothing
from collections import deque

previous_predictions = deque(maxlen=5)
smoothing_window = 3  # balanced stability
min_confidence = 0.35
# Debounce counter for "next" gesture to avoid accidental triggers
next_streak = 0
# Letter stability to avoid single-frame flips
stable_letter = ""
stable_count = 0

# Global text state
sentence = ""
prev_char = ""
count = -1
ten_prev_char = [" "] * 10

# Shared last detection result for secondary display
last_result = {"letter": "", "sentence": "", "ts": 0}

def _reset_state():
    """Clear prediction buffers so stale letters don't persist across restarts/logouts."""
    global sentence, prev_char, count, ten_prev_char, previous_predictions, next_streak, stable_letter, stable_count, last_result
    sentence = ""
    prev_char = ""
    count = -1
    ten_prev_char = [" "] * 10
    previous_predictions.clear()
    next_streak = 0
    stable_letter = ""
    stable_count = 0
    last_result = {"letter": "", "sentence": "", "ts": int(time.time() * 1000)}

_reset_state()

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def _adjust_gamma(image, gamma: float):
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    return cv2.LUT(image, table.astype("uint8"))

def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Lighting normalization for dark/bright backgrounds to stabilize detection."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        norm = frame

    mean = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).mean()
    if mean < 90:
        norm = _adjust_gamma(norm, 0.85)  # brighten dark scenes
    elif mean > 170:
        norm = _adjust_gamma(norm, 1.15)  # tame bright scenes
    return norm

def _is_next_gesture(pts):
    """
    Detect a deliberate closed-fist "next" gesture.
    Tightens conditions so the ASL letter 'A' (thumb outside) doesn't trigger it.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    # Require a reasonably large hand in frame to reduce far-distance misfires
    if diag < 70:
        return False

    fold_margin = max(12, diag * 0.10)

    # All fingertips must be clearly below their MCP joints (folded fingers)
    folded = (
        pts[8][1] > pts[5][1] + fold_margin and
        pts[12][1] > pts[9][1] + fold_margin and
        pts[16][1] > pts[13][1] + fold_margin and
        pts[20][1] > pts[17][1] + fold_margin
    )

    # Thumb tip should sit inside the palm box (between index and ring MCP x-range) and below its MCP
    palm_x_min = min(pts[5][0], pts[9][0], pts[13][0])
    palm_x_max = max(pts[5][0], pts[9][0], pts[13][0])
    thumb_inside = (palm_x_min + 6) <= pts[4][0] <= (palm_x_max - 6)
    thumb_below_mcp = pts[4][1] > pts[2][1] + 4

    return folded and thumb_inside and thumb_below_mcp

def _update_sentence(ch):
    global sentence, prev_char, count, ten_prev_char
    count += 1
    ten_prev_char[count % 10] = ch

    if ch == "next" and prev_char != "next":
        # commit previous stable character
        prior = ten_prev_char[(count - 2) % 10]
        if prior == "Backspace":
            sentence = sentence[:-1]
        elif prior not in ["next", "Backspace", ""]:
            sentence += prior
    elif ch == " " and prev_char != " ":
        sentence += " "
    elif ch == "Backspace" and prev_char != "Backspace":
        sentence = sentence[:-1]

    prev_char = ch

def predict(test_image, pts):
    global previous_predictions, last_result, next_streak, stable_letter, stable_count
    # Reject frames where the hand is too small (user is too far from camera)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span_w = max(xs) - min(xs)
    span_h = max(ys) - min(ys)
    diag = math.sqrt(span_w ** 2 + span_h ** 2)
    if diag < 55:
        return ""  # hand too far / not in frame enough for a reliable prediction

    # Fast path with debounce: detect "next" gesture directly from landmarks
    if _is_next_gesture(pts):
        next_streak += 1
        if next_streak >= 2:  # require two consecutive frames
            ch1 = "next"
            previous_predictions.clear()
            _update_sentence(ch1)
            last_result = {"letter": ch1, "sentence": sentence, "ts": int(time.time() * 1000)}
            return ch1
    else:
        next_streak = 0

    # Ensure model input size
    white = cv2.resize(test_image, (400, 400))
    white = white.astype(np.float32)
    white = white.reshape(1, 400, 400, 3)

    prob = np.array(model.predict(white, verbose=0)[0], dtype='float32')

    # Get confidence scores with top-2 margin filtering
    max_prob = float(np.max(prob))
    confidence_threshold = 0.32  # Minimum confidence for prediction
    size_score = min(1.0, diag / 170.0)  # boost confidence if the hand is large/close
    quality = max_prob * size_score

    top1 = int(np.argmax(prob))
    prob[top1] = 0.0
    top2 = float(np.max(prob))
    margin = max_prob - top2

    if max_prob < confidence_threshold or quality < 0.22 or margin < 0.12:
        return ""  # Low confidence or low quality, don't predict

    ch1 = top1
    ch2 = int(np.argmax(prob))
    prob[ch2] = 0
    ch3 = int(np.argmax(prob))

    pl = [ch1, ch2]

    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
        (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # -------------------------condn for 8 groups  ends

    # -------------------------condn for subgroups  starts
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

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = 'Y'
        else:
            ch1 = 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = 'Z'
            else:
                ch1 = 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'

    # Apply prediction smoothing
    previous_predictions.append(ch1)

    # Return most common prediction in the window, but only if confidence is high
    if max_prob >= min_confidence and previous_predictions:
        from collections import Counter
        most_common = Counter(previous_predictions).most_common(1)[0][0]
        ch1 = most_common
    elif max_prob >= min_confidence:
        ch1 = ch1
    else:
        ch1 = ""  # Low confidence, no prediction

    # Suppress spurious 'J' unless confidence and hand size are sufficient
    if ch1 == 'J':
        if quality < 0.33 or diag < 110:
            ch1 = ""

    # Require short stability for letter outputs (not for controls)
    if ch1 in ascii_uppercase:
        if ch1 == stable_letter:
            stable_count += 1
        else:
            stable_letter = ch1
            stable_count = 1
        if stable_count < 2:
            return ""
    else:
        stable_letter = ""
        stable_count = 0

    # Special gesture controls
    # Space: mostly closed hand with pinky up
    if ch1 in ['B', 'E', 'S', 'X', 'Y']:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and
                pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = " "

    # Next: thumb tucked, all fingers folded
    if ch1 in ['E', 'Y', 'B']:
        if (pts[4][0] < pts[5][0] and
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and
                pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "next"

    # Backspace: open palm facing camera (all tips above base)
    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and \
       (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and \
       (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
        ch1 = "Backspace"

    _update_sentence(ch1)
    # update shared result
    last_result = {"letter": ch1, "sentence": sentence, "ts": int(time.time() * 1000)}
    return ch1

@app.post("/predict")
async def predict_endpoint(data: dict, request: Request):
    token = request.cookies.get("session")
    user = _get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    username = user[1]
    image_data = data['image']
    # Remove the data URL prefix
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"letter": ""}
    # Normalize lighting to handle dark/bright backgrounds
    frame = _normalize_frame(frame)
    # Flip to correct inverted camera if needed
    cv2image = cv2.flip(frame, 1) if mirror_input else frame
    hands, _ = hd.findHands(cv2image, draw=False, flipType=False)
    letter = ""
    if hands:
        hand = hands[0]
        if hand:
            x, y, w, h = hand['bbox']
            h_img, w_img = cv2image.shape[:2]
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, w_img)
            y2 = min(y + h + offset, h_img)
            if x2 > x1 and y2 > y1:
                image = cv2image[y1:y2, x1:x2]
                # Upscale small crops to help detection when the signer is farther away
                h_crop, w_crop = image.shape[:2]
                if min(h_crop, w_crop) < 160:
                    scale = 180 / max(1, min(h_crop, w_crop))
                    new_w = max(200, int(w_crop * scale))
                    new_h = max(200, int(h_crop * scale))
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                hands2, _ = hd2.findHands(image, draw=False, flipType=False)
                if hands2:
                    hand2 = hands2[0]
                    if hand2:
                        pts = hand2['lmList']
                        _, _, w2, h2 = hand2['bbox']
                        # Draw skeleton on white
                        os = ((400 - w2) // 2) - 15
                        os1 = ((400 - h2) // 2) - 15
                        for t in range(0, 4, 1):
                            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(5, 8, 1):
                            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(9, 12, 1):
                            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(13, 16, 1):
                            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(17, 20, 1):
                            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                        for i in range(21):
                            cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)
                        res = white
                        # Prepare model input from the original cropped hand (better accuracy than skeleton only)
                        image_for_model = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
                        # Only run prediction/update if this is the primary user
                        if username.lower() == "primary":
                            letter = predict(image_for_model, pts)
                        else:
                            letter = ""
                        print("Recognized letter:", letter)
    return {"letter": letter, "sentence": sentence}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Clear any existing session cookie on the login screen to avoid stale logins
    _reset_state()
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.delete_cookie("session")
    return resp

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/tips", response_class=HTMLResponse)
async def tips_page(request: Request):
    return templates.TemplateResponse("tips.html", {"request": request})

@app.get("/call", response_class=HTMLResponse)
async def call(request: Request):
    token = request.cookies.get("session")
    user = _get_user_from_session(token)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    _reset_state()
    return templates.TemplateResponse("in_call.html", {"request": request, "username": user[1]})

@app.post("/signup")
async def signup(data: dict, response: Response):
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists.")
    if email:
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Email already exists.")
    if not _is_password_unique(password):
        conn.close()
        raise HTTPException(status_code=400, detail="Password already in use. Choose a different password.")

    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    cur.execute(
        "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, email or None, pw_hash, salt, int(time.time()))
    )
    user_id = cur.lastrowid
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at)
    )
    conn.commit()
    conn.close()
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return {"ok": True, "user_id": user_id}

@app.post("/login")
async def login(data: dict, request: Request, response: Response):
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required.")

    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        # Clear any existing session if present
        token = request.cookies.get("session")
        if token:
            cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            response.delete_cookie("session")
        conn.close()
        raise HTTPException(status_code=401, detail="User not found. Please create an account.")
    user_id, pw_hash, salt = row
    if not _verify_password(password, salt, pw_hash):
        token = request.cookies.get("session")
        if token:
            cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            response.delete_cookie("session")
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at)
    )
    conn.commit()
    conn.close()

    response.set_cookie("session", token, httponly=True, samesite="lax")
    _reset_state()
    return {"ok": True, "created": False}

@app.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        conn = _db()
        cur = conn.cursor()
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    response.delete_cookie("session")
    _reset_state()
    return {"ok": True}

@app.post("/reset_state")
async def reset_state():
    _reset_state()
    return {"ok": True}


@app.get("/live_result")
async def live_result(role: str = ""):
    # Secondary polls this to display current letter/sentence
    return last_result
