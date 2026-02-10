import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import base64
import hashlib
import hmac
import math
import sqlite3
import secrets
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model

app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")

# TURN / STUN config from environment (for deployment flexibility)
TURN_URLS = os.getenv(
    "TURN_URLS",
    "turn:openrelay.metered.ca:80,turn:openrelay.metered.ca:443,turn:openrelay.metered.ca:443?transport=tcp",
)
TURN_USER = os.getenv("TURN_USER", "openrelayproject")
TURN_PASS = os.getenv("TURN_PASS", "openrelayproject")
STUN_URLS = os.getenv(
    "STUN_URLS",
    "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
)
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
# Keep detector defaults; the user-supplied logic expects raw landmarks
hd = HandDetector(maxHands=1, detectionCon=0.50, minTrackCon=0.35)
hd2 = HandDetector(maxHands=1, detectionCon=0.50, minTrackCon=0.35)
# Match the margin used in the user's reference code and mirror like the desktop app
offset = 29
mirror_input = True
# Keep the web path close to final_pred.py for speed/behavior parity.
ENABLE_FRAME_NORMALIZATION = False
UPSCALE_SMALL_CROPS = False

# Auth / DB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "signflow.db")
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours
session_user_cache: Dict[str, Tuple[int, str, int]] = {}
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

def _issue_session(user_id: int, username: Optional[str] = None) -> str:
    """Create a session token for the given user and persist it."""
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at)
    )
    if username is None:
        cur.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        username = row[0] if row else ""
    conn.commit()
    conn.close()
    if username:
        session_user_cache[token] = (user_id, username, expires_at)
    return token

def _drop_session(token: str, delete_db: bool = False):
    if not token:
        return
    session_user_cache.pop(token, None)
    if not delete_db:
        return
    conn = _db()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()

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
    now = int(time.time())
    cached = session_user_cache.get(token)
    if cached:
        cached_user_id, cached_username, cached_expires = cached
        if cached_expires >= now:
            return (cached_user_id, cached_username)
        session_user_cache.pop(token, None)

    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    user_id, expires_at = row
    if expires_at < now:
        cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    cur.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    if user:
        session_user_cache[token] = (user[0], user[1], expires_at)
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

# Global state for final_pred-style sentence assembly
smoothing_window = 3

# Global text state
sentence = " "
prev_char = ""
count = -1
ten_prev_char = [" "] * 10

# Shared last detection result for secondary display
last_result = {"letter": "", "sentence": "", "ts": 0}

def _reset_state():
    """Clear prediction buffers so stale letters don't persist across restarts/logouts."""
    global sentence, prev_char, count, ten_prev_char, last_result
    sentence = " "
    prev_char = ""
    count = -1
    ten_prev_char = [" "] * 10
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

def _is_letter_symbol(sym) -> bool:
    return isinstance(sym, str) and len(sym) == 1 and sym.isalpha()

def _hand_diag(pts) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

def _is_next_gesture_strict(pts) -> bool:
    """
    Tuned "next" gesture: permissive enough to trigger reliably, including
    both left and right hands, while still requiring a clear folded-finger pose.
    """
    diag = _hand_diag(pts)
    if diag < 50:
        return False

    fold_margin = max(3, diag * 0.02)
    folded_votes = 0
    folded_votes += int(pts[8][1] > pts[6][1] + fold_margin)
    folded_votes += int(pts[12][1] > pts[10][1] + fold_margin)
    folded_votes += int(pts[16][1] > pts[14][1] + fold_margin)
    folded_votes += int(pts[20][1] > pts[18][1] + fold_margin)

    palm_x_min = min(pts[5][0], pts[9][0], pts[13][0], pts[17][0])
    palm_x_max = max(pts[5][0], pts[9][0], pts[13][0], pts[17][0])
    thumb_tol = max(8, int(diag * 0.05))
    thumb_inside_palm = (palm_x_min - thumb_tol) <= pts[4][0] <= (palm_x_max + thumb_tol)
    thumb_not_above = pts[4][1] > pts[2][1] - 2

    return folded_votes >= 3 and thumb_inside_palm and thumb_not_above

def _is_backspace_gesture(pts) -> bool:
    """
    Direct geometric backspace gesture check (independent of model class label).
    """
    diag = _hand_diag(pts)
    if diag < 70:
        return False
    return (
        pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0] and
        pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1] and
        pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]
    )

def _update_sentence(ch):
    global sentence, prev_char, count, ten_prev_char
    # Keep sentence behavior aligned with the desktop final_pred.py flow.
    if ch == "next" and prev_char != "next":
        prior = ten_prev_char[(count - 2) % 10]
        if prior != "next":
            if prior == "Backspace":
                sentence = sentence[:-1]
            elif _is_letter_symbol(prior):
                sentence += prior
        else:
            latest = ten_prev_char[(count - 0) % 10]
            if _is_letter_symbol(latest):
                sentence += latest

    if ch == "  " and prev_char != "  ":
        sentence += "  "

    prev_char = ch
    count += 1
    ten_prev_char[count % 10] = ch

def predict(test_image, pts):
    global last_result

    def distance(x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    # Model inference on 400x400 input; skip resize when already at target size.
    if test_image.shape[0] != 400 or test_image.shape[1] != 400:
        white = cv2.resize(test_image, (400, 400))
    else:
        white = test_image
    white = white.reshape(1, 400, 400, 3)
    prob = np.array(model(white, training=False)[0], dtype='float32')

    ch1 = int(np.argmax(prob))
    prob[ch1] = 0
    ch2 = int(np.argmax(prob))
    prob[ch2] = 0
    ch3 = int(np.argmax(prob))
    prob[ch3] = 0

    pl = [ch1, ch2]

    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

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

    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 1

    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
                (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 7

    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

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

    if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = " "

    # Keep "next" rule aligned with final_pred.py behavior.
    if ch1 in ['E', 'Y', 'B']:
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "next"

    if _is_backspace_gesture(pts):
        ch1 = 'Backspace'

    # Do not emit raw class-group integers (0..7) to UI/sentence.
    if _is_letter_symbol(ch1):
        ch1 = ch1.upper()
    elif ch1 == " ":
        ch1 = "  "
    elif ch1 not in ("next", "Backspace", "  "):
        ch1 = ""

    # Match desktop interpreter print behavior (ten_prev_char + next gesture).
    _update_sentence(ch1)

    last_result = {"letter": ch1, "sentence": sentence, "ts": int(time.time() * 1000)}
    return ch1

@app.post("/predict")
async def predict_endpoint(request: Request):
    token = request.cookies.get("session")
    user = _get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    image_bytes = b""
    content_type = request.headers.get("content-type", "").lower()
    if "image/jpeg" in content_type or "application/octet-stream" in content_type:
        image_bytes = await request.body()
    else:
        # Backward-compatible JSON path for older clients.
        try:
            data = await request.json()
        except Exception:
            data = {}
        image_data = data.get("image", "")
        if isinstance(image_data, str) and image_data:
            encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
            image_bytes = base64.b64decode(encoded)

    if not image_bytes:
        return {"letter": "", "sentence": sentence}

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"letter": "", "sentence": sentence}

    if ENABLE_FRAME_NORMALIZATION:
        frame = _normalize_frame(frame)

    cv2image = cv2.flip(frame, 1) if mirror_input else frame
    hands, _ = hd.findHands(cv2image, draw=False, flipType=True)
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
                if UPSCALE_SMALL_CROPS:
                    h_crop, w_crop = image.shape[:2]
                    if min(h_crop, w_crop) < 160:
                        scale = 180 / max(1, min(h_crop, w_crop))
                        new_w = max(200, int(w_crop * scale))
                        new_h = max(200, int(h_crop * scale))
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                hands2, _ = hd2.findHands(image, draw=False, flipType=True)
                if hands2:
                    hand2 = hands2[0]
                    if hand2:
                        pts = hand2['lmList']
                        _, _, w2, h2 = hand2['bbox']
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
                        letter = predict(white, pts)
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
    # Map arbitrary usernames to the two signaling roles expected by the client
    role = "primary" if user[1].lower() == "primary" else "secondary"
    _reset_state()
    return templates.TemplateResponse(
        "in_call.html",
        {
            "request": request,
            "username": user[1],
            "role": role,
            "turn_urls": TURN_URLS,
            "turn_user": TURN_USER,
            "turn_pass": TURN_PASS,
            "stun_urls": STUN_URLS,
        },
    )

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
    conn.commit()
    conn.close()
    token = _issue_session(user_id, username)
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
        conn.close()
        if token:
            _drop_session(token, delete_db=True)
            response.delete_cookie("session")
        raise HTTPException(status_code=401, detail="User not found. Please create an account.")
    user_id, pw_hash, salt = row
    if not _verify_password(password, salt, pw_hash):
        token = request.cookies.get("session")
        conn.close()
        if token:
            _drop_session(token, delete_db=True)
            response.delete_cookie("session")
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    conn.close()
    token = _issue_session(user_id, username)

    response.set_cookie("session", token, httponly=True, samesite="lax")
    _reset_state()
    return {"ok": True, "created": False}

@app.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        _drop_session(token, delete_db=True)
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
