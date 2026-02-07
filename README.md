# SignFlow - ASL Video Call & Interpreter

SignFlow is a FastAPI + WebRTC web app that turns a browser video call into a live ASL interpreter. As of **February 6, 2026** the system recognizes ASL alphabet letters (A-Z) with helper gestures (space / next / backspace). A simplified gesture set for common phrases is on the near-term roadmap.

## What This Project Does
- Runs a browser-based call between two roles: `primary` (signer) and `secondary` (viewer).
- Captures the primary user's camera, extracts hand landmarks with cvzone/MediaPipe, draws a skeleton frame, and classifies it with `cnn8grps_rad1_model.h5` (TensorFlow 2.13).
- Smooths predictions, converts them into characters, and builds a sentence with gesture controls:
  - `next` commits the previous stable letter to the sentence.
  - `Backspace` removes the last character.
  - A relaxed closed-hand gesture inserts a space.
- Shares the current letter + sentence with the secondary user in real time over HTTP polling (with WebRTC media transport).
- Provides signup/login, PBKDF2 password hashing, session cookies, and default demo accounts.

## Roadmap (near future)
- Replace alphabet-only model with a simple gesture vocabulary (yes/no/hello/thanks/help, etc.).
- Lighter model export for edge devices; optional GPU build.
- WebSocket signaling path as default (HTTP polling already present).
- In-call accessibility additions (captions, audio read-out).

## Tech Stack
- FastAPI 0.103 + Starlette templates (Jinja2) and static assets.
- WebRTC media with TURN fallback (openrelay.metered.ca) and HTTP signaling.
- TensorFlow/Keras 2.13.1 + OpenCV + cvzone + MediaPipe for hand landmarking.
- SQLite (`signflow.db`) for users/sessions; PBKDF2 (sha256) for password hashing.

## Repository Layout
- `camera_feed.py` - main FastAPI app, auth, signaling endpoints, and ASL inference pipeline.
- `templates/` - Jinja2 views (`login.html`, `signup.html`, `in_call.html`).
- `static/css/style.css` - UI styling for auth and call surfaces.
- `cnn8grps_rad1_model.h5` - trained alphabet classifier (required at runtime).
- `signflow.db` - SQLite store created/updated on startup.
- `simple_interpreter.py` - standalone webcam interpreter demo.
- `requirements.txt` - pinned dependencies.

## Prerequisites
- Python 3.10-3.11 (TensorFlow 2.13 wheels target these versions).
- OS: Windows/macOS/Linux with a working camera and microphone.
- pip, and roughly 2 GB free disk for TensorFlow plus the model weights.

## Installation
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If TensorFlow fails to install on CPU-only machines, try `pip install tensorflow-cpu==2.13.1` (and keep `tensorflow-intel` pinned on Windows if available).

## Running the App
```bash
uvicorn camera_feed:app --reload --host 0.0.0.0 --port 8000
```
- Open `http://localhost:8000` in two browser sessions (or one incognito window).
- Sign in as:
  - `primary` / `primary123` (acts as signer and runs inference).
  - `secondary` / `secondary123` (viewer; receives text output).
  - Or create new accounts via Sign up; only the `primary` user triggers predictions.
- Allow camera/mic access. The primary user's hand will be sampled every ~200 ms.

## Using Gestures
- Letters: A-Z (model heuristics map hand landmarks to letters).
- Controls:
  - `next` gesture commits the last stable letter to the sentence builder.
  - `Backspace` gesture deletes the last character.
  - Closed-hand with pinky up -> space.
- Live results are available at `/live_result` and pushed to the secondary UI.

## API/Endpoint Reference
- `GET /` - login page.
- `GET /signup` - signup page.
- `GET /call` - authenticated call UI (requires session cookie).
- `POST /signup` - create user.
- `POST /login` - authenticate user; issues `session` cookie.
- `POST /logout` - clear session.
- `POST /predict` - accepts base64 JPEG, returns `{"letter": "...", "sentence": "..."}` (auth required; only runs for username `primary`).
- `POST /signal/send` / `GET /signal/recv` - HTTP signaling for WebRTC.
- `GET /live_result` - latest detected letter/sentence for polling.
- `WS /ws/signaling` - optional WebSocket signaling channel (not used by current UI).

## Troubleshooting
- If video is mirrored, set `mirror_input = True` in `camera_feed.py`.
- Delete `signflow.db` to reset users/sessions (a new DB is created on next start).
- Model file must stay in the project root or update the path in `camera_feed.py`.
- Performance: ensure good lighting; keep hand within ~120-250 px diagonal for confident predictions.

## Contributing / Next Steps
1) Swap in the upcoming simple-gesture model and update `/predict` mapping.
2) Add tests for auth/session flows and predict endpoint.
3) Tune confidence thresholds and smoothing window (`smoothing_window` in `camera_feed.py`) for latency vs. stability.

