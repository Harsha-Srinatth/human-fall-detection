# Human Fall Detection

Simple web app that detects a fallen person (YOLOv8n via OpenCV DNN) from your
webcam or an uploaded video and sends **one alert per fall**.

## How it works
1. Browser sends frames to `/api/frame`.
2. A person whose box is wider than tall is "possibly fallen".
3. If they stay still for `EMERGENCY_TIMEOUT_SEC` (default 10s) -> one alert
   (snapshot + time) is created.
4. No more alerts for that fall, even if the person lies there for 30 minutes.
   Alerts re-arm only after the person stands up for `RECOVER_SEC` (default 3s).

Optional Telegram message: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://localhost:8000

## Deploy
Railway: uses `railway.json` / `Procfile` (start: `uvicorn app:app --host 0.0.0.0 --port $PORT`).
