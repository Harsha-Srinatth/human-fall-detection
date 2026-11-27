"""Human Fall Detection web app.

The browser sends webcam / video frames to /api/frame. The server detects
people, runs the fall logic and creates ONE alert per fall event.
"""

import json
import os
import threading
import urllib.parse
import urllib.request
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import PersonDetector
from fall_monitor import FallMonitor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_DIR = os.getenv("ALERTS_DIR", os.path.join(BASE_DIR, "alerts"))
os.makedirs(ALERTS_DIR, exist_ok=True)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI(title="Human Fall Detection")
detector = PersonDetector()
lock = threading.Lock()
monitors: dict[str, FallMonitor] = {}
alerts: list[dict] = []  # newest first


def send_telegram(alert, image_path):
    """Optional: send the alert once to Telegram if env vars are set."""
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        text = f"EMERGENCY: {alert['message']}\nTime: {alert['time']}\nDate: {alert['date']}"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:  # never break detection because of a notification
        print("Telegram send failed:", e)


def save_alert(frame, detections, state):
    now = datetime.now()
    alert_id = now.strftime("%Y%m%d_%H%M%S_%f")
    for x1, y1, x2, y2, _ in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    image_path = os.path.join(ALERTS_DIR, f"{alert_id}.jpg")
    cv2.imwrite(image_path, frame)
    alert = {
        "id": alert_id,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d-%m-%Y"),
        "message": ("Person fallen and not moving" if state["alert_count"] == 1
                    else f"REMINDER #{state['alert_count'] - 1}: person still down for {state['down_for'] // 60} min {state['down_for'] % 60} s"),
        "image": f"/alerts/{alert_id}.jpg",
    }
    alerts.insert(0, alert)
    del alerts[50:]
    print("EMERGENCY ALERT:", json.dumps(alert))
    threading.Thread(target=send_telegram, args=(alert, image_path), daemon=True).start()
    return alert


@app.post("/api/frame")
async def process_frame(frame: UploadFile, session: str = Form(...)):
    data = np.frombuffer(await frame.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    detections = detector.detect(img)
    with lock:
        monitor = monitors.setdefault(session[:64], FallMonitor())
        state = monitor.update(img, detections)
        alert = save_alert(img, detections, state) if state["new_alert"] else None

    state["boxes"] = [[x1, y1, x2, y2, round(c, 2)] for x1, y1, x2, y2, c in detections]
    state["alert"] = alert
    return state


@app.post("/api/reset")
def reset(session: str = Form(...)):
    with lock:
        monitors.pop(session[:64], None)
    return {"ok": True}


@app.get("/api/alerts")
def get_alerts():
    return alerts


@app.get("/alerts/{name}")
def alert_image(name: str):
    path = os.path.join(ALERTS_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
