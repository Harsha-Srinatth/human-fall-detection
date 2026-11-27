# Human Fall Detection

Simple web app that detects a fallen person (YOLOv8n via OpenCV DNN) from your
webcam or an uploaded video and sends **one alert per fall**.

## How it works
Browser sends frames to `/api/frame`. YOLOv8n-pose (ONNX, OpenCV DNN, no PyTorch) finds people and
17 body keypoints. Rules are taken from existing open-source projects and commercial devices:

| Rule | Value | Source |
|---|---|---|
| Lying: torso angle from vertical | > 60 deg (upright < 45 deg) | taufeeque9/HumanFallDetection uses pi/4; papers use 60-120 deg |
| Lying: box width / height | >= 1.2 | cwlroda/falldetection_openpifpaf |
| Lying: keypoints spread horizontally | x-range > y-range | MoveNet rule-based papers |
| Fall (not just lying down) | upright < 1.5 s before lying, or hips drop >= 40% of height | cwlroda: centroid moves 0.5 x diagonal in ~1 s |
| Smoothing | lying in >= 50% of frames over 1 s | taufeeque9 needs 9 consecutive frames |
| Alert | fall + no movement for 10 s (`STILL_ALERT_SEC`) | Apple Watch: hard fall + ~1 min no movement |
| Alert | on the floor 30 s even if moving (`LONG_LIE_SEC`) | "can't get up" case |
| Reminder | every 120 s while still down (`REMIND_EVERY_SEC`) | escalation without spamming |
| Reset | upright for 3 s (`RECOVER_SEC`); detection gaps < 2 s ignored | cwlroda drops a track after 2 x fps frames |

Sitting close to the camera (wide box, no visible hips) is **not** a fall. The old box-shape-only
logic raised false alerts in exactly that case.

Uploaded videos are analysed in video time (0.2 s steps), so results don't depend on server speed.
After the video ends the last frame is held, as if the person stays on the floor.

Optional Telegram message: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://localhost:8000

## Deploy
Render (free): uses `render.yaml`. Railway: uses `railway.json`. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
