# Human Fall Detection

Real-time human fall detection using YOLOv8, MediaPipe pose estimation, and motion analysis.

## Project Structure

```
HUMAN-FALL_DETECTION/
├── models/          # YOLOv8 model weights
├── videos/          # Input test videos
├── images/          # Input images
├── alerts/          # Saved fall alert snapshots
├── output/          # Processed output videos
├── utils/           # Core detection modules
├── main.py          # Entry point
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the YOLOv8 model at `models/yolov8n.pt` (downloaded automatically on first run if missing).

3. Add test videos to `videos/` (e.g. `test1.mp4`, `test2.mp4`).

## Usage

```bash
python main.py --video test1.mp4
python main.py --video test2.mp4 --output result.mp4
```

Press `q` to quit the preview window.

## How It Works

1. **Detection** — YOLOv8 detects humans in each frame.
2. **Pose** — MediaPipe estimates body angle; angles > 60° from vertical indicate a fall.
3. **Motion** — Sudden downward motion is analyzed to reduce false positives.
4. **Alert** — On fall detection, a snapshot is saved to `alerts/`.
