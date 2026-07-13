---
title: VigilOS
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: gradio
app_port: 7860
pinned: false
---

## Violence Detection System (YOLOv8)

A modular Python 3.10+ project that detects potentially violent objects (knife, gun, bottle, axe) using YOLOv8, sends WhatsApp alerts (Twilio or pywhatkit), saves a snapshot and a 10-second clip (5s before and 5s after), and logs detections to CSV and SQLite.

### Features

- **YOLOv8 detection** for target classes with configurable confidence threshold
- **Apple Silicon acceleration** — auto-detects MPS (Metal) GPU, falls back to CUDA or CPU
- **Face occlusion detection** — flags persons whose face is not visible (mask, hood, turned away) using MediaPipe
- **Smart frame skipping** — runs inference every 3rd frame, reuses cached bounding boxes for smooth display
- **WhatsApp alert** via Twilio (recommended, headless) or pywhatkit (requires WhatsApp Web)
- Saves snapshot image and 10s video clip (pre/post)
- Logs to `logs/detections.csv` and `logs/detections.db`
- Modular structure for future action/audio/IoT extensions

### Project Structure

- `main.py`: runs capture, detection, alerting, logging
- `detector.py`: YOLOv8 wrapper with MPS/CUDA/CPU auto-detection and COCO class filtering
- `face_checker.py`: MediaPipe-based face occlusion / concealment checker
- `alert.py`: WhatsApp via Twilio/pywhatkit
- `utils.py`: helpers (timestamps, saving, CSV/SQLite, ring buffer)
- `requirements.txt`: dependencies

### Setup

1. Python 3.10+
2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Optional: Configure environment for alerts. Create `.env` in project root (or use system environment variables). See `.env.example` below.

### .env example

```
# Twilio (recommended)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_WHATSAPP_TO=whatsapp:+91XXXXXXXXXX

# pywhatkit (fallback)
WHATSAPP_PHONE=+91XXXXXXXXXX

# Optional: if you host media files and want to include them as media_url
MEDIA_URL_PREFIX=https://your.cdn.example/media
```

### Usage

- Default camera index 0, YOLOv8 nano model, 50% confidence threshold

```bash
python main.py
```

- Stop with 'q' in the display window

### Performance Tuning

| Setting              | Default    | How to change                          | Effect                          |
| -------------------- | ---------- | -------------------------------------- | ------------------------------- |
| Inference resolution | 416        | `IMG_SIZE=320 python main.py`          | Smaller = faster, less accurate |
| Model                | yolov8n.pt | `MODEL_NAME=yolov8m.pt python main.py` | Larger = more accurate, slower  |
| Confidence           | 0.5        | `CONF_THRESHOLD=0.7 python main.py`    | Higher = fewer false positives  |
| Debug detections     | off        | `DEBUG_DETECTIONS=1 python main.py`    | Print raw YOLO output           |

**Device auto-detection**: The system automatically uses Apple MPS (Metal) on Apple Silicon Macs, CUDA on NVIDIA GPUs, or CPU as fallback. No configuration needed.

**CoreML export** (optional, macOS-only, maximum speed):

```bash
yolo export model=yolov8n.pt format=coreml
```

Then load the `.mlpackage` for native Neural Engine inference.

### Notes

- On macOS, grant camera and screen control permissions to Terminal/IDE if using pywhatkit.
- Twilio Sandbox for WhatsApp requires joining the sandbox from your phone.
- Local file attachments in WhatsApp require a public URL; otherwise, the message will be text-only. The snapshot/clip are always saved locally in `media/`.
- Face occlusion detection requires `mediapipe`. If not installed, the feature is silently disabled.

### Future expansion

- Add action recognition models for fights/punching/kicking
- Integrate audio event detection (screams, gunshots)
- Add GPIO/IoT triggers for alarms/locks
- Fine-tune YOLOv8 on weapon-specific datasets (Roboflow) for better knife/gun accuracy
