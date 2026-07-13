"""Flask web server for the Violence Detection System dashboard."""

import os
import sqlite3
from flask import Flask, render_template, Response, jsonify, request
from engine import DetectionEngine
from utils import DEFAULT_DB_PATH

app = Flask(__name__)

# Single global engine instance
engine = DetectionEngine(
    camera_index=int(os.getenv("CAMERA_INDEX", "0")),
    model_name=os.getenv("MODEL_NAME", "yolov8m.pt"),
    conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.4")),
)
engine.start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        engine.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": engine.status,
        "fps": round(engine.display_fps, 1),
        "error": engine.error_msg,
        "total_detections": engine.total_detections,
        "total_concealed": engine.total_concealed,
        "threat_counts": engine.threat_counts,
    })


@app.route("/api/logs")
def api_logs():
    """Return the 50 most recent detection events from SQLite."""
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """SELECT id, timestamp, object, confidence, image_path, clip_path
               FROM detections ORDER BY id DESC LIMIT 50"""
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/control/stop")
def api_stop():
    engine.stop()
    return jsonify({"ok": True, "status": "stopped"})


@app.route("/api/control/start")
def api_start():
    engine.start()
    return jsonify({"ok": True, "status": "starting"})


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    try:
        frame_bytes = request.data
        if not frame_bytes:
            return "No image data", 400

        annotated_bytes, has_weapon = engine.process_client_frame(frame_bytes)
        if not annotated_bytes:
            return "Failed to process image", 500

        return Response(annotated_bytes, mimetype="image/jpeg")
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
