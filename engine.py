"""Detection engine — headless version of main.py.

Instead of calling cv2.imshow, this exposes a `generate_frames()` generator
that yields JPEG-encoded bytes suitable for MJPEG streaming over HTTP.
All alerting, logging and clip-saving logic is unchanged.
"""

import os
import time
import threading
from typing import Optional, List, Tuple, Generator

import cv2
import numpy as np

from detector import YoloViolenceDetector
from face_checker import FaceOcclusionChecker
from alert import AlertSender
from utils import (
    ensure_directories,
    init_csv,
    init_db,
    append_csv,
    insert_db,
    save_frame_snapshot,
    timestamp_now,
    VideoRingBuffer,
    DEFAULT_MEDIA_DIR,
)


def _save_post_event_clip(cap: cv2.VideoCapture, seconds: int, fps: int, frame_size, out_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    for _ in range(int(seconds * fps)):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    return out_path


class DetectionEngine:
    """Runs YOLO + face-occlusion detection in a background thread and
    exposes the latest annotated frame for MJPEG streaming."""

    def __init__(
        self,
        camera_index: int = 0,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        pre_seconds: int = 5,
        post_seconds: int = 5,
        frame_skip: int = 3,
    ):
        self.camera_index = camera_index
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.frame_skip = frame_skip

        # Shared state (written by bg thread, read by Flask thread)
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None  # JPEG bytes
        self._running = False
        self._status = "stopped"   # "starting" | "running" | "stopped" | "error"
        self._error_msg = ""
        self._display_fps = 0.0
        self._thread: Optional[threading.Thread] = None

        # Stats surfaced to the web dashboard
        self.total_detections = 0
        self.total_concealed = 0
        self.threat_counts: dict = {}

        # Lazy initializers for sharing between local loop and client API
        self._detector = None
        self._face_checker = None
        self._alerter = None
        self._last_alert_time = None
        self.client_mode = os.getenv("CLIENT_MODE", "1") == "1"

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YoloViolenceDetector(model_name=self.model_name, confidence_threshold=self.conf_threshold)
        return self._detector

    @property
    def face_checker(self):
        if self._face_checker is None:
            self._face_checker = FaceOcclusionChecker()
        return self._face_checker

    @property
    def alerter(self):
        if self._alerter is None:
            self._alerter = AlertSender()
        return self._alerter

    # ------------------------------------------------------------------ #
    #  Public control API                                                  #
    # ------------------------------------------------------------------ #

    def start(self):
        if self._running:
            return
        self._running = True
        if self.client_mode:
            self._status = "running"
            print("[Engine] Started in Client-Side Streaming Mode (no server webcam thread).")
            return
        self._status = "starting"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._status = "stopped"

    @property
    def status(self) -> str:
        return self._status

    @property
    def display_fps(self) -> float:
        return self._display_fps

    @property
    def error_msg(self) -> str:
        return self._error_msg

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    # ------------------------------------------------------------------ #
    #  MJPEG generator (called by Flask route)                            #
    # ------------------------------------------------------------------ #

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG chunks for the /video_feed route."""
        while self._running:
            frame = self.get_jpeg_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps cap on the stream side

    # ------------------------------------------------------------------ #
    #  Background detection loop                                          #
    # ------------------------------------------------------------------ #

    def _loop(self):
        ensure_directories()
        init_csv()
        init_db()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._status = "error"
            self._error_msg = "Cannot open camera. Check permissions."
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(2)  # macOS camera warm-up

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        frame_size = (width, height)

        model_name = self.model_name or os.getenv("MODEL_NAME", "yolov8n.pt")
        conf_threshold = self.conf_threshold if self.conf_threshold is not None \
            else float(os.getenv("CONF_THRESHOLD", 0.5))

        detector     = self.detector
        face_checker = self.face_checker
        alerter      = self.alerter
        buffer       = VideoRingBuffer(max_seconds=self.pre_seconds, fps=int(fps), frame_size=frame_size)

        self._status = "running"

        self._last_alert_time = None
        cooldown_seconds = 120  # 2 minutes — saves Twilio credits
        cached_threats: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        cached_concealed: List[Tuple[int, int, int, int]] = []

        fps_start = time.time()
        fps_frame_count = 0
        frame_count = 0
        consecutive_failures = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        self._status = "error"
                        self._error_msg = "Camera lost — too many read failures."
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0
                frame_count += 1
                fps_frame_count += 1

                # FPS measurement
                if fps_frame_count >= 30:
                    elapsed = time.time() - fps_start
                    if elapsed > 0:
                        self._display_fps = fps_frame_count / elapsed
                    fps_start = time.time()
                    fps_frame_count = 0

                # Frame skipping — reuse cached detections
                if frame_count % self.frame_skip != 0:
                    annotated = self._annotate(frame, cached_threats, cached_concealed, detector)
                    self._publish(annotated)
                    continue

                buffer.push(frame)

                try:
                    threats, person_boxes = detector.detect(frame)
                except Exception as e:
                    print(f"[Engine] Detection error: {e}")
                    continue

                concealed_boxes: List[Tuple[int, int, int, int]] = []
                if person_boxes:
                    try:
                        concealed_boxes = face_checker.check(frame, person_boxes)
                    except Exception as e:
                        print(f"[Engine] Face check error: {e}")

                cached_threats   = threats
                cached_concealed = concealed_boxes

                # Update stats
                if threats:
                    self.total_detections += len(threats)
                    for label, _, _ in threats:
                        self.threat_counts[label] = self.threat_counts.get(label, 0) + 1
                if concealed_boxes:
                    self.total_concealed += len(concealed_boxes)

                # ─── ALERT PIPELINE ───
                # Priority: weapon threats send WhatsApp + log.
                # Concealment alone only logs to DB/dashboard (no WhatsApp).
                has_weapon = bool(threats)
                has_concealment = bool(concealed_boxes)

                if has_weapon:
                    best_label, best_conf, _ = max(threats, key=lambda d: d[1])
                    if self._last_alert_time is None or (time.time() - self._last_alert_time) > cooldown_seconds:
                        snapshot_path = save_frame_snapshot(frame)
                        base_name  = f"event_{best_label}_{int(time.time())}"
                        pre_path   = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_pre.mp4")
                        post_path  = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_post.mp4")
                        full_path  = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_full.mp4")

                        buffer.dump_to_video(pre_path)
                        _save_post_event_clip(cap, self.post_seconds, int(fps), frame_size, post_path)

                        try:
                            writer = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
                            for p in [pre_path, post_path]:
                                cap2 = cv2.VideoCapture(p)
                                while True:
                                    ok, fr = cap2.read()
                                    if not ok:
                                        break
                                    writer.write(fr)
                                cap2.release()
                            writer.release()
                        except Exception:
                            full_path = pre_path

                        ts = timestamp_now()
                        # Build a clear, professional alert message
                        message = (
                            f"🚨 *SECURITY ALERT — WEAPON DETECTED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"🔪 Object: {best_label.upper()}\n"
                            f"📊 Confidence: {best_conf:.0%}\n"
                            f"🕐 Time: {ts}\n"
                        )
                        if has_concealment:
                            message += f"⚠️ Warning: {len(concealed_boxes)} person(s) with concealed face\n"
                        message += (
                            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📸 Snapshot and 10s video clip saved.\n"
                            f"🖥️ View dashboard: http://localhost:8080"
                        )

                        media_url_prefix = os.getenv("MEDIA_URL_PREFIX")
                        media_url = None
                        if media_url_prefix:
                            media_url = media_url_prefix.rstrip("/") + "/" + os.path.basename(snapshot_path)

                        sent_via = alerter.send_alert(message, media_url=media_url)
                        print(f"[ALERT] {best_label.upper()} detected at {ts}, WhatsApp sent via {sent_via}")

                        append_csv(ts, best_label, best_conf, snapshot_path, full_path)
                        insert_db(ts, best_label, best_conf, snapshot_path, full_path)
                        self._last_alert_time = time.time()

                elif has_concealment:
                    # Log concealment to DB for dashboard, but do NOT send WhatsApp
                    ts = timestamp_now()
                    append_csv(ts, "concealed_person", 0.99, "", "")
                    insert_db(ts, "concealed_person", 0.99, "", "")

                annotated = self._annotate(frame, threats, concealed_boxes, detector)
                self._publish(annotated)

        finally:
            cap.release()
            face_checker.close()
            cv2.destroyAllWindows()
            self._status = "stopped"
            print("[Engine] Detection stopped.")

    def process_client_frame(self, frame_bytes: bytes) -> Tuple[bytes, bool]:
        """Process a single frame sent by the client.
        Runs YOLO + Face detection, updates stats, registers alerts,
        and returns (annotated_jpeg_bytes, has_weapon_flag).
        """
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return b"", False

        try:
            threats, person_boxes = self.detector.detect(frame)
        except Exception as e:
            print(f"[Engine] Client frame detection error: {e}")
            return b"", False

        concealed_boxes = []
        if person_boxes:
            try:
                concealed_boxes = self.face_checker.check(frame, person_boxes)
            except Exception as e:
                print(f"[Engine] Client face check error: {e}")

        # Update stats
        if threats:
            self.total_detections += len(threats)
            for label, _, _ in threats:
                self.threat_counts[label] = self.threat_counts.get(label, 0) + 1
        if concealed_boxes:
            self.total_concealed += len(concealed_boxes)

        has_weapon = bool(threats)
        has_concealment = bool(concealed_boxes)
        cooldown_seconds = 120

        if has_weapon:
            best_label, best_conf, _ = max(threats, key=lambda d: d[1])
            if self._last_alert_time is None or (time.time() - self._last_alert_time) > cooldown_seconds:
                snapshot_path = save_frame_snapshot(frame)
                ts = timestamp_now()
                message = (
                    f"🚨 *SECURITY ALERT — WEAPON DETECTED*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔪 Object: {best_label.upper()}\n"
                    f"📊 Confidence: {best_conf:.0%}\n"
                    f"🕐 Time: {ts}\n"
                )
                if has_concealment:
                    message += f"⚠️ Warning: {len(concealed_boxes)} person(s) with concealed face\n"
                message += (
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📸 Snapshot saved.\n"
                    f"🖥️ View dashboard: http://localhost:8080"
                )

                media_url_prefix = os.getenv("MEDIA_URL_PREFIX")
                media_url = None
                if media_url_prefix:
                    media_url = media_url_prefix.rstrip("/") + "/" + os.path.basename(snapshot_path)

                sent_via = self.alerter.send_alert(message, media_url=media_url)
                print(f"[ALERT-CLIENT] {best_label.upper()} at {ts}, WhatsApp sent via {sent_via}")

                append_csv(ts, best_label, best_conf, snapshot_path, "")
                insert_db(ts, best_label, best_conf, snapshot_path, "")
                self._last_alert_time = time.time()

        elif has_concealment:
            ts = timestamp_now()
            append_csv(ts, "concealed_person", 0.99, "", "")
            insert_db(ts, "concealed_person", 0.99, "", "")

        annotated = self._annotate(frame, threats, concealed_boxes, self.detector)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            jpeg_bytes = buf.tobytes()
            with self._lock:
                self._latest_frame = jpeg_bytes
            return jpeg_bytes, has_weapon

        return b"", False

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _annotate(self, frame, threats, concealed_boxes, detector) -> np.ndarray:
        annotated = frame.copy()
        if threats:
            annotated = detector.draw_detections(annotated, threats)
        if concealed_boxes:
            annotated = detector.draw_concealment_warnings(annotated, concealed_boxes)
        cv2.putText(annotated, f"FPS: {self._display_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

    def _publish(self, frame: np.ndarray):
        """Encode frame as JPEG and store for streaming."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._latest_frame = buf.tobytes()
