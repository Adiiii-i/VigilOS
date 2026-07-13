import os
import time
from typing import Optional, List, Tuple

import cv2

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


def save_post_event_clip(cap: cv2.VideoCapture, seconds: int, fps: int, frame_size, out_path: str):
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
	frames_to_write = int(seconds * fps)
	for _ in range(frames_to_write):
		ret, frame = cap.read()
		if not ret:
			break
		writer.write(frame)
	writer.release()
	return out_path


def run(camera_index: int = 0, model_name: str = None, conf_threshold: float = None, pre_seconds: int = 5, post_seconds: int = 5):
	print("Starting Violence Detection System...")
	
	ensure_directories()
	init_csv()
	init_db()
	print("Directories and databases initialized")

	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		raise RuntimeError("Cannot open camera/device")

	# Optimize camera settings for performance
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

	# Wait for camera to initialize (important on macOS)
	print("Waiting for camera to initialize...")
	time.sleep(2)
	
	# Try to read a few frames to ensure camera is working
	for i in range(5):
		ret, frame = cap.read()
		if ret:
			print(f"Camera test frame {i+1} successful")
			break
		else:
			print(f"Camera test frame {i+1} failed, retrying...")
			time.sleep(0.5)
	else:
		print("Warning: Camera may not be working properly")

	fps = cap.get(cv2.CAP_PROP_FPS)
	if not fps or fps <= 1:
		fps = 15.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
	frame_size = (width, height)
	
	print(f"Camera initialized: {width}x{height} @ {fps}fps")

	# Use smaller model and higher threshold for better performance
	model_name = model_name or os.getenv("MODEL_NAME", "yolov8n.pt")
	conf_threshold = conf_threshold if conf_threshold is not None else float(os.getenv("CONF_THRESHOLD", 0.5))
	
	print(f"Loading model: {model_name} with threshold: {conf_threshold}")

	detector = YoloViolenceDetector(model_name=model_name, confidence_threshold=conf_threshold)
	face_checker = FaceOcclusionChecker()
	print("Model loaded successfully")
	
	alerter = AlertSender()
	buffer = VideoRingBuffer(max_seconds=pre_seconds, fps=int(fps), frame_size=frame_size)

	last_alert_time: Optional[float] = None
	cooldown_seconds = 120  # 2 minutes — saves Twilio credits
	frame_skip = 3  # Process every Nth frame for better performance

	# Cached results from last inference (reused on skipped frames)
	cached_threats: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
	cached_concealed: List[Tuple[int, int, int, int]] = []

	# FPS measurement
	fps_start = time.time()
	fps_frame_count = 0
	display_fps = 0.0

	print("Starting detection loop...")
	try:
		frame_count = 0
		consecutive_failures = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				consecutive_failures += 1
				print(f"Failed to read frame (attempt {consecutive_failures})")
				if consecutive_failures > 10:
					print("Too many consecutive failures, stopping")
					break
				time.sleep(0.1)
				continue
			
			consecutive_failures = 0  # Reset on success
			frame_count += 1
			fps_frame_count += 1

			# Measure display FPS every 30 frames
			if fps_frame_count >= 30:
				elapsed = time.time() - fps_start
				if elapsed > 0:
					display_fps = fps_frame_count / elapsed
				fps_start = time.time()
				fps_frame_count = 0

			# --- Skip frames: reuse cached detections ---
			if frame_count % frame_skip != 0:
				annotated = frame.copy()
				if cached_threats:
					annotated = detector.draw_detections(annotated, cached_threats)
				if cached_concealed:
					annotated = detector.draw_concealment_warnings(annotated, cached_concealed)
				# Show FPS
				cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, 25),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				cv2.imshow("Violence Detection", annotated)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				continue

			buffer.push(frame)
			
			try:
				threats, person_boxes = detector.detect(frame)
			except Exception as e:
				print(f"Detection error: {e}")
				continue

			# Face occlusion check on detected persons
			concealed_boxes: List[Tuple[int, int, int, int]] = []
			if person_boxes:
				try:
					concealed_boxes = face_checker.check(frame, person_boxes)
				except Exception as e:
					print(f"Face check error: {e}")

			# Cache for skipped frames
			cached_threats = threats
			cached_concealed = concealed_boxes

			# --- Alert logic ---
			# Combine threats + concealment into one detection event
			all_detections = list(threats)  # copy
			for bbox in concealed_boxes:
				all_detections.append(("concealed_person", 0.99, bbox))

			if all_detections:
				best_label, best_conf, _ = max(all_detections, key=lambda d: d[1])

				if last_alert_time is None or (time.time() - last_alert_time) > cooldown_seconds:
					# Snapshot
					snapshot_path = save_frame_snapshot(frame)

					# Build clip paths
					base_name = f"event_{best_label}_{int(time.time())}"
					pre_path = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_pre.mp4")
					post_path = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_post.mp4")
					full_clip_path = os.path.join(DEFAULT_MEDIA_DIR, base_name + "_full.mp4")

					# Save pre-buffer clip
					buffer.dump_to_video(pre_path)
					# Save post segment live
					save_post_event_clip(cap, post_seconds, int(fps), frame_size, post_path)

					# Concatenate pre and post into full clip if possible (simple re-encode)
					try:
						fourcc = cv2.VideoWriter_fourcc(*"mp4v")
						writer = cv2.VideoWriter(full_clip_path, fourcc, fps, frame_size)
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
						full_clip_path = pre_path

					# Alert message — differentiate threat types
					ts = timestamp_now()
					if best_label == "concealed_person":
						message = f"⚠️ Concealed person detected!\nFace not visible — possible mask/hood.\nTime: {ts}"
					else:
						message = f"🚨 THREAT detected! {best_label} found (conf {best_conf:.0%})\nTime: {ts}"
						if concealed_boxes:
							message += f"\n⚠️ Also: {len(concealed_boxes)} concealed person(s)"

					media_url_prefix = os.getenv("MEDIA_URL_PREFIX")
					media_url = None
					if media_url_prefix:
						media_url = media_url_prefix.rstrip("/") + "/" + os.path.basename(snapshot_path)

					sent_via = alerter.send_alert(message, media_url=media_url)
					print(f"[ALERT] {best_label.upper()} detected at {ts}, alert sent via {sent_via}")

					append_csv(ts, best_label, best_conf, snapshot_path, full_clip_path)
					insert_db(ts, best_label, best_conf, snapshot_path, full_clip_path)

					last_alert_time = time.time()

			# --- Draw annotations ---
			annotated = frame.copy()
			if threats:
				annotated = detector.draw_detections(annotated, threats)
			if concealed_boxes:
				annotated = detector.draw_concealment_warnings(annotated, concealed_boxes)
			# Show FPS
			cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, 25),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			cv2.imshow("Violence Detection", annotated)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		face_checker.close()
		cv2.destroyAllWindows()
		print("Detection stopped")


if __name__ == "__main__":
	run()
