import os
import csv
import sqlite3
from datetime import datetime
from typing import Tuple

import cv2


# Paths
DEFAULT_LOG_DIR = "logs"
DEFAULT_MEDIA_DIR = "media"
DEFAULT_DB_PATH = os.path.join(DEFAULT_LOG_DIR, "detections.db")
DEFAULT_CSV_PATH = os.path.join(DEFAULT_LOG_DIR, "detections.csv")


def ensure_directories() -> Tuple[str, str, str]:
	os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
	os.makedirs(DEFAULT_MEDIA_DIR, exist_ok=True)
	return DEFAULT_LOG_DIR, DEFAULT_MEDIA_DIR, DEFAULT_DB_PATH


def timestamp_now() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_filename() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_frame_snapshot(frame, media_dir: str = DEFAULT_MEDIA_DIR) -> str:
	filename = f"snapshot_{timestamp_filename()}.jpg"
	path = os.path.join(media_dir, filename)
	cv2.imwrite(path, frame)
	return path


def init_csv(csv_path: str = DEFAULT_CSV_PATH):
	if not os.path.exists(csv_path):
		with open(csv_path, mode="w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["timestamp", "object", "confidence", "image_path", "clip_path"])  # header


def append_csv(timestamp: str, obj: str, confidence: float, image_path: str, clip_path: str = "", csv_path: str = DEFAULT_CSV_PATH):
	with open(csv_path, mode="a", newline="") as f:
		writer = csv.writer(f)
		writer.writerow([timestamp, obj, f"{confidence:.2f}", image_path, clip_path])


def init_db(db_path: str = DEFAULT_DB_PATH):
	conn = sqlite3.connect(db_path)
	try:
		cur = conn.cursor()
		cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS detections (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp TEXT NOT NULL,
				object TEXT NOT NULL,
				confidence REAL NOT NULL,
				image_path TEXT NOT NULL,
				clip_path TEXT
			)
			"""
		)
		conn.commit()
	finally:
		conn.close()


def insert_db(timestamp: str, obj: str, confidence: float, image_path: str, clip_path: str = "", db_path: str = DEFAULT_DB_PATH):
	conn = sqlite3.connect(db_path)
	try:
		cur = conn.cursor()
		cur.execute(
			"INSERT INTO detections (timestamp, object, confidence, image_path, clip_path) VALUES (?, ?, ?, ?, ?)",
			(timestamp, obj, float(confidence), image_path, clip_path),
		)
		conn.commit()
	finally:
		conn.close()


class VideoRingBuffer:
	"""Fixed-size video frame buffer to hold N seconds of frames before trigger."""

	def __init__(self, max_seconds: int, fps: int, frame_size: Tuple[int, int]):
		self.max_frames = max(1, int(max_seconds * fps))
		self.buffer = []
		self.fps = fps
		self.frame_size = frame_size

	def push(self, frame):
		self.buffer.append(frame.copy())
		if len(self.buffer) > self.max_frames:
			self.buffer.pop(0)

	def dump_to_video(self, out_path: str):
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_path, fourcc, self.fps, self.frame_size)
		for f in self.buffer:
			writer.write(f)
		writer.release()
		return out_path
