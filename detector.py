from typing import List, Tuple, Optional, Dict

import os
import numpy as np
import cv2
from ultralytics import YOLO

import sys

# Alleviate Hugging Face ZeroGPU restriction dynamically
try:
	import spaces
except ImportError:
	# Define a mock spaces module so local runs don't crash on macOS/CPU
	class DummySpaces:
		@staticmethod
		def GPU(func=None, duration=None):
			if func is not None:
				return func
			def decorator(f):
				return f
			return decorator
	sys.modules["spaces"] = DummySpaces
	import spaces

# Classes considered violent / threatening
THREAT_CLASSES = {"knife", "fork", "scissors", "bottle"}

# COCO class IDs we care about:
# 0=person, 39=bottle, 42=fork, 43=knife, 76=scissors
COCO_TARGET_IDS = [0, 39, 42, 43, 76]

PERSON_LABEL = "person"


try:
	import torch
except ImportError:
	torch = None

@spaces.GPU
def _run_yolo_inference(model, frame, imgsz, device, classes):
	return model(
		frame,
		imgsz=imgsz,
		device=device,
		classes=classes,
		verbose=False,
	)[0]


def _select_device() -> str:
	"""Auto-detect the best available compute device."""
	if torch is not None:
		if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return "mps"
		if torch.cuda.is_available():
			return "cuda"
	return "cpu"


class YoloViolenceDetector:
	def __init__(self, model_name: str = "yolov8m.pt", confidence_threshold: float = 0.4, class_name_alias: Optional[Dict[str, str]] = None):
		self.device = _select_device()
		self.model = YOLO(model_name)
		self.confidence_threshold = confidence_threshold
		self.class_name_alias = class_name_alias or {}
		self.imgsz = int(os.getenv("IMG_SIZE", "480"))  # 480 for better accuracy
		self.debug = os.getenv("DEBUG_DETECTIONS", "0") == "1"
		print(f"[Detector] device={self.device}, imgsz={self.imgsz}, model={model_name}")

	def _normalize_label(self, label: str) -> str:
		label_lower = label.strip().lower()
		if label_lower in self.class_name_alias:
			label_lower = self.class_name_alias[label_lower]
		return label_lower

	def detect(self, frame: np.ndarray) -> Tuple[
		List[Tuple[str, float, Tuple[int, int, int, int]]],
		List[Tuple[int, int, int, int]],
	]:
		"""Run YOLO inference on *frame*.

		Returns
		-------
		threats : list[(label, confidence, (x, y, w, h))]
			Detections that match THREAT_CLASSES.
		person_boxes : list[(x, y, w, h)]
			Bounding boxes for every detected person (used by face checker).
		"""
		results = _run_yolo_inference(
			self.model,
			frame,
			self.imgsz,
			self.device,
			COCO_TARGET_IDS
		)

		if self.debug:
			# Print top 5 raw detections regardless of filtering
			raw = []
			for box in results.boxes:
				cls_id = int(box.cls[0])
				conf = float(box.conf[0])
				label = results.names.get(cls_id, str(cls_id))
				raw.append((label, conf))
			raw.sort(key=lambda x: x[1], reverse=True)
			print("Top detections:", [f"{l}:{c:.2f}" for l, c in raw[:5]])

		threats: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
		person_boxes: List[Tuple[int, int, int, int]] = []

		for box in results.boxes:
			cls_id = int(box.cls[0])
			conf = float(box.conf[0])
			label = results.names.get(cls_id, str(cls_id))
			norm_label = self._normalize_label(label)
			x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
			bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)

			if norm_label == PERSON_LABEL and conf >= self.confidence_threshold:
				person_boxes.append(bbox)

			if conf >= self.confidence_threshold and norm_label in THREAT_CLASSES:
				threats.append((norm_label, conf, bbox))

		return threats, person_boxes

	@staticmethod
	def draw_detections(frame: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
		for label, conf, (x, y, w, h) in detections:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			text = f"{label} {conf*100:.1f}%"
			cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		return frame

	@staticmethod
	def draw_concealment_warnings(frame: np.ndarray, concealed_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
		"""Draw yellow bounding boxes for persons whose face is not visible."""
		for (x, y, w, h) in concealed_boxes:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
			cv2.putText(frame, "!! CONCEALED", (x, max(20, y - 10)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
		return frame
