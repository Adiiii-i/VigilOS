"""Face occlusion / concealment detector.

Uses MediaPipe Face Detection (tasks API) to check whether a detected
person's face is visible.  If YOLO finds a person but MediaPipe finds
**no face** in the head region across multiple consecutive frames, the
person is flagged as potentially concealed (mask, helmet, hoodie, etc.).

Includes debouncing: a person is only flagged as concealed if their face
is missing for CONSECUTIVE_THRESHOLD frames in a row, preventing false
positives from momentary glances away or lighting changes.
"""

import os
from typing import List, Tuple, Dict

import cv2
import numpy as np

try:
	import mediapipe as mp
	_MP_AVAILABLE = True
except ImportError:
	mp = None
	_MP_AVAILABLE = False

# Path to the BlazeFace TFLite model (downloaded alongside this file)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blaze_face_short_range.tflite")

# A person must have NO face detected for this many consecutive inference
# frames before being flagged as "concealed".  At frame_skip=3 and 15 fps
# this is roughly 3 × 3 / 15 ≈ 0.6 seconds of sustained no-face.
CONSECUTIVE_THRESHOLD = 3


class FaceOcclusionChecker:
	"""Lightweight face-presence checker backed by MediaPipe with debounce."""

	def __init__(self, min_detection_confidence: float = 0.5):
		"""
		Parameters
		----------
		min_detection_confidence : float
			MediaPipe face-detection threshold.  0.5 is balanced — high
			enough to avoid false "concealed" flags on clearly visible
			faces, low enough to catch real faces at slight angles.
		"""
		self.enabled = _MP_AVAILABLE and os.path.exists(_MODEL_PATH)
		if self.enabled:
			opts = mp.tasks.vision.FaceDetectorOptions(
				base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
				min_detection_confidence=min_detection_confidence,
			)
			self._face_detector = mp.tasks.vision.FaceDetector.create_from_options(opts)
		else:
			self._face_detector = None
			if not _MP_AVAILABLE:
				print("[FaceChecker] mediapipe not installed — face occlusion checks disabled")
			elif not os.path.exists(_MODEL_PATH):
				print(f"[FaceChecker] model not found at {_MODEL_PATH} — face occlusion checks disabled")

		# Track consecutive no-face counts per approximate person position
		# Key = rough grid position, Value = consecutive miss count
		self._miss_counts: Dict[Tuple[int, int], int] = {}

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def check(
		self,
		frame: np.ndarray,
		person_boxes: List[Tuple[int, int, int, int]],
	) -> List[Tuple[int, int, int, int]]:
		"""Return bounding boxes of persons whose face is NOT visible
		for at least CONSECUTIVE_THRESHOLD frames in a row.

		Parameters
		----------
		frame : np.ndarray
			The full BGR camera frame.
		person_boxes : list[(x, y, w, h)]
			Person bounding boxes from YOLO.

		Returns
		-------
		concealed : list[(x, y, w, h)]
			Subset of *person_boxes* that are confirmed concealed.
		"""
		if not self.enabled or not person_boxes:
			return []

		concealed: List[Tuple[int, int, int, int]] = []
		h_frame, w_frame = frame.shape[:2]
		seen_keys: set = set()

		for bbox in person_boxes:
			x, y, w, h = bbox

			# Skip very small person detections (too far away to check)
			if w < 60 or h < 80:
				continue

			# Crop the top ~40 % of the person bbox as the "head region"
			# (slightly larger than before to catch faces better)
			head_bottom = y + int(h * 0.40)
			# Clamp to frame boundaries
			cx1 = max(0, x)
			cy1 = max(0, y)
			cx2 = min(w_frame, x + w)
			cy2 = min(h_frame, head_bottom)

			if cx2 - cx1 < 30 or cy2 - cy1 < 30:
				continue

			head_crop = frame[cy1:cy2, cx1:cx2]
			grid_key = self._grid_key(x, y, w, h)
			seen_keys.add(grid_key)

			if self._has_face(head_crop):
				# Face found — reset miss counter
				self._miss_counts[grid_key] = 0
			else:
				# No face — increment miss counter
				count = self._miss_counts.get(grid_key, 0) + 1
				self._miss_counts[grid_key] = count
				if count >= CONSECUTIVE_THRESHOLD:
					concealed.append(bbox)

		# Clean up old keys for persons no longer in frame
		stale = [k for k in self._miss_counts if k not in seen_keys]
		for k in stale:
			del self._miss_counts[k]

		return concealed

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	@staticmethod
	def _grid_key(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
		"""Quantize person center to a coarse grid so we can track them
		across slight position changes between frames."""
		cx = (x + w // 2) // 80   # ~80px grid cells
		cy = (y + h // 2) // 80
		return (cx, cy)

	def _has_face(self, bgr_crop: np.ndarray) -> bool:
		"""Return True if MediaPipe detects at least one face in *bgr_crop*."""
		rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		result = self._face_detector.detect(mp_image)
		return bool(result.detections)

	def close(self):
		"""Release MediaPipe resources."""
		if self._face_detector is not None:
			self._face_detector.close()
