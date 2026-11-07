"""Runs MediaPipe GestureRecognizer (D9) and PoseLandmarker (D3) on a frame.

Both models run in VIDEO mode: synchronous calls with our own timestamps, so a
recorded video replays exactly like the live camera (D15).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


@dataclass
class Hand:
    landmarks: list[tuple[float, float]]  # 21 points, normalized to the frame (0..1)
    handedness: str
    gesture: str
    gesture_score: float


@dataclass
class Detection:
    hands: list[Hand] = field(default_factory=list)
    poses: list[list[tuple[float, float, float]]] = field(default_factory=list)  # 33 points (x, y, visibility)
    gesture_ms: float = 0.0
    pose_ms: float = 0.0


def _create(task_cls, options_cls, model: str, use_gpu: bool, **kwargs):
    """Try the GPU first and fall back to the CPU (D8). Returns (task, delegate name)."""
    path = str(MODELS_DIR / model)
    delegates = [BaseOptions.Delegate.GPU] if use_gpu else []
    for delegate in delegates + [BaseOptions.Delegate.CPU]:
        try:
            options = options_cls(
                base_options=BaseOptions(model_asset_path=path, delegate=delegate),
                running_mode=vision.RunningMode.VIDEO,
                **kwargs,
            )
            return task_cls.create_from_options(options), delegate.name
        except Exception as error:  # GPU is unsupported on Windows and can fail to start on Linux
            if delegate == BaseOptions.Delegate.CPU:
                raise
            print(f"[detect] GPU unavailable for {model}, using CPU: {error}")


class Detector:
    def __init__(self, max_people: int = 4, use_gpu: bool = True):
        self._gestures, self.gesture_delegate = _create(
            vision.GestureRecognizer, vision.GestureRecognizerOptions,
            "gesture_recognizer.task", use_gpu, num_hands=max_people * 2,
        )
        if self.gesture_delegate == "GPU":
            # MediaPipe logs "contains CPU only ops": its gesture classifier always runs on the CPU
            self.gesture_delegate = "GPU+CPU"
        self._pose, self.pose_delegate = _create(
            vision.PoseLandmarker, vision.PoseLandmarkerOptions,
            "pose_landmarker_lite.task", use_gpu, num_poses=max_people,
        )
        self._last_ts = -1

    def process(self, image_bgr, t_capture: float) -> Detection:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing millisecond timestamps
        ts = max(int(t_capture * 1000), self._last_ts + 1)
        self._last_ts = ts

        detection = Detection()

        t0 = time.perf_counter()
        g = self._gestures.recognize_for_video(image, ts)
        t1 = time.perf_counter()
        p = self._pose.detect_for_video(image, ts)
        t2 = time.perf_counter()
        detection.gesture_ms = (t1 - t0) * 1000
        detection.pose_ms = (t2 - t1) * 1000

        for i, points in enumerate(g.hand_landmarks):
            top = g.gestures[i][0] if g.gestures and g.gestures[i] else None
            detection.hands.append(Hand(
                landmarks=[(pt.x, pt.y) for pt in points],
                handedness=g.handedness[i][0].category_name if g.handedness else "?",
                gesture=top.category_name if top else "None",
                gesture_score=top.score if top else 0.0,
            ))
        for points in p.pose_landmarks:
            detection.poses.append([(pt.x, pt.y, pt.visibility or 0.0) for pt in points])
        return detection

    def close(self) -> None:
        self._gestures.close()
        self._pose.close()
