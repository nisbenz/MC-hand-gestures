"""Runs MediaPipe GestureRecognizer (D9) and PoseLandmarker (D3) on a frame.

Both models run in VIDEO mode: synchronous calls with our own timestamps, so a
recorded video replays exactly like the live camera (D15).
"""

import time
from concurrent.futures import ThreadPoolExecutor
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
    """Speed options (O11):
    scale       resize the frame before the models (0.5 = half width and height)
    pose_every  run Pose only every Nth frame and reuse its last result in between
    parallel    run Pose on a second thread while GestureRecognizer runs
    """

    def __init__(self, max_people: int = 4, use_gpu: bool = True,
                 scale: float = 1.0, pose_every: int = 1, parallel: bool = True):
        self._scale = scale
        self._pose_every = pose_every
        self._pool = ThreadPoolExecutor(max_workers=1) if parallel else None
        self._frame = 0
        self._last_poses = []
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

    def _timed(self, fn, image, ts):
        t0 = time.perf_counter()
        result = fn(image, ts)
        return result, (time.perf_counter() - t0) * 1000

    def process(self, image_bgr, t_capture: float) -> Detection:
        if self._scale != 1.0:
            image_bgr = cv2.resize(image_bgr, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing millisecond timestamps
        ts = max(int(t_capture * 1000), self._last_ts + 1)
        self._last_ts = ts

        detection = Detection()
        run_pose = self._frame % self._pose_every == 0
        self._frame += 1

        pose_job = None
        if run_pose and self._pool:
            pose_job = self._pool.submit(self._timed, self._pose.detect_for_video, image, ts)
        g, detection.gesture_ms = self._timed(self._gestures.recognize_for_video, image, ts)
        if pose_job:
            p, detection.pose_ms = pose_job.result()
        elif run_pose:
            p, detection.pose_ms = self._timed(self._pose.detect_for_video, image, ts)
        if run_pose:
            self._last_poses = [[(pt.x, pt.y, pt.visibility or 0.0) for pt in points] for points in p.pose_landmarks]
        detection.poses = self._last_poses

        for i, points in enumerate(g.hand_landmarks):
            top = g.gestures[i][0] if g.gestures and g.gestures[i] else None
            detection.hands.append(Hand(
                landmarks=[(pt.x, pt.y) for pt in points],
                handedness=g.handedness[i][0].category_name if g.handedness else "?",
                gesture=top.category_name if top else "None",
                gesture_score=top.score if top else 0.0,
            ))
        return detection

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown()
        self._gestures.close()
        self._pose.close()
