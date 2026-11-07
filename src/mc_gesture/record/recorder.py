"""Records camera sessions for replay (D15).

A session is a folder in recordings/:
    video.avi    the raw camera frames, MJPG (D17)
    frames.csv   index, t (seconds since the first frame)
    meta.json    camera format, start time, engine version
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from mc_gesture.capture.camera import Frame

RECORDINGS_DIR = Path(__file__).resolve().parents[3] / "recordings"


class Recorder:
    def __init__(self, camera_description: str, fps: float = 30.0, root: Path = RECORDINGS_DIR):
        self.path = root / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path.mkdir(parents=True)
        self._fps = fps
        self._writer = None
        self._t0 = None
        self._csv_file = open(self.path / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["index", "t"])
        self._meta = {
            "camera": camera_description,
            "started": datetime.now().isoformat(timespec="seconds"),
            "format": "MJPG",
        }
        self.frames = 0

    def write(self, frame: Frame) -> None:
        if self._writer is None:
            h, w = frame.image_bgr.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), self._fps, (w, h)
            )
            self._t0 = frame.t_capture
        self._writer.write(frame.image_bgr)
        self._csv.writerow([self.frames, f"{frame.t_capture - self._t0:.6f}"])
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._csv_file.close()
        self._meta["frames"] = self.frames
        (self.path / "meta.json").write_text(json.dumps(self._meta, indent=2))


class Replay:
    """Plays a recorded session back with the same interface as Camera.

    Frames carry their recorded timestamps, so the detector sees the same timing as
    it did live. With realtime=True it also waits between frames like the camera did.
    """

    def __init__(self, session: Path, realtime: bool = True):
        self.path = Path(session)
        self._cap = cv2.VideoCapture(str(self.path / "video.avi"))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open recording {self.path}")
        with open(self.path / "frames.csv", newline="") as f:
            self._times = [float(row["t"]) for row in csv.DictReader(f)]
        self._meta = json.loads((self.path / "meta.json").read_text())
        self._realtime = realtime
        self._index = 0
        self._wall_start = None

    def describe(self) -> str:
        return f"replay {self.path.name} ({len(self._times)} frames, recorded from {self._meta['camera']})"

    def read(self) -> Frame | None:
        if self._index >= len(self._times):
            return None
        ok, image = self._cap.read()
        if not ok:
            return None
        t = self._times[self._index]
        if self._realtime:
            if self._wall_start is None:
                self._wall_start = time.perf_counter() - t
            delay = self._wall_start + t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        frame = Frame(self._index, image, t)
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
