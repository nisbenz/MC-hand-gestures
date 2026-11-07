"""Webcam input: hands out frames stamped with their capture time."""

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    image_bgr: np.ndarray
    t_capture: float  # time.perf_counter() right after the frame was read


class Camera:
    live = True  # a recording (Replay) ends; a camera that stops may come back

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720):
        self._device, self._size = device, (width, height)
        self._index = 0
        if not self._open():
            raise RuntimeError(f"Could not open camera {device}")

    def _open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            return False
        # Many USB webcams default to uncompressed YUYV, which manages only ~5 fps at 720p.
        # MJPG is compressed and reaches 30 fps. Set it before the size.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        return True

    def reopen(self) -> bool:
        """After the camera stopped (e.g. unplugged): try to open it again (O9)."""
        self._cap.release()
        return self._open() and self._cap.read()[0]

    def describe(self) -> str:
        """Format, size and rate the camera actually agreed to (it may ignore what we asked for)."""
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"{fourcc} {w}x{h} @ {self._cap.get(cv2.CAP_PROP_FPS):.0f} fps"

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        frame = Frame(self._index, image, time.perf_counter())
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
