"""Health status (P5): tells games when tracking will be unreliable, instead of glitching silently.

Sent once a second on the "health" layer:
    {"type": "health", "status": "ok" | "degraded" | "bad", "fps": 29.8,
     "brightness": 112, "camera": "ok" | "lost", "players": 1, "reasons": [...]}
"""

import cv2

MIN_FPS_OK, MIN_FPS_BAD = 20, 10
DARK, BRIGHT = 50, 215   # mean grey level (0-255)


class Health:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._frames = 0
        self._window_start = None
        self._brightness = None
        self.camera_ok = True
        self.status = None

    def frame(self, image_bgr, now: float) -> None:
        if self._window_start is None:
            self._window_start = now
        self._frames += 1
        if self._frames % 10 == 1:  # brightness doesn't change fast; measure on a small copy
            small = cv2.resize(image_bgr, (64, 36), interpolation=cv2.INTER_AREA)
            self._brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

    def report(self, now: float, players: int) -> dict | None:
        """Returns a health message once per interval, else None."""
        if self._window_start is None:
            self._window_start = now
        elapsed = now - self._window_start
        if elapsed < self.interval_s:
            return None
        fps = self._frames / elapsed
        self._frames, self._window_start = 0, now

        reasons = []
        if not self.camera_ok:
            reasons.append("camera lost")
        if fps < MIN_FPS_OK:
            reasons.append(f"low frame rate ({fps:.0f} fps)")
        if self._brightness is not None and self._brightness < DARK:
            reasons.append("too dark")
        if self._brightness is not None and self._brightness > BRIGHT:
            reasons.append("too bright / backlit")
        bad = not self.camera_ok or fps < MIN_FPS_BAD
        self.status = "bad" if bad else ("degraded" if reasons else "ok")
        return {
            "type": "health", "status": self.status, "fps": round(fps, 1),
            "brightness": None if self._brightness is None else round(self._brightness),
            "camera": "ok" if self.camera_ok else "lost", "players": players, "reasons": reasons,
        }
