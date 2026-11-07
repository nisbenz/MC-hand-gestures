"""The pipeline: one frame in, cleaned players and events out.

Live, replay, the server and the report all go through Engine.step, so a
recording is processed exactly like the camera.
"""

from dataclasses import dataclass, field

from mc_gesture.capture.camera import Frame
from mc_gesture.config import Config
from mc_gesture.detect.detector import Detection, Detector
from mc_gesture.players import tracker


@dataclass
class Output:
    t: float                       # seconds since the first frame
    detection: Detection
    players: tracker.Frame
    events: list[dict] = field(default_factory=list)


class Engine:
    def __init__(self, config: Config | None = None, detector: Detector | None = None, use_gpu: bool = True):
        self.config = config or Config()
        self.detector = detector or Detector(use_gpu=use_gpu)
        self.tracker = tracker.PlayerTracker(self.config)
        self._t0 = None

    def step(self, frame: Frame) -> Output:
        if self._t0 is None:
            self._t0 = frame.t_capture
        t = frame.t_capture - self._t0
        detection = self.detector.process(frame.image_bgr, frame.t_capture)
        players = self.tracker.update(detection, t)
        events = [dict(e, t=round(t, 3)) for e in players.events]
        return Output(t, detection, players, events)

    def close(self) -> None:
        self.detector.close()
