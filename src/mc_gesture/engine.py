"""The pipeline: one frame in, cleaned players and events out.

Live, replay, the server and the report all go through Engine.step, so a
recording is processed exactly like the camera.
"""

from dataclasses import dataclass, field

from mc_gesture.capture.camera import Frame
from mc_gesture.config import Config
from mc_gesture.detect.detector import Detection, Detector
from mc_gesture.players import tracker
from mc_gesture.players.lock import PlayerLock
from mc_gesture.smoothing.points import HandSmoother
from mc_gesture.stabilize.gestures import GestureStabilizer


@dataclass
class Output:
    t: float                       # seconds since the first frame
    detection: Detection
    players: tracker.Frame
    events: list[dict] = field(default_factory=list)
    cursors: list[dict] = field(default_factory=list)     # layer "cursor"
    landmarks: list[dict] = field(default_factory=list)   # layer "landmarks"


class Engine:
    def __init__(self, config: Config | None = None, detector: Detector | None = None, use_gpu: bool = True):
        self.config = config or Config()
        self.detector = detector or Detector(use_gpu=use_gpu)
        self.tracker = tracker.PlayerTracker(self.config)
        self.stabilizer = GestureStabilizer(self.config)
        self.smoother = HandSmoother(self.config)
        self.lock = PlayerLock(self.config) if self.config.lock else None
        self._t0 = None

    def _x(self, x: float) -> float:
        return round(1 - x if self.config.mirror else x, 4)

    def step(self, frame: Frame) -> Output:
        if self._t0 is None:
            self._t0 = frame.t_capture
        t = frame.t_capture - self._t0
        detection = self.detector.process(frame.image_bgr, frame.t_capture)
        players = self.tracker.update(detection, t)
        gesture_events = self.stabilizer.update(players.players, t)
        # player_left comes after the gesture_end / hand_lost events it causes
        joined = [e for e in players.events if e["type"] != "player_left"]
        left = [e for e in players.events if e["type"] == "player_left"]
        events = joined + gesture_events + left
        if self.lock:
            events = self.lock.update(events, t)
        events = [dict(e, t=round(t, 3)) for e in events]

        out = Output(t, detection, players, events)
        keys = set()
        for player in players.players:
            if self.lock and not self.lock.allows(player.id):
                continue
            hands = {}
            for side, hand in player.hands.items():
                key = (player.id, side)
                keys.add(key)
                x, y = self.smoother.cursor(key, hand.landmarks, t)
                out.cursors.append({"player": player.id, "hand": side, "x": self._x(x), "y": round(y, 4),
                                    "gesture": self.stabilizer.active(player.id, side)})
                hands[side] = [[self._x(px), round(py, 4)] for px, py in self.smoother.smooth(key, hand.landmarks, t)]
            out.landmarks.append({
                "player": player.id, "zone": player.zone, "size": round(player.size, 4),
                "body": [[self._x(x), round(y, 4), round(v, 2)] for x, y, v in player.body],
                "hands": hands,
            })
        self.smoother.keep_only(keys)
        return out

    def close(self) -> None:
        self.detector.close()
