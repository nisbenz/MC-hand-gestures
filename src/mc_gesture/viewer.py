"""Debug viewer: draws what the engine sees on top of the camera image, with timings.

Run:     uv run mc-gesture-view                         (press q to quit)
Record:  uv run mc-gesture-view --record
Replay:  uv run mc-gesture-view --replay recordings/<session>
"""

import argparse
import signal
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
from mediapipe.tasks.python import vision

from mc_gesture.capture.camera import Camera
from mc_gesture.config import Config
from mc_gesture.engine import Engine, Output
from mc_gesture.record.recorder import Recorder, Replay

HAND_LINKS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
POSE_LINKS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
WINDOW = "MC-Gesture"
PLAYER_COLORS = [(80, 200, 255), (255, 160, 60), (120, 255, 120), (255, 100, 220)]
GREY, RED = (110, 110, 110), (60, 60, 230)


class Drawer:
    def __init__(self, image, mirror: bool):
        self.image, self.mirror = image, mirror
        self.h, self.w = image.shape[:2]

    def px(self, point):
        x = 1 - point[0] if self.mirror else point[0]
        return int(x * self.w), int(point[1] * self.h)

    def body(self, body, color, thickness=2):
        for a, b in POSE_LINKS:
            if body[a][2] > 0.5 and body[b][2] > 0.5:
                cv2.line(self.image, self.px(body[a]), self.px(body[b]), color, thickness)

    def hand(self, hand, color, label=None):
        for a, b in HAND_LINKS:
            cv2.line(self.image, self.px(hand.landmarks[a]), self.px(hand.landmarks[b]), color, 2)
        if label:
            x, y = self.px(hand.landmarks[0])
            text(self.image, label, (x - 40, y + 30), color)


def text(image, label, origin, color=(255, 255, 255), scale=0.65):
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.7 else 2)


def draw(view, out: Output, mirror: bool, active=lambda player, side: None) -> None:
    d = Drawer(view, mirror)
    for body in out.players.bystanders:
        d.body(body, GREY, 1)
    for hand in out.players.dropped_hands:
        d.hand(hand, RED, "dropped")
    for player in out.players.players:
        color = PLAYER_COLORS[(player.id - 1) % len(PLAYER_COLORS)]
        d.body(player.body, color)
        x, y = d.px(player.body[0])
        tag = f"P{player.id}" + (f" [{player.zone}]" if player.zone else "")
        text(view, tag, (x - 20, y - 30), color, 0.9)
        for side, hand in player.hands.items():
            label = f"{side[0].upper()} {hand.gesture} {hand.gesture_score:.2f}"
            if gesture := active(player.id, side):
                label = f">> {gesture} <<  ({hand.gesture_score:.2f})"  # the event the game sees
            d.hand(hand, color, label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="skip the GPU attempt")
    parser.add_argument("--config", type=Path, help="settings JSON, e.g. from calibration")
    parser.add_argument("--record", action="store_true", help="save the session to recordings/ (off by default)")
    parser.add_argument("--replay", type=Path, metavar="SESSION", help="play a recorded session instead of the camera")
    parser.add_argument("--fast", action="store_true", help="with --replay: run as fast as possible, not in real time")
    args = parser.parse_args()

    source = Replay(args.replay, realtime=not args.fast) if args.replay else Camera(args.camera)
    print(f"[source] {source.describe()}")
    recorder = Recorder(source.describe()) if args.record and not args.replay else None
    if recorder:
        print(f"[record] saving to {recorder.path}")
    config = Config.load(args.config)
    engine = Engine(config, use_gpu=not args.cpu)
    # GUI_NORMAL drops the Qt toolbar and status bar. The status bar redraws pixel
    # coordinates on every mouse move, and with OpenCV's font-less Qt build each redraw
    # prints a "QFont::setPointSizeF" warning.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    window = deque(maxlen=60)  # last ~2 s of frames for the on-screen averages
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame = source.read()
            t_read = time.perf_counter()  # frame.t_capture is the recorded time during replay
            if frame is None:
                break
            if recorder:
                recorder.write(frame)
            out = engine.step(frame)
            t_done = time.perf_counter()
            for event in out.events:
                if event["type"] != "gesture_hold":
                    print("[event]", event)

            window.append((
                out.detection.gesture_ms,
                out.detection.pose_ms,
                (t_done - t_read) * 1000,
                t_done - last_frame_time,
            ))
            last_frame_time = t_done

            view = cv2.flip(frame.image_bgr, 1) if config.mirror else frame.image_bgr.copy()
            draw(view, out, config.mirror, engine.stabilizer.active)

            g_ms, p_ms, total_ms, dt = (statistics.median(col) for col in zip(*window))
            detector = engine.detector
            for i, line in enumerate([
                f"FPS {1 / dt:5.1f}",
                f"gesture {g_ms:5.1f} ms  [{detector.gesture_delegate}]",
                f"pose    {p_ms:5.1f} ms  [{detector.pose_delegate}]",
                f"read->result {total_ms:5.1f} ms (median)",
                f"players {len(out.players.players)}  bystanders {len(out.players.bystanders)}  "
                f"dropped hands {len(out.players.dropped_hands)}",
            ]):
                text(view, line, (10, 28 + i * 26))
            if recorder:
                cv2.circle(view, (view.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                text(view, "REC", (view.shape[1] - 90, 38), (0, 0, 255), 0.7)
            elif args.replay:
                text(view, f"REPLAY {frame.index}", (view.shape[1] - 200, 38), (255, 200, 0), 0.7)
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass  # Ctrl+C: leave the loop, then close the models normally below
    finally:
        # Under `uv run`, Ctrl+C reaches Python twice (from the terminal and forwarded by uv).
        # A second interrupt inside MediaPipe's close() leaves it hanging, so ignore it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        engine.close()
        source.close()
        if recorder:
            recorder.close()
            print(f"[record] {recorder.frames} frames saved to {recorder.path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
