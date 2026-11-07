"""Debug viewer: camera + GestureRecognizer + Pose drawn on screen, with timings.

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
from mc_gesture.detect.detector import Detection, Detector
from mc_gesture.record.recorder import Recorder, Replay

HAND_LINKS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
POSE_LINKS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
WINDOW = "MC-Gesture step 1"


def _px(point, w, h):
    return int(point[0] * w), int(point[1] * h)


def draw(image, detection: Detection) -> None:
    h, w = image.shape[:2]
    for pose in detection.poses:
        for a, b in POSE_LINKS:
            if pose[a][2] > 0.5 and pose[b][2] > 0.5:
                cv2.line(image, _px(pose[a], w, h), _px(pose[b], w, h), (200, 200, 200), 2)
    for hand in detection.hands:
        for a, b in HAND_LINKS:
            cv2.line(image, _px(hand.landmarks[a], w, h), _px(hand.landmarks[b], w, h), (0, 220, 120), 2)
        x, y = _px(hand.landmarks[0], w, h)
        label = f"{hand.gesture} {hand.gesture_score:.2f}"
        cv2.putText(image, label, (x - 40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)


def draw_stats(image, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="skip the GPU attempt")
    parser.add_argument("--record", action="store_true", help="save the session to recordings/ (off by default)")
    parser.add_argument("--replay", type=Path, metavar="SESSION", help="play a recorded session instead of the camera")
    parser.add_argument("--fast", action="store_true", help="with --replay: run as fast as possible, not in real time")
    args = parser.parse_args()

    if args.replay:
        camera = Replay(args.replay, realtime=not args.fast)
    else:
        camera = Camera(args.camera)
    print(f"[source] {camera.describe()}")
    recorder = Recorder(camera.describe()) if args.record and not args.replay else None
    if recorder:
        print(f"[record] saving to {recorder.path}")
    detector = Detector(use_gpu=not args.cpu)
    # GUI_NORMAL drops the Qt toolbar and status bar. The status bar redraws pixel
    # coordinates on every mouse move, and with OpenCV's font-less Qt build each redraw
    # prints a "QFont::setPointSizeF" warning.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    window = deque(maxlen=60)  # last ~2 s of frames for the on-screen averages
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame = camera.read()
            t_read = time.perf_counter()  # frame.t_capture is the recorded time during replay
            if frame is None:
                break
            if recorder:
                recorder.write(frame)
            detection = detector.process(frame.image_bgr, frame.t_capture)
            t_done = time.perf_counter()

            window.append((
                detection.gesture_ms,
                detection.pose_ms,
                (t_done - t_read) * 1000,
                t_done - last_frame_time,
            ))
            last_frame_time = t_done

            view = cv2.flip(frame.image_bgr, 1)  # mirror so it feels natural; landmarks are flipped below
            for hand in detection.hands:
                hand.landmarks = [(1 - x, y) for x, y in hand.landmarks]
            detection.poses = [[(1 - x, y, v) for x, y, v in pose] for pose in detection.poses]
            draw(view, detection)

            g_ms, p_ms, total_ms, dt = (statistics.median(col) for col in zip(*window))
            draw_stats(view, [
                f"FPS {1 / dt:5.1f}",
                f"gesture {g_ms:5.1f} ms  [{detector.gesture_delegate}]",
                f"pose    {p_ms:5.1f} ms  [{detector.pose_delegate}]",
                f"read->result {total_ms:5.1f} ms (median)",
                f"hands {len(detection.hands)}  bodies {len(detection.poses)}",
            ])
            if recorder:
                cv2.circle(view, (view.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(view, "REC", (view.shape[1] - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif args.replay:
                cv2.putText(view, f"REPLAY {frame.index}", (view.shape[1] - 200, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass  # Ctrl+C: leave the loop, then close the models normally below
    finally:
        # Under `uv run`, Ctrl+C reaches Python twice (from the terminal and forwarded by uv).
        # A second interrupt inside MediaPipe's close() leaves it hanging, so ignore it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        detector.close()
        camera.close()
        if recorder:
            recorder.close()
            print(f"[record] {recorder.frames} frames saved to {recorder.path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
