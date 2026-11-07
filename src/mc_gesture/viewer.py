"""Step 1 debug viewer: camera + GestureRecognizer + Pose drawn on screen, with timings.

Run:  uv run mc-gesture-view      (press q to quit)
"""

import argparse
import signal
import statistics
import time
from collections import deque

import cv2
from mediapipe.tasks.python import vision

from mc_gesture.capture.camera import Camera
from mc_gesture.detect.detector import Detection, Detector

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
    args = parser.parse_args()

    camera = Camera(args.camera)
    print(f"[camera] {camera.describe()}")
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
            if frame is None:
                break
            detection = detector.process(frame.image_bgr, frame.t_capture)
            t_done = time.perf_counter()

            window.append((
                detection.gesture_ms,
                detection.pose_ms,
                (t_done - frame.t_capture) * 1000,
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
                f"capture->result {total_ms:5.1f} ms (median)",
                f"hands {len(detection.hands)}  bodies {len(detection.poses)}",
            ])
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
