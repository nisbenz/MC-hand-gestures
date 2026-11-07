"""Replays a recorded session twice and checks both runs give the same detections.

    uv run python tools/check_replay.py recordings/<session> [--cpu]

If the runs differ, measurements made from replays can't be trusted (D15).
"""

import argparse
import signal
import sys
from pathlib import Path

from mc_gesture.detect.detector import Detector
from mc_gesture.record.recorder import Replay


def run(session: Path, use_gpu: bool) -> list[list[tuple]]:
    source, detector = Replay(session, realtime=False), Detector(use_gpu=use_gpu)
    results = []
    try:
        while (frame := source.read()) is not None:
            d = detector.process(frame.image_bgr, frame.t_capture)
            results.append([(h.gesture, round(h.gesture_score, 4), round(h.landmarks[0][0], 4)) for h in d.hands])
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        detector.close()
        source.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    a, b = run(args.session, not args.cpu), run(args.session, not args.cpu)
    with_hands = sum(1 for r in a if r)
    different = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"frames {len(a)}, with hands {with_hands}, differing between runs {len(different)}")
    if with_hands == 0:
        print("WARNING: no hands in this recording, so the check proves little")
    for i in different[:5]:
        print(f"  frame {i}: {a[i]}  vs  {b[i]}")
    return 1 if different or len(a) != len(b) else 0


if __name__ == "__main__":
    sys.exit(main())
