"""Compares detector speed options (O11) on a recorded session.

    uv run python tools/bench_speed.py recordings/<session> [--cpu]

For each option it reports process() time and how often its detections agree with
the baseline (full size, Pose every frame): same number of hands and same gesture
labels on a frame.
"""

import argparse
import signal
import statistics
import time
from pathlib import Path

from mc_gesture.detect.detector import Detector
from mc_gesture.record.recorder import Replay

OPTIONS = {
    "baseline": {"parallel": False},
    "scale 0.5": {"parallel": False, "scale": 0.5},
    "pose every 2": {"parallel": False, "pose_every": 2},
    "parallel": {"parallel": True},
    "parallel + scale 0.5": {"parallel": True, "scale": 0.5},
    "parallel + pose every 2": {"parallel": True, "pose_every": 2},
}


def run(session: Path, use_gpu: bool, **options):
    source, detector = Replay(session, realtime=False), Detector(use_gpu=use_gpu, **options)
    times, labels, bodies = [], [], []
    try:
        while (frame := source.read()) is not None:
            t = time.perf_counter()
            d = detector.process(frame.image_bgr, frame.t_capture)
            times.append((time.perf_counter() - t) * 1000)
            labels.append(sorted(h.gesture for h in d.hands))
            bodies.append(len(d.poses))
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        detector.close()
        source.close()
    return times[10:], labels, bodies  # skip warm-up frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base = None
    print(f"{'option':26s} {'median ms':>9s} {'p95 ms':>7s} {'max fps':>7s} {'gestures agree':>15s} {'bodies agree':>13s}")
    for name, options in OPTIONS.items():
        times, labels, bodies = run(args.session, not args.cpu, **options)
        if base is None:
            base = labels, bodies
        agree = sum(a == b for a, b in zip(labels, base[0])) / len(labels)
        body_agree = sum(a == b for a, b in zip(bodies, base[1])) / len(bodies)
        median, p95 = statistics.median(times), sorted(times)[int(len(times) * 0.95)]
        print(f"{name:26s} {median:9.1f} {p95:7.1f} {1000 / median:7.1f} {agree:14.1%} {body_agree:12.1%}")


if __name__ == "__main__":
    main()
