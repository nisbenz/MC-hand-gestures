"""Latency / error report for a recording (D13).

    uv run python tools/report.py recordings/<session> [--cpu] [--config calibration.json]

Runs the engine over the recording exactly as it runs live, then:
  - engine latency: time from a frame being read to its events being ready
  - if <session>/labels.csv exists (tools/label.py): misses, false activations and
    detection delay against the hand-made labels
Writes <session>/report.md and prints it.
"""

import argparse
import platform
import signal
import statistics
import time
from datetime import datetime
from pathlib import Path

from mc_gesture.config import Config
from mc_gesture.engine import Engine
from mc_gesture.record import labels as gt
from mc_gesture.record.recorder import Replay

TARGET_LATENCY_MS = 100     # D13: median latency
TARGET_FALSE_PER_5MIN = 1   # D13: false activations per 5 minutes of play
TARGET_MISS_RATE = 0.05     # D13: missed gestures


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config = Config.load(args.config)
    source, engine = Replay(args.session, realtime=False), Engine(config, use_gpu=not args.cpu)
    latencies, events, frames, duration = [], [], 0, 0.0
    try:
        while (frame := source.read()) is not None:
            t0 = time.perf_counter()
            out = engine.step(frame)
            latencies.append((time.perf_counter() - t0) * 1000)
            events += out.events
            frames, duration = frames + 1, out.t
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        delegates = f"gesture {engine.detector.gesture_delegate}, pose {engine.detector.pose_delegate}"
        engine.close()
        source.close()
    latencies = latencies[10:]  # skip warm-up

    median_ms = statistics.median(latencies)
    lines = [
        f"# MC-Gesture report: {args.session.name}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} on {platform.node()} ({platform.processor() or platform.machine()}), "
        f"{delegates}. {frames} frames, {duration:.1f} s ({frames / max(duration, 1e-6):.1f} fps recorded).",
        "",
        "## Latency",
        "",
        "| Measure | Value | Target | Result |",
        "|---|---|---|---|",
        f"| Engine latency, median (frame read → events ready) | {median_ms:.1f} ms | ≤ {TARGET_LATENCY_MS} ms | "
        f"{mark(median_ms <= TARGET_LATENCY_MS)} |",
        f"| Engine latency, 95th percentile | {pct(latencies, 0.95):.1f} ms | | |",
        f"| Engine latency, worst | {max(latencies):.1f} ms | | |",
        "",
        "Not included: the camera's own delay (exposure, USB), the game's frame and the display. "
        f"A gesture must also be held {config.min_hold_s * 1000:.0f} ms (`min_hold_s`) before `gesture_start` is sent; "
        "the detection delay below includes that.",
        "",
    ]

    counts = {}
    for e in events:
        if e["type"] == "gesture_start":
            counts[e["gesture"]] = counts.get(e["gesture"], 0) + 1
    lines += ["## Events", "", "| Gesture | gesture_start events |", "|---|---|"]
    lines += [f"| {g} | {counts.get(g, 0)} |" for g in config.gestures]
    lines.append("")

    labels = gt.load(args.session)
    if labels is None:
        lines += ["## Errors", "",
                  f"No `labels.csv` in this session, so misses and false activations can't be measured. "
                  f"Label it with `uv run python tools/label.py {args.session}` and run the report again.", ""]
    else:
        s = gt.score(labels, events)
        minutes = duration / 60
        false_per_5 = len(s.false) / max(minutes, 1e-6) * 5
        miss_rate = len(s.misses) / max(len(labels), 1)
        delays = [d * 1000 for _, d in s.hits]
        lines += [
            "## Errors (against hand-made labels)", "",
            "| Measure | Value | Target | Result |",
            "|---|---|---|---|",
            f"| Labeled gestures | {len(labels)} | | |",
            f"| Missed | {len(s.misses)} ({miss_rate:.0%}) | ≤ {TARGET_MISS_RATE:.0%} | {mark(miss_rate <= TARGET_MISS_RATE)} |",
            f"| False activations | {len(s.false)} ({false_per_5:.1f} per 5 min) | ≤ {TARGET_FALSE_PER_5MIN} per 5 min | "
            f"{mark(false_per_5 <= TARGET_FALSE_PER_5MIN)} |",
        ]
        if delays:
            lines.append(f"| Detection delay, median (label start → gesture_start) | {statistics.median(delays):.0f} ms | | |")
        lines += ["", "Per gesture:", "", "| Gesture | Labeled | Hit | Missed | False |", "|---|---|---|---|---|"]
        for g in sorted({l.gesture for l in labels} | {e["gesture"] for e in s.false}):
            lines.append(f"| {g} | {sum(l.gesture == g for l in labels)} | {sum(l.gesture == g for l, _ in s.hits)} | "
                         f"{sum(l.gesture == g for l in s.misses)} | {sum(e['gesture'] == g for e in s.false)} |")
        if minutes < 5:
            lines += ["", f"Note: only {duration:.0f} s of footage. False activations per 5 min are extrapolated; "
                          "record at least 5 minutes for a trustworthy number."]
        lines.append("")

    report = "\n".join(lines)
    (args.session / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
