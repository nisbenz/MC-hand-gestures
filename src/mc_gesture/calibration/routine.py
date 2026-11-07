"""Calibration: a short guided routine that tunes the engine to this venue and player.

    uv run mc-gesture-calibrate            (then run the engine with --config calibration.json)

The player follows on-screen prompts:
  1. stand where you will play          -> min_body_size (bystander cut-off, O4)
  2. relax your hands                   -> scores that gestures get by accident
  3. make each gesture in turn          -> scores the real gestures get
From these it sets per-gesture enter/exit thresholds and writes calibration.json.
"""

import argparse
import signal
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from mc_gesture.capture.camera import Camera
from mc_gesture.config import Config
from mc_gesture.engine import Engine
from mc_gesture.record.recorder import Replay

READY_S, COLLECT_S = 2.0, 3.0
MIN_SAMPLES = 10


@dataclass
class Phase:
    name: str
    prompt: str
    gesture: str | None = None
    sizes: list[float] = field(default_factory=list)
    scores: dict[str, list[float]] = field(default_factory=dict)  # recognized label -> scores
    frames: int = 0


def phases(config: Config) -> list[Phase]:
    steps = [Phase("stand", "Stand where you will play and face the camera"),
             Phase("relax", "Keep your hands relaxed: no gesture")]
    steps += [Phase(g, f"Make this gesture with one or both hands: {g}", g) for g in config.gestures]
    return steps


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


def compute(config: Config, results: list[Phase]) -> tuple[Config, list[str]]:
    """Turns the collected samples into settings. Returns the new config and notes for the user."""
    notes = []
    stand = next(p for p in results if p.name == "stand")
    if stand.sizes:
        size = statistics.median(stand.sizes)
        config.min_body_size = round(min(max(size * 0.6, 0.05), 0.3), 3)
        notes.append(f"Body size {size:.3f}; bystander cut-off set to {config.min_body_size} (60% of yours).")
    else:
        notes.append("No player found while standing: body size cut-off unchanged. Is your upper body in view?")

    relax = next(p for p in results if p.name == "relax")
    thresholds = {}
    for phase in (p for p in results if p.gesture):
        g = phase.gesture
        scores = phase.scores.get(g, [])
        share = len(scores) / max(phase.frames, 1)
        if len(scores) < MIN_SAMPLES:
            notes.append(f"{g}: recognized in only {share:.0%} of frames, thresholds unchanged. "
                         "It may be a poor gesture for this camera (O1).")
            continue
        enter = min(max(pct(scores, 0.2) - 0.03, 0.5), 0.9)
        accidental = relax.scores.get(g, [])
        if len(accidental) >= MIN_SAMPLES and pct(accidental, 0.95) >= enter:
            enter = min(pct(accidental, 0.95) + 0.02, 0.95)
            notes.append(f"{g}: relaxed hands already score up to {pct(accidental, 0.95):.2f}; "
                         f"enter raised to {enter:.2f} to avoid accidental starts.")
        exit_ = min(max(min(enter - 0.15, pct(scores, 0.05)), 0.3), enter - 0.05)
        thresholds[g] = [round(enter, 2), round(exit_, 2)]
        notes.append(f"{g}: recognized in {share:.0%} of frames, median score {statistics.median(scores):.2f} "
                     f"-> enter {enter:.2f}, exit {exit_:.2f}.")
    config.thresholds = thresholds
    return config, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", type=Path, help="start from these settings")
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    parser.add_argument("--replay", type=Path, help="calibrate from a recording (for testing)")
    parser.add_argument("--no-view", dest="view", action="store_false", help="no window (prompts in the terminal)")
    args = parser.parse_args()

    config = Config.load(args.config)
    source = Replay(args.replay, realtime=True) if args.replay else Camera(args.camera)
    engine = Engine(config, use_gpu=not args.cpu)
    viewer = None
    if args.view:
        from mc_gesture.viewer import Viewer
        viewer = Viewer(config.mirror)

    steps = phases(config)
    current, phase_start, announced = 0, None, False
    try:
        while current < len(steps):
            frame = source.read()
            if frame is None:
                break
            out = engine.step(frame)
            now = time.perf_counter()
            phase = steps[current]
            if phase_start is None:
                phase_start = now
            elapsed = now - phase_start
            if not announced:
                print(f"[calibrate] {phase.prompt}")
                announced = True

            collecting = elapsed >= READY_S
            if collecting:
                phase.frames += 1
                if out.players.players:
                    player = max(out.players.players, key=lambda p: p.size)  # the person nearest the camera
                    phase.sizes.append(player.size)
                    for hand in player.hands.values():
                        phase.scores.setdefault(hand.gesture, []).append(hand.gesture_score)
            if elapsed >= READY_S + COLLECT_S:
                current, phase_start, announced = current + 1, None, False

            if viewer:
                remaining = (READY_S - elapsed) if not collecting else (READY_S + COLLECT_S - elapsed)
                lines = [f"Step {current + 1 if current < len(steps) else len(steps)}/{len(steps)}", phase.prompt,
                         ("Hold it... " if collecting else "Get ready... ") + f"{max(remaining, 0):.1f} s"]
                if not out.players.players:
                    lines.append("No player found: step closer / show your upper body")
                if not viewer.show(frame.image_bgr, out, engine.stabilizer.active, lines, "CALIBRATING"):
                    print("[calibrate] cancelled")
                    return
    except KeyboardInterrupt:
        print("[calibrate] cancelled")
        return
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        engine.close()
        source.close()
        if viewer:
            viewer.close()

    if current < len(steps):
        print("[calibrate] the video ended before all steps finished; nothing saved")
        return
    config, notes = compute(config, steps)
    config.save(args.out)
    for note in notes:
        print("  -", note)
    print(f"[calibrate] saved {args.out}. Use it with:  uv run mc-gesture --config {args.out}")


if __name__ == "__main__":
    main()
