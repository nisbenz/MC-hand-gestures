"""Runs the engine: camera (or a recording) in, WebSocket messages out.

    uv run mc-gesture                     engine + WebSocket server on ws://127.0.0.1:8765
    uv run mc-gesture --view              ... plus the debug window (press q to quit)
    uv run mc-gesture-view                debug window only, no server
    options: --replay recordings/<session> [--fast], --record, --config calibration.json, --cpu
"""

import argparse
import signal
import statistics
import time
from collections import deque
from pathlib import Path

from mc_gesture.capture.camera import Camera
from mc_gesture.config import Config
from mc_gesture.engine import Engine
from mc_gesture.record.recorder import Recorder, Replay


def parse(argv=None, view: bool = False, server: bool = True):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="skip the GPU attempt")
    parser.add_argument("--config", type=Path, help="settings JSON, e.g. from calibration")
    parser.add_argument("--record", action="store_true", help="save the session to recordings/ (off by default)")
    parser.add_argument("--replay", type=Path, metavar="SESSION", help="play a recorded session instead of the camera")
    parser.add_argument("--fast", action="store_true", help="with --replay: run as fast as possible, not in real time")
    parser.add_argument("--view", action="store_true", default=view, help="show the debug window")
    parser.add_argument("--no-server", dest="server", action="store_false", default=server)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def run(args) -> None:
    source = Replay(args.replay, realtime=not args.fast) if args.replay else Camera(args.camera)
    print(f"[source] {source.describe()}")
    config = Config.load(args.config)
    recorder = Recorder(source.describe()) if args.record and not args.replay else None
    if recorder:
        print(f"[record] saving to {recorder.path}")
    server = None
    if args.server:
        from mc_gesture.transport.server import Server
        server = Server(args.host, args.port)
        print(f"[server] ws://{args.host}:{args.port}")
    engine = Engine(config, use_gpu=not args.cpu)
    viewer = None
    if args.view:
        from mc_gesture.viewer import Viewer
        viewer = Viewer(config.mirror)

    stats = deque(maxlen=60)  # last ~2 s of frames for the on-screen averages
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
            if server:
                for event in out.events:
                    server.publish("events", event)
                server.publish("cursor", {"type": "cursor", "t": round(out.t, 3), "cursors": out.cursors})
                server.publish("landmarks", {"type": "landmarks", "t": round(out.t, 3), "players": out.landmarks})
            t_done = time.perf_counter()
            for event in out.events:
                if event["type"] != "gesture_hold":
                    print("[event]", event)

            stats.append((out.detection.gesture_ms, out.detection.pose_ms, (t_done - t_read) * 1000,
                          t_done - last_frame_time))
            last_frame_time = t_done
            if viewer:
                g_ms, p_ms, total_ms, dt = (statistics.median(col) for col in zip(*stats))
                lines = [
                    f"FPS {1 / dt:5.1f}",
                    f"gesture {g_ms:5.1f} ms  [{engine.detector.gesture_delegate}]",
                    f"pose    {p_ms:5.1f} ms  [{engine.detector.pose_delegate}]",
                    f"read->events {total_ms:5.1f} ms (median)",
                    f"players {len(out.players.players)}  bystanders {len(out.players.bystanders)}  "
                    f"dropped hands {len(out.players.dropped_hands)}",
                ]
                if server:
                    lines.append(f"clients {server.clients}")
                badge = "REC" if recorder else (f"REPLAY {frame.index}" if args.replay else None)
                if not viewer.show(frame.image_bgr, out, engine.stabilizer.active, lines, badge):
                    break
    except KeyboardInterrupt:
        pass  # Ctrl+C: leave the loop, then close the models normally below
    finally:
        # Under `uv run`, Ctrl+C reaches Python twice (from the terminal and forwarded by uv).
        # A second interrupt inside MediaPipe's close() leaves it hanging, so ignore it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        engine.close()
        source.close()
        if server:
            server.close()
        if recorder:
            recorder.close()
            print(f"[record] {recorder.frames} frames saved to {recorder.path}")
        if viewer:
            viewer.close()


def main() -> None:
    run(parse())


def view_main() -> None:
    run(parse(view=True, server=False))


if __name__ == "__main__":
    main()
