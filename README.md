# MC-Gesture

A reusable local gesture-control engine for interactive installations: club events, open days, anything with a webcam and a screen.

It turns webcam input into **stable gesture events** that any game can consume over a WebSocket, so games never contain vision code. It deals with the hard parts once: noisy detection, bystanders' hands, short tracking loss, lighting, and measuring latency and errors.

```
camera ─► MediaPipe (gestures + body pose) ─► player filter ─► stabilize ─► WebSocket ─► your game
                                              drop bystanders   start/hold/end   JSON events
```

Design decisions and their reasons are in [DECISIONS.md](DECISIONS.md).

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and a webcam. Built for Linux, macOS and Windows (so far tested on Linux only). It uses the GPU on Linux and macOS when available, and the CPU otherwise.

```bash
uv sync                                   # install (Python 3.12, MediaPipe 1.1.1)
uv run python tools/download_models.py    # fetch the two MediaPipe models (~14 MB)
uv run mc-gesture --view                  # engine + debug window + ws://127.0.0.1:8765
```

To watch the messages a game receives, open `examples/ws_client/index.html` in a browser. If your browser blocks the connection from a local file, serve the folder: `python3 -m http.server -d examples` and open `http://localhost:8000/ws_client/`.

## Gestures

The engine reports three of MediaPipe's built-in gestures: `Closed_Fist` ✊, `Open_Palm` ✋ and `Pointing_Up` ☝️. You can change them with the `gestures` setting; the other built-in gestures are `Thumb_Up`, `Thumb_Down`, `Victory` and `ILoveYou`.

## For game developers: the protocol

Connect to `ws://127.0.0.1:8765`. Every message is a JSON object with a `type`. Coordinates are 0–1 fractions of the camera image, mirrored like a selfie view by default. `t` is seconds since the engine started.

Choose what you receive by sending `{"subscribe": [...]}`. The default is `events` and `health`.

| Layer | Messages |
|---|---|
| `events` | `gesture_start`, `gesture_hold` (every 0.25 s while held), `gesture_end` (with `held_s` and `reason`: `released` or `lost`), `hand_lost`, `player_joined`, `player_left`, `zone_changed`, `player_locked`, `player_unlocked` |
| `cursor` | Every frame: `{"type": "cursor", "cursors": [{"player": 1, "hand": "right", "x": 0.42, "y": 0.61, "gesture": "Closed_Fist" or null}]}`. Smoothed palm center. |
| `landmarks` | Every frame: each player's 33 body points and smoothed 21-point hands |
| `health` | Once a second: `status` (`ok`, `degraded` or `bad`), `fps`, `brightness`, `camera`, `reasons` |

```json
{"type": "gesture_start", "player": 1, "hand": "right", "gesture": "Closed_Fist", "score": 0.91, "t": 12.48}
```

`player` IDs stay the same while a person stays in view. `hand` is the person's own left or right. Your game decides what players and gestures mean; the engine only reports them.

## Running it

| Command | What it does |
|---|---|
| `uv run mc-gesture` | Engine + WebSocket server. Add `--view` for the debug window |
| `uv run mc-gesture-view` | Debug window only, no server |
| `uv run mc-gesture-calibrate` | ~25 s guided routine; writes `calibration.json` |
| `uv run mc-gesture --config calibration.json` | Run with calibrated settings |
| `--record` | Save the session to `recordings/`. **Off by default**; only record people who agreed |
| `--replay recordings/<session>` | Run on a recording instead of the camera (`--fast` = not real time) |
| `--cpu` | Skip the GPU |
| `--host 0.0.0.0` | Accept games from other machines (default: this machine only) |

### At an event

1. Run `uv run mc-gesture-calibrate` with the player standing where they'll play.
2. Start `uv run mc-gesture --config calibration.json --view`.
3. Watch the `HEALTH` line in the debug window. `degraded` usually means bad light or a slow laptop.

## Settings

All settings live in `src/mc_gesture/config.py` and can be overridden by a JSON file (`--config`). The most useful ones:

| Setting | Default | Meaning |
|---|---|---|
| `gestures` | fist, palm, point | Gestures that produce events |
| `min_body_size` | 0.10 | Shoulder width (fraction of frame width) needed to become a player; smaller = bystander. Set by calibration |
| `enter_threshold` / `exit_threshold` | 0.6 / 0.45 | Score to start / end a gesture. Calibration sets these per gesture in `thresholds` |
| `min_hold_s` | 0.1 | How long a gesture must last before `gesture_start` |
| `zones` | none | `{"left": [0, 0, 0.5, 1], "right": [0.5, 0, 1, 1]}` labels players by area (e.g. split-screen PvP) |
| `zone_mode` | `label` | `filter` also ignores players outside every zone |
| `lock` | false | `true`: a player holds an open palm for 1 s to take control; everyone else is ignored |
| `mirror` | true | Mirror x, so moving your hand right moves the cursor right |
| `min_cutoff`, `beta` | 1.0, 0.05 | Cursor smoothing (OneEuroFilter): less jitter ↔ less lag |

## Measuring it: latency and error report

```bash
uv run mc-gesture --record                                # record a session (with consent)
uv run python tools/label.py recordings/<session>         # mark when each gesture was really made
uv run python tools/report.py recordings/<session> --cpu  # latency + misses + false activations
```

The report checks the targets in DECISIONS.md (D13): median latency ≤ 100 ms, at most 1 false activation per 5 minutes, at most 5% missed gestures. It writes `report.md` into the session folder.

Other tools:
- `tools/check_replay.py` checks that a recording replays identically.
- `tools/bench_speed.py` compares speed options on a recording.

## Development

```bash
uv run pytest          # unit tests (no camera needed)
```

| Path | Contents |
|---|---|
| `src/mc_gesture/engine.py` | The pipeline: one frame in, players and events out |
| `capture/`, `detect/` | Webcam input; MediaPipe GestureRecognizer + Pose |
| `players/` | Bystander filter, stable player IDs, zones, lock |
| `stabilize/` | Gesture guesses → start/hold/end events |
| `smoothing/` | OneEuroFilter for the cursor and landmarks |
| `transport/` | WebSocket server |
| `record/` | Recorder, replay, labels |
| `calibration/` | Calibration routine |
| `examples/` | Browser test client |
| `tools/` | Model download, labeling, report, benchmarks |
