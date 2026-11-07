# MC-Gesture: Decision Log

A reusable local engine that turns webcam input into stable gesture events for games and interactive displays.

**MVP scope (from the brief):** one camera, three gestures, one demo game, and a measurable latency/error report.

Each entry records who made the decision and why. "Owner" = decided by the project owner; "Delegated" = chosen during the build after the owner handed the decision over, still to be reviewed.

---

## Decided

| # | Decision | By | Why |
|---|---|---|---|
| D1 | **Python engine**, consumed by games over a network protocol | Owner | Any program in any language can consume it. (The protocol gives compatibility, not Python itself; Python gives MediaPipe, replay and analysis.) |
| D2 | **The engine does all the cleaning.** Games receive simple events and interpret them however they want. Games never clean raw MediaPipe data. | Owner | The point of the project: no vision or pre-processing code inside each game. |
| D3 | **Bystander handling = MediaPipe Pose + simple rules.** Pose finds bodies, hands are matched to bodies, and rules (body size, optional zone, optional lock) pick the players. | Owner | Matches common practice in Kinect installations and public-display research: the model finds people, rules decide who plays. This is a starting bet, not a proven optimum; it will be measured. |
| D5 | **Optional zones defined by the app.** The default is no zone (the whole frame). | Owner | Lets games such as split-screen PvP label players by area without hard-coding that in the engine. |
| D6 | **Smoothing with the `OneEuroFilter` package.** Internals are delegated; the two tuning settings are known: `min_cutoff` (less shaking ↔ more lag) and `beta` (faster catch-up on quick moves). | Owner | A published, well-known method beats a from-scratch filter. |
| D7 | **Output layers:** `events` (gesture start/hold/end, hand lost), `cursor` (smoothed position), `landmarks` (optional cleaned points). Games subscribe to what they need. | Owner | Most games only need events; landmarks give freedom to the rare game that wants more. |
| D8 | **GPU auto-detect with CPU fallback.** Try GPU first and fall back to CPU if it fails. Report which one is running. **Latency must be acceptable on CPU.** | Owner | Club laptops are a mix of Linux, macOS and Windows. MediaPipe's Python GPU option only works on Linux/macOS, so the Windows laptops will run on CPU. |
| D9 | **Classifier = MediaPipe's built-in GestureRecognizer**, behind a swappable interface (camera image in → gesture scores out). | Owner | The brief needs three gestures, not custom ones; the built-in set covers that. Tuning against false activations happens in our stabilize stage either way. Switch to custom geometric rules only if replay shows problems that threshold tuning can't fix. |
| D10 | **Transport = WebSocket** | Owner | Works with web pages and game engines. OSC can come later if installation tools need it. |
| D11 | **Zones support both modes**, filter (drop players outside every zone) and label (report the zone, drop nobody), chosen by the app | Owner | Gives the app freedom. |
| D12 | **Gestures = the 3 most reliable built-in gestures**, chosen by measurement | Owner | "Most reliable" still needs a definition and a test; see O1. |
| D13 | **Report targets (starting values):** median latency ≤ 100 ms on a CPU laptop; false activations ≤ 1 per 5 min of play; miss rate ≤ 5% | Owner | These give the report pass/fail lines. They're starting guesses, not standards; revisit after the first replay measurements. |
| D14 | **Build order:** (1) camera + GestureRecognizer + Pose on screen with timings → (2) recorder + replay → (3) player filter → (4) stabilize → events → (5) WebSocket + layers (cursor smoothing here) → (6) report → (7) calibration → (8) demo game → (9) zones, lock, health | Owner | Each step ends with something runnable. Recording comes early so every later step is tested on the same footage. |
| D15 | **Record and replay harness** (was P4): record the **camera video** plus timestamps, label ground truth, rerun the engine offline to measure false activations, misses and latency | Owner | Required to check D12 and D13. Recording video (not only results) lets us rerun with different settings, such as GPU vs CPU or lighting fixes. |

| D16 | **Python 3.12, MediaPipe pinned to 1.1.1**, environment managed with `uv` | Owner (`uv`: delegated) | 3.12 has the widest package support for Windows and macOS laptops; 1.1.1 is recent but past its first bug-fix release. |

| D17 | **Recordings use MJPG video** (`video.avi` + `frames.csv` timestamps + `meta.json`), and **recording is off by default** (`--record`) | Owner | Measured on the dev webcam: ≈1.5 GB per 10 min, average pixel error 1.4 vs live. mp4v was smaller (0.5 GB) but drifted more (2.5); FFV1 was exact but 6.4 GB. |

## Delegated build choices (to review)

| # | Choice | Why | Cost to change |
|---|---|---|---|
| C1 | **Pose model = `pose_landmarker_lite`** | Fastest option, for the CPU latency target (D8, D13) | Low: swap the file in `tools/download_models.py` |
| C2 | **Both models run in VIDEO mode** (synchronous calls, our own timestamps) | A recorded video then replays exactly like the live camera (D15) | Medium |
| C3 | **Detect up to 4 people / 8 hands** | Bystanders must be *detected* before the player filter can drop them (D3) | Low: one parameter |
| C5 | **Camera requests MJPG** and the engine prints the format/size/rate it actually got | The default YUYV gave 4 FPS at 720p on the dev webcam; MJPG gives 30. Club webcams may behave differently, so the actual format is reported | Low |
| C6 | **Gesture and Pose models run in parallel** (Pose on a second thread), on by default. Was O11. Resizing (`scale`) and skipping Pose frames (`pose_every`) are available but off | Measured on the 26 s recording with `tools/bench_speed.py`. CPU: 42.8 → 24.5 ms per frame (p95 51 → 32 ms, fits the 33 ms camera frame). GPU: 27.5 → 19.7 ms. Results 100% identical to before. Half-size input changed 6% of gesture labels and didn't help on CPU. Measured on a desktop CPU; laptops will be slower | Low: one flag |
| C4 | **Only `opencv-contrib-python`** (pulled in by MediaPipe), not `opencv-python` too | Two OpenCV packages overwrite each other's `cv2` files | Low |
| C7 | **Body size = shoulder width / frame width. A new body needs ≥ 0.10 to join**; this is the default for O4, and calibration (step 7) can set it per venue | Your shoulder width in the test recording was 0.15–0.31 at desk distance; 0.10 is roughly 2.5 m from the camera | Low: one setting |
| C8 | **An existing player only leaves when their median size over ~1 s drops below half the join size**, or after 0.5 s unseen | Pose misjudged your shoulders (0.15 → 0.04) for nearly a second in the recording and created a new player ID; this rule keeps one ID for the whole clip | Low |
| C9 | **A hand belongs to a player if its wrist is within 0.6 shoulder widths of one of the player's pose wrists**; otherwise it's dropped | In the recording, 90% of hands were within 0.29 | Low |
| C10 | **Output coordinates are mirrored by default** (`mirror: true`), like a selfie view | Players face the screen, so moving a hand right should move the cursor right | Low: one setting |
| C11 | **Default thresholds: a gesture starts at score ≥ 0.6 and ends below 0.45** (was 0.7 / 0.5 in P1); calibration can set them per gesture | In the recording, Open_Palm scored a median of 0.61 and only 8% of its frames reached 0.7, so at 0.7 it would rarely fire | Low: settings |
| C12 | **Timing (P1, P2):** a gesture must last 0.1 s to start and be gone 0.1 s to end; 0.2 s cooldown per hand; a hand can vanish for 0.15 s without losing its state; `gesture_hold` every 0.25 s; one active gesture per hand | The values proposed in P1/P2, now built and covered by unit tests | Low: settings |

## Proposed  not yet confirmed

| # | Proposal | Status |
|---|---|---|
| P3 | **Wake gesture + lock** (hold an open palm for 1 s to claim control) as an **optional helper** a game can turn on | Not confirmed |
| P5 | **Health status** (OK / degraded / bad) from detection confidence and frame brightness, so games can react to bad lighting | Not confirmed |
| P6 | **The game shows who is in control** (e.g. highlights the locked player) | Requirement from research; not confirmed |

## Still open

| # | Question | Notes |
|---|---|---|
| O1 | **What makes a gesture "the best"?** | Proposed: recognized reliably (high score when made) **and** rarely made by accident. Measure each built-in gesture with the replay harness, then pick 3. **First evidence (unlabeled, one person, one camera):** median scores Pointing_Up 0.81, Closed_Fist 0.78, Open_Palm 0.61, Thumb_Up 0.52. The engine uses Closed_Fist, Open_Palm and Pointing_Up for now; the labeled report (step 6) is the real test. |
| O4 | **Body size threshold:** fixed value or set by calibration per venue? | |
| O5 | **Calibration flow:** what the user does (hold each gesture for N seconds?) and what it sets | |
| O7 | **Demo game:** there will be one (User); still open: what it is and which gestures it uses | Must use only the engine's events, never MediaPipe directly. |
| O10 | **Recording consent:** who may be recorded at events (volunteers only, or public sessions with signage)? | User chose "decide later". **Must be settled before the first event recording.** Recordings stay local (gitignored). |
| O9 | **Camera unplugged mid-session:** right now the engine just stops. Should it keep retrying and report `camera_lost` to games (part of P5 health)? | Found in step 1 testing: unplugging the webcam ends the loop (`errno=19 No such device`). A loose USB cable at an event would freeze the game silently. |
| O8 | **Packaging / install:** how people get the engine onto a laptop (e.g. `pip install`, or a single executable for Windows) | Not about GPU; GPU is settled in D8. Low priority for the MVP. |

## Hopes, not yet evidence

These need measuring with the replay harness before we rely on them.

- The built-in GestureRecognizer is accurate enough for event use.
- Pose + Hands together run fast enough on a Windows laptop CPU.
- Body size separates players from people behind them well enough.
- Lighting fixes (camera exposure lock, contrast correction) actually help MediaPipe.
