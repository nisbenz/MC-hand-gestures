# MC-Gesture: Decision Log

A reusable local engine that turns webcam input into stable gesture events for games and interactive displays.

**MVP scope (from the brief):** one camera, three gestures, one demo game, and a measurable latency/error report.

Each entry records who made the decision and why. "User" = project owner; "Claude" = proposed by the assistant and not yet confirmed.

---

## Decided

| # | Decision | By | Why |
|---|---|---|---|
| D1 | **Python engine**, consumed by games over a network protocol | User | Any program in any language can consume it. (The protocol gives compatibility, not Python itself; Python gives MediaPipe, replay and analysis.) |
| D2 | **The engine does all the cleaning.** Games receive simple events and interpret them however they want. Games never clean raw MediaPipe data. | User | The point of the project: no vision or pre-processing code inside each game. |
| D3 | **Bystander handling = MediaPipe Pose + simple rules.** Pose finds bodies, hands are matched to bodies, and rules (body size, optional zone, optional lock) pick the players. | User | Matches common practice in Kinect installations and public-display research: the model finds people, rules decide who plays. This is a starting bet, not a proven optimum; it will be measured. |
| D5 | **Optional zones defined by the app.** The default is no zone (the whole frame). | User | Lets games such as split-screen PvP label players by area without hard-coding that in the engine. |
| D6 | **Smoothing with the `OneEuroFilter` package.** Internals are delegated; the two tuning settings are known: `min_cutoff` (less shaking ↔ more lag) and `beta` (faster catch-up on quick moves). | User | A published, well-known method beats a from-scratch filter. |
| D7 | **Output layers:** `events` (gesture start/hold/end, hand lost), `cursor` (smoothed position), `landmarks` (optional cleaned points). Games subscribe to what they need. | User | Most games only need events; landmarks give freedom to the rare game that wants more. |
| D8 | **GPU auto-detect with CPU fallback.** Try GPU first and fall back to CPU if it fails. Report which one is running. **Latency must be acceptable on CPU.** | User | Club laptops are a mix of Linux, macOS and Windows. MediaPipe's Python GPU option only works on Linux/macOS, so the Windows laptops will run on CPU. |
| D9 | **Classifier = MediaPipe's built-in GestureRecognizer**, behind a swappable interface (camera image in → gesture scores out). | User | The brief needs three gestures, not custom ones; the built-in set covers that. Tuning against false activations happens in our stabilize stage either way. Switch to custom geometric rules only if replay shows problems that threshold tuning can't fix. |
| D10 | **Transport = WebSocket** | User | Works with web pages and game engines. OSC can come later if installation tools need it. |
| D11 | **Zones support both modes**, filter (drop players outside every zone) and label (report the zone, drop nobody), chosen by the app | User | Gives the app freedom. |
| D12 | **Gestures = the 3 most reliable built-in gestures**, chosen by measurement | User | "Most reliable" still needs a definition and a test; see O1. |
| D13 | **Report targets (starting values):** median latency ≤ 100 ms on a CPU laptop; false activations ≤ 1 per 5 min of play; miss rate ≤ 5% | User (values suggested by Claude) | These give the report pass/fail lines. They're starting guesses, not standards; revisit after the first replay measurements. |
| D14 | **Build order:** (1) camera + GestureRecognizer + Pose on screen with timings → (2) recorder + replay → (3) player filter → (4) stabilize → events → (5) WebSocket + layers (cursor smoothing here) → (6) report → (7) calibration → (8) demo game → (9) zones, lock, health | User (order proposed by Claude) | Each step ends with something runnable. Recording comes early so every later step is tested on the same footage. |
| D15 | **Record and replay harness** (was P4): record the **camera video** plus timestamps, label ground truth, rerun the engine offline to measure false activations, misses and latency | User | Required to check D12 and D13. Recording video (not only results) lets us rerun with different settings, such as GPU vs CPU or lighting fixes. |

## Proposed  not yet confirmed

| # | Proposal | Status |
|---|---|---|
| P1 | **Stabilize stage:** separate enter/exit thresholds, minimum hold time (~100 ms), cooldown, one active gesture at a time | Accepted implicitly; confirm the numbers during the build |
| P2 | **Tracking-loss grace period** (~150 ms): keep the last state through short dropouts before sending `hand_lost` | Not discussed directly |
| P3 | **Wake gesture + lock** (hold an open palm for 1 s to claim control) as an **optional helper** a game can turn on | Not confirmed |
| P5 | **Health status** (OK / degraded / bad) from detection confidence and frame brightness, so games can react to bad lighting | Not confirmed |
| P6 | **The game shows who is in control** (e.g. highlights the locked player) | Requirement from research; not confirmed |

## Still open

| # | Question | Notes |
|---|---|---|
| O1 | **What makes a gesture "the best"?** | Proposed: recognized reliably (high score when made) **and** rarely made by accident. Measure each built-in gesture with the replay harness, then pick 3. |
| O4 | **Body size threshold:** fixed value or set by calibration per venue? | |
| O5 | **Calibration flow:** what the user does (hold each gesture for N seconds?) and what it sets | |
| O7 | **Demo game:** there will be one (User); still open: what it is and which gestures it uses | Must use only the engine's events, never MediaPipe directly. |
| O8 | **Packaging / install:** how people get the engine onto a laptop (e.g. `pip install`, or a single executable for Windows) | Not about GPU; GPU is settled in D8. Low priority for the MVP. |

## Hopes, not yet evidence

These need measuring with the replay harness before we rely on them.

- The built-in GestureRecognizer is accurate enough for event use.
- Pose + Hands together run fast enough on a Windows laptop CPU.
- Body size separates players from people behind them well enough.
- Lighting fixes (camera exposure lock, contrast correction) actually help MediaPipe.
