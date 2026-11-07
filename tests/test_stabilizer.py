from helpers import hand

from mc_gesture.config import Config
from mc_gesture.players.tracker import Player
from mc_gesture.stabilize.gestures import GestureStabilizer

DT = 1 / 30


def player(gesture=None, score=0.0, present=True):
    p = Player(id=1, body=[], size=0.2, center=(0.5, 0.5), last_seen=0.0)
    if present:
        p.hands["right"] = hand(0.5, 0.5, gesture or "None", score)
    return p


def feed(stab, frames, start=0.0):
    """frames: list of (gesture, score) or None for 'hand missing'. Returns (t, event) pairs."""
    out = []
    for i, f in enumerate(frames):
        t = start + i * DT
        p = player(present=False) if f is None else player(*f)
        out += [(round(t, 3), e) for e in stab.update([p], t)]
    return out


def types(events):
    return [e["type"] for _, e in events]


def test_brief_flicker_does_not_start_a_gesture():
    # 2 frames ≈ 67 ms, shorter than min_hold_s (100 ms)
    events = feed(GestureStabilizer(Config()), [("Closed_Fist", 0.9)] * 2 + [(None, 0)] * 10)
    assert events == []


def test_held_gesture_starts_holds_and_ends():
    frames = [("Closed_Fist", 0.9)] * 30 + [(None, 0)] * 10
    events = feed(GestureStabilizer(Config()), frames)
    assert types(events)[0] == "gesture_start"
    assert "gesture_hold" in types(events)
    assert types(events)[-1] == "gesture_end"
    assert events[-1][1]["reason"] == "released"


def test_low_score_never_starts():
    events = feed(GestureStabilizer(Config()), [("Closed_Fist", 0.55)] * 30)
    assert events == []


def test_hysteresis_keeps_gesture_between_exit_and_enter():
    # starts at 0.9, then wobbles at 0.5: above exit (0.45) so it must stay active
    frames = [("Closed_Fist", 0.9)] * 6 + [("Closed_Fist", 0.5)] * 30
    events = feed(GestureStabilizer(Config()), frames)
    assert "gesture_end" not in types(events)


def test_short_dropout_is_bridged():
    # hand missing for 2 frames (67 ms) < hand_grace_s (150 ms)
    frames = [("Open_Palm", 0.9)] * 10 + [None] * 2 + [("Open_Palm", 0.9)] * 10
    events = feed(GestureStabilizer(Config()), frames)
    assert types(events).count("gesture_start") == 1
    assert "hand_lost" not in types(events)


def test_long_dropout_ends_with_lost():
    frames = [("Open_Palm", 0.9)] * 10 + [None] * 10
    events = feed(GestureStabilizer(Config()), frames)
    assert types(events)[-2:] == ["gesture_end", "hand_lost"]
    assert events[-2][1]["reason"] == "lost"


def test_unlisted_gesture_is_ignored():
    events = feed(GestureStabilizer(Config()), [("Victory", 0.95)] * 30)
    assert events == []


def test_cooldown_blocks_immediate_restart():
    cfg = Config()
    frames = [("Closed_Fist", 0.9)] * 10 + [(None, 0)] * 4 + [("Closed_Fist", 0.9)] * 3
    events = feed(GestureStabilizer(cfg), frames)
    assert types(events).count("gesture_start") == 1


def test_calibrated_thresholds_override_defaults():
    cfg = Config(thresholds={"Closed_Fist": [0.5, 0.3]})
    events = feed(GestureStabilizer(cfg), [("Closed_Fist", 0.6)] * 10)
    assert types(events)[0] == "gesture_start"
