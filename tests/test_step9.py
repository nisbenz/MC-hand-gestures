import numpy as np
from helpers import body, detection

from mc_gesture.config import Config
from mc_gesture.health import Health
from mc_gesture.players.lock import PlayerLock
from mc_gesture.players.tracker import PlayerTracker

HALVES = {"left": [0.0, 0.0, 0.5, 1.0], "right": [0.5, 0.0, 1.0, 1.0]}


# --- zones (D5, D11) ---

def test_zones_label_players_in_output_coordinates():
    # camera x 0.25 is mirrored to output x 0.75 -> "right" half
    out = PlayerTracker(Config(zones=HALVES)).update(detection([body(0.25, 0.4, 0.2), body(0.75, 0.4, 0.2)]), 0.0)
    zones = {p.center[0]: p.zone for p in out.players}
    assert zones == {0.25: "right", 0.75: "left"}
    assert sorted(e["zone"] for e in out.events if e["type"] == "zone_changed") == ["left", "right"]


def test_filter_mode_leaves_out_players_outside_every_zone():
    cfg = Config(zones={"middle": [0.3, 0.0, 0.7, 1.0]}, zone_mode="filter")
    out = PlayerTracker(cfg).update(detection([body(0.5, 0.4, 0.2), body(0.1, 0.4, 0.15)]), 0.0)
    assert [p.zone for p in out.players] == ["middle"]


def test_label_mode_keeps_everyone():
    cfg = Config(zones={"middle": [0.3, 0.0, 0.7, 1.0]})
    out = PlayerTracker(cfg).update(detection([body(0.5, 0.4, 0.2), body(0.1, 0.4, 0.15)]), 0.0)
    assert sorted(str(p.zone) for p in out.players) == ["None", "middle"]


# --- lock helper (P3) ---

def g(kind, player, gesture="Open_Palm"):
    return {"type": kind, "player": player, "hand": "right", "gesture": gesture}


def test_lock_needs_a_full_hold_and_then_filters_others():
    lock = PlayerLock(Config(lock=True))
    assert lock.update([g("gesture_start", 2)], 0.0) == []           # nothing passes while unlocked
    assert lock.update([], 0.5) == []
    assert lock.update([], 1.05) == [{"type": "player_locked", "player": 2}]
    out = lock.update([g("gesture_start", 1, "Closed_Fist"), g("gesture_start", 2, "Closed_Fist")], 1.5)
    assert [e["player"] for e in out] == [2]


def test_releasing_early_does_not_lock():
    lock = PlayerLock(Config(lock=True))
    lock.update([g("gesture_start", 1)], 0.0)
    lock.update([g("gesture_end", 1)], 0.5)
    assert lock.update([], 2.0) == []


def test_lock_released_when_player_leaves():
    lock = PlayerLock(Config(lock=True))
    lock.update([g("gesture_start", 1)], 0.0)
    lock.update([], 1.1)
    out = lock.update([{"type": "player_left", "player": 1}], 3.0)
    assert [e["type"] for e in out] == ["player_left", "player_unlocked"]
    assert lock.locked is None


# --- health (P5) ---

def frames(health, image, fps, seconds):
    t = 0.0
    while t < seconds:
        health.frame(image, t)
        t += 1 / fps
    return health.report(t, 1)


def test_health_ok_dark_and_slow():
    grey = np.full((36, 64, 3), 120, np.uint8)
    assert frames(Health(), grey, 30, 1.05)["status"] == "ok"
    dark = frames(Health(), np.full((36, 64, 3), 20, np.uint8), 30, 1.05)
    assert dark["status"] == "degraded" and "too dark" in dark["reasons"]
    assert frames(Health(), grey, 5, 1.05)["status"] == "bad"


def test_health_reports_lost_camera():
    h = Health()
    h.camera_ok = False
    message = h.report(0.0, 0) or h.report(1.5, 0)
    assert message["camera"] == "lost" and message["status"] == "bad"


def test_standing_on_a_zone_boundary_does_not_flicker():
    tracker = PlayerTracker(Config(zones=HALVES))
    events = []
    for i in range(30):  # center wobbles across output x = 0.5 every other frame
        x = 0.49 if i % 2 else 0.51
        events += tracker.update(detection([body(x, 0.4, 0.2)]), i / 30).events
    assert len([e for e in events if e["type"] == "zone_changed"]) == 1


def test_moving_to_another_zone_counts_after_the_hold():
    tracker = PlayerTracker(Config(zones=HALVES))
    for i in range(10):
        tracker.update(detection([body(0.7, 0.4, 0.2)]), i / 30)       # output x 0.3: "left"
    events = []
    for i in range(10, 30):
        events += tracker.update(detection([body(0.3, 0.4, 0.2)]), i / 30).events  # "right"
    assert [e["zone"] for e in events if e["type"] == "zone_changed"] == ["right"]


def test_other_hand_releasing_does_not_cancel_the_lock_hold():
    lock = PlayerLock(Config(lock=True))
    left, right = dict(g("gesture_start", 1), hand="left"), g("gesture_start", 1)
    lock.update([left], 0.0)
    lock.update([right], 0.2)
    lock.update([dict(g("gesture_end", 1), hand="left")], 0.9)
    assert lock.update([], 1.25) == [{"type": "player_locked", "player": 1}]
