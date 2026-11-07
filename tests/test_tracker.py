from helpers import body, detection, hand

from mc_gesture.config import Config
from mc_gesture.players.tracker import PlayerTracker


def run(tracker, frames, dt=1 / 30, start=0.0):
    out = None
    for i, d in enumerate(frames):
        out = tracker.update(d, start + i * dt)
    return out


def test_far_body_is_a_bystander():
    out = PlayerTracker(Config()).update(detection([body(0.5, 0.4, 0.2), body(0.2, 0.3, 0.05)]), 0.0)
    assert [p.id for p in out.players] == [1]
    assert len(out.bystanders) == 1


def test_hand_links_to_nearest_wrist_and_stray_hand_is_dropped():
    b = body(0.5, 0.4, 0.2)
    left_wrist = b[15]
    out = PlayerTracker(Config()).update(
        detection([b], [hand(left_wrist[0] + 0.01, left_wrist[1]), hand(0.95, 0.05)]), 0.0
    )
    player = out.players[0]
    assert list(player.hands) == ["left"]
    assert len(out.dropped_hands) == 1


def test_bystanders_hand_is_dropped():
    far = body(0.2, 0.3, 0.05)
    out = PlayerTracker(Config()).update(detection([body(0.6, 0.4, 0.2), far], [hand(*far[15][:2])]), 0.0)
    assert out.players[0].hands == {}
    assert len(out.dropped_hands) == 1


def test_two_players_keep_their_ids_while_moving():
    tracker = PlayerTracker(Config())
    frames = [detection([body(0.3 + i * 0.002, 0.4, 0.2), body(0.7 - i * 0.002, 0.4, 0.2)]) for i in range(30)]
    out = run(tracker, frames)
    assert sorted(p.id for p in out.players) == [1, 2]
    assert out.players[0].center[0] > 0.3 and out.players[1].center[0] < 0.7


def test_short_size_glitch_keeps_the_player():
    tracker = PlayerTracker(Config())
    frames = [detection([body(0.5, 0.4, 0.2)])] * 20 + [detection([body(0.5, 0.4, 0.04)])] * 15
    frames += [detection([body(0.5, 0.4, 0.2)])] * 5
    out = run(tracker, frames)
    assert [p.id for p in out.players] == [1]


def test_player_leaves_after_grace_period():
    tracker = PlayerTracker(Config())
    run(tracker, [detection([body(0.5, 0.4, 0.2)])] * 5)
    out = tracker.update(detection(), 0.3)
    assert [p.id for p in out.players] == [1]  # still within the grace period
    out = tracker.update(detection(), 1.0)
    assert out.players == []
    assert out.events == [{"type": "player_left", "player": 1}]
