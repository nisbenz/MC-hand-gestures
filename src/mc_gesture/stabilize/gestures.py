"""Stabilize (P1, P2): turns per-frame gesture guesses into start / hold / end events.

Per hand (player id + side), at most one gesture is active at a time:

    idle ──score ≥ enter for min_hold_s──► active ──score < exit for release_s──► idle
                                             │                                 (+ cooldown)
                                             └── every hold_interval_s: gesture_hold

A hand that disappears keeps its state for hand_grace_s (short dropouts);
after that an active gesture ends with reason "lost" and hand_lost is sent.
"""

from dataclasses import dataclass

from mc_gesture.config import Config
from mc_gesture.players.tracker import Player


@dataclass
class HandState:
    active: str | None = None
    active_since: float = 0.0
    last_hold: float = 0.0
    candidate: str | None = None
    candidate_since: float = 0.0
    below_since: float | None = None  # when the active gesture's score first dropped below exit
    cooldown_until: float = 0.0
    last_seen: float = 0.0
    score: float = 0.0


class GestureStabilizer:
    def __init__(self, config: Config):
        self.config = config
        self._hands: dict[tuple[int, str], HandState] = {}

    def active(self, player: int, side: str) -> str | None:
        state = self._hands.get((player, side))
        return state.active if state else None

    def update(self, players: list[Player], t: float) -> list[dict]:
        events: list[dict] = []
        cfg = self.config
        present = set()

        for player in players:
            for side, hand in player.hands.items():
                key = (player.id, side)
                present.add(key)
                state = self._hands.setdefault(key, HandState(last_seen=t))
                state.last_seen = t
                gesture = hand.gesture if hand.gesture in cfg.gestures else None
                score = hand.gesture_score if gesture else 0.0
                self._step(key, state, gesture, score, t, events)

        player_ids = {p.id for p in players}
        for key, state in list(self._hands.items()):
            if key in present:
                continue
            gone_player = key[0] not in player_ids
            if gone_player or t - state.last_seen > cfg.hand_grace_s:
                if state.active:
                    events.append(self._event("gesture_end", key, state.active, t, held_s=t - state.active_since,
                                              reason="lost"))
                events.append({"type": "hand_lost", "player": key[0], "hand": key[1]})
                del self._hands[key]
        return events

    def _step(self, key, state: HandState, gesture: str | None, score: float, t: float, events: list[dict]) -> None:
        cfg = self.config
        if state.active:
            _, exit_ = cfg.enter_exit(state.active)
            if gesture == state.active and score >= exit_:
                state.below_since = None
                state.score = score
                if t - state.last_hold >= cfg.hold_interval_s:
                    events.append(self._event("gesture_hold", key, state.active, t, held_s=t - state.active_since))
                    state.last_hold = t
                return
            if state.below_since is None:
                state.below_since = t
            if t - state.below_since >= cfg.release_s:
                events.append(self._event("gesture_end", key, state.active, t, held_s=t - state.active_since,
                                          reason="released"))
                state.active, state.below_since = None, None
                state.cooldown_until = t + cfg.cooldown_s
                state.candidate = None
            return

        enter, _ = cfg.enter_exit(gesture) if gesture else (1.0, 1.0)
        if gesture is None or score < enter or t < state.cooldown_until:
            state.candidate = None
            return
        if state.candidate != gesture:
            state.candidate, state.candidate_since = gesture, t
        if t - state.candidate_since >= cfg.min_hold_s:
            state.active, state.active_since, state.last_hold = gesture, t, t
            state.score, state.candidate = score, None
            events.append(self._event("gesture_start", key, gesture, t, score=round(score, 3)))

    @staticmethod
    def _event(kind: str, key, gesture: str, t: float, **extra) -> dict:
        event = {"type": kind, "player": key[0], "hand": key[1], "gesture": gesture}
        for name, value in extra.items():
            event[name] = round(value, 3) if isinstance(value, float) else value
        return event
