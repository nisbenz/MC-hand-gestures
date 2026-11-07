"""Player filter (D3): turns raw bodies and hands into a stable list of players.

1. Bodies keep the same player id from frame to frame (nearest match).
2. Size gate: a new body must have wide enough shoulders to join (narrow = far
   away = bystander). An existing player only leaves when their median size over
   the last second drops below half of that, because Pose sometimes misjudges a
   body's shoulders for up to a second (seen in testing).
3. Each hand is linked to the nearest wrist of a player's body; hands with no
   owner (stray detections, bystanders' hands) are dropped.
"""

import statistics
from collections import deque
from dataclasses import dataclass, field

from mc_gesture.config import Config
from mc_gesture.detect.detector import Detection, Hand

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
VISIBLE = 0.5


@dataclass
class Player:
    id: int
    body: list[tuple[float, float, float]]
    size: float                      # shoulder width / frame width
    center: tuple[float, float]      # shoulder midpoint, camera coordinates
    last_seen: float
    hands: dict[str, Hand] = field(default_factory=dict)  # "left"/"right" = the body's own side
    zone: str | None = None
    sizes: deque = field(default_factory=lambda: deque(maxlen=30))  # ~1 s of shoulder widths


@dataclass
class Frame:
    players: list[Player]
    bystanders: list[list[tuple[float, float, float]]]  # bodies dropped as too small
    dropped_hands: list[Hand]                             # hands with no player
    events: list[dict]


def _dist(a, b, aspect: float) -> float:
    """Distance in frame-width units (y is scaled so both axes match)."""
    return ((a[0] - b[0]) ** 2 + ((a[1] - b[1]) / aspect) ** 2) ** 0.5


class PlayerTracker:
    def __init__(self, config: Config):
        self.config = config
        self._players: dict[int, Player] = {}
        self._next_id = 1

    def update(self, detection: Detection, t: float) -> Frame:
        aspect = detection.aspect
        events: list[dict] = []

        candidates, bystanders = [], []
        for body in detection.poses:
            ls, rs = body[L_SHOULDER], body[R_SHOULDER]
            if ls[2] < VISIBLE or rs[2] < VISIBLE:
                bystanders.append(body)  # can't measure it, so it can't be a player
                continue
            center = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            candidates.append((body, _dist(ls, rs, aspect), center))

        for player in self._players.values():
            player.hands = {}  # a player only has the hands linked this frame

        # 1. keep ids stable: greedy nearest-center matching, closest pairs first
        pairs = sorted(
            (_dist(center, p.center, aspect), i, pid)
            for i, (_, size, center) in enumerate(candidates)
            for pid, p in self._players.items()
            if _dist(center, p.center, aspect) < max(size, p.size)
        )
        assigned_bodies, assigned_ids = {}, set()
        for _, i, pid in pairs:
            if i not in assigned_bodies and pid not in assigned_ids:
                assigned_bodies[i] = pid
                assigned_ids.add(pid)

        # 2. size gate
        seen = []
        min_size = self.config.min_body_size
        for i, (body, size, center) in enumerate(candidates):
            if i in assigned_bodies:
                player = self._players[assigned_bodies[i]]
                player.sizes.append(size)
                player.size = max(statistics.median(player.sizes), 1e-6)
                player.body, player.center = body, center
                if player.size < min_size * self.config.leave_factor:
                    bystanders.append(body)  # walked away; leaves after the grace period
                    continue
                player.last_seen = t
            elif size >= min_size:
                player = Player(self._next_id, body, size, center, t)
                player.sizes.append(size)
                self._players[player.id] = player
                self._next_id += 1
                events.append({"type": "player_joined", "player": player.id})
            else:
                bystanders.append(body)
                continue
            seen.append(player)

        for pid in [pid for pid, p in self._players.items() if t - p.last_seen > self.config.player_grace_s]:
            del self._players[pid]
            events.append({"type": "player_left", "player": pid})

        # 3. link hands to the nearest wrist of a player seen this frame
        links = sorted(
            (_dist(hand.landmarks[0], player.body[wrist], aspect) / player.size, h, player.id, side)
            for h, hand in enumerate(detection.hands)
            for player in seen
            for wrist, side in ((L_WRIST, "left"), (R_WRIST, "right"))
        )
        by_id = {p.id: p for p in seen}
        used_hands = set()
        for dist, h, pid, side in links:
            if dist > self.config.hand_match:
                break
            player = by_id[pid]
            if h in used_hands or side in player.hands:
                continue
            player.hands[side] = detection.hands[h]
            used_hands.add(h)
        dropped = [hand for h, hand in enumerate(detection.hands) if h not in used_hands]

        players = sorted(self._players.values(), key=lambda p: p.id)
        return Frame(players, bystanders, dropped, events)
