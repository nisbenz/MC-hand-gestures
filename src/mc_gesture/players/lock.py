"""Optional lock helper (P3): one player claims control by holding the lock gesture.

Off by default (config.lock = false). When on:
  - until someone is locked, only player_joined / player_left get through;
  - holding config.lock_gesture for lock_hold_s locks that player (player_locked);
  - then only the locked player's events and cursor get through;
  - when the locked player leaves, the lock is released (player_unlocked).
"""

from mc_gesture.config import Config

LIFECYCLE = {"player_joined", "player_left", "zone_changed"}


class PlayerLock:
    def __init__(self, config: Config):
        self.config = config
        self.locked: int | None = None
        self._holding: dict[tuple[int, str], float] = {}  # (player, hand) -> when the lock gesture started

    def update(self, events: list[dict], t: float) -> list[dict]:
        out = []
        for e in events:
            if e["type"] == "gesture_start" and e["gesture"] == self.config.lock_gesture:
                self._holding[(e["player"], e["hand"])] = t
            elif e["type"] == "gesture_end" and e["gesture"] == self.config.lock_gesture:
                self._holding.pop((e["player"], e["hand"]), None)
            elif e["type"] in ("player_left", "hand_lost"):
                for key in [k for k in self._holding if k[0] == e["player"] and e.get("hand", k[1]) == k[1]]:
                    del self._holding[key]
            if e["type"] == "player_left" and e["player"] == self.locked:
                self.locked = None
                out.append(e)
                out.append({"type": "player_unlocked", "player": e["player"]})
                continue
            if e["type"] in LIFECYCLE or (self.locked is not None and e.get("player") == self.locked):
                out.append(e)

        if self.locked is None:
            ready = [key for key, since in self._holding.items() if t - since >= self.config.lock_hold_s]
            if ready:
                self.locked = min(ready, key=lambda k: self._holding[k])[0]  # whoever started first
                self._holding.clear()
                out.append({"type": "player_locked", "player": self.locked})
        return out

    def allows(self, player: int) -> bool:
        return self.locked == player
