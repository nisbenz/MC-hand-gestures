"""All engine settings in one place. Calibration (step 7) writes a JSON file that overrides them."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Gestures the engine reports (D12; provisional until measured, see O1)
    gestures: tuple[str, ...] = ("Closed_Fist", "Open_Palm", "Pointing_Up")

    # Player filter (D3)
    min_body_size: float = 0.10     # shoulder width / frame width needed to join; smaller = bystander (O4)
    leave_factor: float = 0.5       # a player leaves when their median size drops below this x min_body_size
    hand_match: float = 0.6         # max hand-to-pose-wrist distance, in shoulder widths
    player_grace_s: float = 0.5     # keep a player through short tracking gaps

    # Stabilize (P1, P2)
    enter_threshold: float = 0.6    # score needed to start a gesture
    exit_threshold: float = 0.45    # score below which a held gesture ends
    thresholds: dict[str, list[float]] = field(default_factory=dict)  # per gesture [enter, exit], from calibration
    min_hold_s: float = 0.1         # gesture must last this long before gesture_start
    release_s: float = 0.1          # and be gone this long before gesture_end
    cooldown_s: float = 0.2         # no new gesture on the same hand right after one ends
    hand_grace_s: float = 0.15      # keep a hand's state through short detection dropouts
    hold_interval_s: float = 0.25   # gesture_hold event period

    # Output
    mirror: bool = True             # report x as seen in a mirror (natural for players facing the screen)
    min_cutoff: float = 1.0         # OneEuroFilter: lower = less jitter, more lag (D6)
    beta: float = 0.05              # OneEuroFilter: higher = catches up faster on quick moves

    # Zones (D5, D11): name -> [x1, y1, x2, y2] in output coordinates (0..1)
    zones: dict[str, list[float]] = field(default_factory=dict)
    zone_mode: str = "label"        # "label" reports the zone; "filter" also drops players outside every zone

    # Lock helper (P3): off unless the app turns it on
    lock: bool = False
    lock_gesture: str = "Open_Palm"
    lock_hold_s: float = 1.0

    def enter_exit(self, gesture: str) -> tuple[float, float]:
        if gesture in self.thresholds:
            enter, exit_ = self.thresholds[gesture]
            return enter, exit_
        return self.enter_threshold, self.exit_threshold

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        config = cls()
        if path:
            for key, value in json.loads(Path(path).read_text()).items():
                if not hasattr(config, key):
                    raise ValueError(f"Unknown setting in {path}: {key}")
                setattr(config, key, tuple(value) if key == "gestures" else value)
        return config

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
