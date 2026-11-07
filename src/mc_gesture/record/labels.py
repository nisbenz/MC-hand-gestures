"""Ground-truth labels for a recording and scoring of engine events against them (D13, D15).

labels.csv (in the session folder): start,end,gesture — times in seconds since the
first frame, written by tools/label.py while watching the replay.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

EARLY_S = 0.3   # an event may come this much before the label start (labels are hand-timed)


@dataclass
class Label:
    start: float
    end: float
    gesture: str


def load(session: Path) -> list[Label] | None:
    path = Path(session) / "labels.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return [Label(float(r["start"]), float(r["end"]), r["gesture"]) for r in csv.DictReader(f)]


def save(session: Path, labels: list[Label]) -> Path:
    path = Path(session) / "labels.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "gesture"])
        for label in sorted(labels, key=lambda l: l.start):
            writer.writerow([f"{label.start:.3f}", f"{label.end:.3f}", label.gesture])
    return path


@dataclass
class Score:
    hits: list[tuple[Label, float]]   # label, delay from label start to gesture_start (s)
    misses: list[Label]               # labels with no matching gesture_start
    false: list[dict]                 # gesture_start events that match no label


def score(labels: list[Label], events: list[dict]) -> Score:
    """Each label is matched to the first unused gesture_start of the same gesture
    between (start - EARLY_S) and its end. Leftover starts are false activations."""
    starts = sorted((e for e in events if e["type"] == "gesture_start"), key=lambda e: e["t"])
    used, hits, misses = set(), [], []
    for label in sorted(labels, key=lambda l: l.start):
        match = next((i for i, e in enumerate(starts)
                      if i not in used and e["gesture"] == label.gesture
                      and label.start - EARLY_S <= e["t"] <= label.end), None)
        if match is None:
            misses.append(label)
        else:
            used.add(match)
            hits.append((label, starts[match]["t"] - label.start))
    false = [e for i, e in enumerate(starts) if i not in used]
    return Score(hits, misses, false)
