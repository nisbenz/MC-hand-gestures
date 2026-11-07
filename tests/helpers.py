"""Builds fake detections, so engine logic can be tested without a camera."""

from mc_gesture.detect.detector import Detection, Hand


def body(cx: float, cy: float, width: float) -> list[tuple[float, float, float]]:
    """A 33-point pose facing the camera: shoulders `width` apart, wrists below them."""
    points = [(cx, cy, 0.0)] * 33
    points[0] = (cx, cy - width * 0.8, 1.0)              # nose
    points[11] = (cx + width / 2, cy, 1.0)              # left shoulder (on the image's right)
    points[12] = (cx - width / 2, cy, 1.0)              # right shoulder
    points[15] = (cx + width / 2, cy + width * 1.5, 1.0)  # left wrist
    points[16] = (cx - width / 2, cy + width * 1.5, 1.0)  # right wrist
    return points


def hand(x: float, y: float, gesture: str = "None", score: float = 0.0) -> Hand:
    return Hand(landmarks=[(x, y)] * 21, handedness="?", gesture=gesture, gesture_score=score)


def detection(bodies=(), hands=()) -> Detection:
    # aspect 1 keeps the test geometry simple: x and y distances are equal
    return Detection(hands=list(hands), poses=list(bodies), aspect=1.0)
