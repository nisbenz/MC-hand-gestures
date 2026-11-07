"""Debug window: draws what the engine sees on top of the camera image.

Started by `uv run mc-gesture-view` or `uv run mc-gesture --view` (see run.py).
"""

import cv2
from mediapipe.tasks.python import vision

from mc_gesture.engine import Output

HAND_LINKS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
POSE_LINKS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
WINDOW = "MC-Gesture"
PLAYER_COLORS = [(80, 200, 255), (255, 160, 60), (120, 255, 120), (255, 100, 220)]
GREY, RED = (110, 110, 110), (60, 60, 230)


def text(image, label, origin, color=(255, 255, 255), scale=0.65):
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.7 else 2)


class Drawer:
    def __init__(self, image, mirror: bool):
        self.image, self.mirror = image, mirror
        self.h, self.w = image.shape[:2]

    def px(self, point):
        x = 1 - point[0] if self.mirror else point[0]
        return int(x * self.w), int(point[1] * self.h)

    def body(self, body, color, thickness=2):
        for a, b in POSE_LINKS:
            if body[a][2] > 0.5 and body[b][2] > 0.5:
                cv2.line(self.image, self.px(body[a]), self.px(body[b]), color, thickness)

    def hand(self, hand, color, label=None):
        for a, b in HAND_LINKS:
            cv2.line(self.image, self.px(hand.landmarks[a]), self.px(hand.landmarks[b]), color, 2)
        if label:
            x, y = self.px(hand.landmarks[0])
            text(self.image, label, (x - 40, y + 30), color)


def draw(view, out: Output, mirror: bool, active=lambda player, side: None) -> None:
    d = Drawer(view, mirror)
    for body in out.players.bystanders:
        d.body(body, GREY, 1)
    for hand in out.players.dropped_hands:
        d.hand(hand, RED, "dropped")
    for player in out.players.players:
        color = PLAYER_COLORS[(player.id - 1) % len(PLAYER_COLORS)]
        d.body(player.body, color)
        x, y = d.px(player.body[0])
        tag = f"P{player.id}" + (f" [{player.zone}]" if player.zone else "")
        text(view, tag, (x - 20, y - 30), color, 0.9)
        for side, hand in player.hands.items():
            label = f"{side[0].upper()} {hand.gesture} {hand.gesture_score:.2f}"
            if gesture := active(player.id, side):
                label = f">> {gesture} <<  ({hand.gesture_score:.2f})"  # the event the game sees
            d.hand(hand, color, label)
    # smoothed cursors (what the cursor layer sends), in output coordinates
    for c in out.cursors:
        x, y = int(c["x"] * d.w), int(c["y"] * d.h)  # the view is mirrored the same way as the output
        cv2.circle(view, (x, y), 9, (255, 255, 255), 2)


class Viewer:
    def __init__(self, mirror: bool):
        self.mirror = mirror
        # GUI_NORMAL drops the Qt toolbar and status bar. The status bar redraws pixel
        # coordinates on every mouse move, and with OpenCV's font-less Qt build each redraw
        # prints a "QFont::setPointSizeF" warning.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    def show(self, image_bgr, out: Output, active, lines: list[str], badge: str | None) -> bool:
        """Draws one frame. Returns False when the user pressed q."""
        view = cv2.flip(image_bgr, 1) if self.mirror else image_bgr.copy()
        draw(view, out, self.mirror, active)
        for i, line in enumerate(lines):
            text(view, line, (10, 28 + i * 26))
        if badge == "REC":
            cv2.circle(view, (view.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            text(view, "REC", (view.shape[1] - 90, 38), (0, 0, 255), 0.7)
        elif badge:
            text(view, badge, (view.shape[1] - 200, 38), (255, 200, 0), 0.7)
        cv2.imshow(WINDOW, view)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
