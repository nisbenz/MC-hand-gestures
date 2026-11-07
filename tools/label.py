"""Label a recording by hand: mark when each gesture was really made (ground truth).

    uv run python tools/label.py recordings/<session>

Keys:  space  play / pause          a / d  step back / forward one frame
       1 2 3  start or stop a label for gesture 1, 2, 3 (shown on screen)
       u      undo the last label   s  save   q  save and quit

Press a number when the gesture starts and again when it ends. Labels are saved to
<session>/labels.csv and used by tools/report.py.
"""

import argparse
import csv
from pathlib import Path

import cv2

from mc_gesture.config import Config
from mc_gesture.record.labels import Label, load, save
from mc_gesture.viewer import text

WINDOW = "MC-Gesture labels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--config", type=Path, help="settings JSON (for the gesture list)")
    args = parser.parse_args()

    gestures = list(Config.load(args.config).gestures)
    with open(args.session / "frames.csv", newline="") as f:
        times = [float(r["t"]) for r in csv.DictReader(f)]
    cap = cv2.VideoCapture(str(args.session / "video.avi"))
    labels = load(args.session) or []
    open_labels: dict[str, float] = {}
    index, playing, image = 0, False, None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        return frame if ok else None

    image = read(0)
    while True:
        t = times[index]
        view = cv2.flip(image, 1)  # mirrored, like the player saw themselves
        text(view, f"t {t:6.2f} s   frame {index}/{len(times) - 1}   {'PLAY' if playing else 'PAUSED'}", (10, 30))
        for n, g in enumerate(gestures, 1):
            state = f"recording since {open_labels[g]:.2f}" if g in open_labels else ""
            done = sum(1 for l in labels if l.gesture == g)
            text(view, f"[{n}] {g}: {done} labels  {state}", (10, 60 + n * 28),
                 (80, 220, 80) if g in open_labels else (255, 255, 255))
        current = [l.gesture for l in labels if l.start <= t <= l.end]
        if current:
            text(view, "labeled here: " + ", ".join(current), (10, view.shape[0] - 20), (80, 200, 255))
        text(view, "space play  a/d step  1-3 label  u undo  s save  q quit", (10, view.shape[0] - 50), (200, 200, 200))
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(int(1000 * (times[min(index + 1, len(times) - 1)] - t)) or 1 if playing else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("a"), ord("d")):
            playing = False
            index = max(0, min(len(times) - 1, index + (1 if key == ord("d") else -1)))
            image = read(index)
            continue
        elif ord("1") <= key < ord("1") + len(gestures):
            g = gestures[key - ord("1")]
            if g in open_labels:
                start = open_labels.pop(g)
                labels.append(Label(min(start, t), max(start, t), g))
            else:
                open_labels[g] = t
        elif key == ord("u") and labels:
            labels.pop()
        elif key == ord("s"):
            print("saved", save(args.session, labels))

        if playing:
            if index + 1 >= len(times):
                playing = False
            else:
                index += 1
                ok, frame = cap.read()
                image = frame if ok else image

    path = save(args.session, labels)
    print(f"saved {len(labels)} labels to {path}" + (f" (unfinished: {list(open_labels)})" if open_labels else ""))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
