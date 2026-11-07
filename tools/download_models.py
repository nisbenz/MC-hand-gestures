"""Download the MediaPipe model files into models/ (they are not committed)."""

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "gesture_recognizer.task": f"{BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"ok       {name}")
            continue
        print(f"download {name}")
        urllib.request.urlretrieve(url, target)
        print(f"         {target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
