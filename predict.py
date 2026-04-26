"""
predict.py — run blood_detector_model.pt on every image in test_images/
and save annotated outputs to test_predictions/.

Color scheme (BGR drawn by OpenCV):
  - RBC                                  -> red
  - Platelets                            -> green
  - any WBC subtype (Neutrophil /
    Lymphocyte / Monocyte / Eosinophil /
    Basophil)                            -> blue

Run:

    python predict.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
PT_PATH = ROOT / "blood_detector_model.pt"
TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "test_predictions"

CONF = 0.15
IOU = 0.7
IMGSZ = 640

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}
COLOR_RBC = (0, 0, 255)
COLOR_PLATELETS = (0, 200, 0)
COLOR_WBC = (255, 80, 0)


def color_for(name: str) -> tuple[int, int, int]:
    if name == "RBC":
        return COLOR_RBC
    if name == "Platelets":
        return COLOR_PLATELETS
    if name in WBC_SUBTYPES:
        return COLOR_WBC
    return (200, 200, 200)


def annotate(img, boxes_xyxy, classes, names) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.4, 1, 1
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
        name = names[int(c)]
        col = color_for(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        ly = y1 - 2
        if ly - th - 2 < 0:
            ly = y1 + th + 4
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 2, ly + 1), col, -1)
        cv2.putText(img, name, (x1 + 1, ly - 1), font, fs,
                    (255, 255, 255), ft, cv2.LINE_AA)


def main() -> None:
    if not PT_PATH.exists():
        raise FileNotFoundError(f"{PT_PATH.name} not found in {ROOT}")
    OUT_DIR.mkdir(exist_ok=True)
    images = sorted(p for p in TEST_DIR.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"no images found in {TEST_DIR}")
        return

    from ultralytics import YOLO
    model = YOLO(str(PT_PATH))
    results = model.predict(
        source=[str(p) for p in images],
        conf=CONF, iou=IOU, imgsz=IMGSZ,
        save=False, verbose=False,
    )
    for img_path, r in zip(images, results):
        img = cv2.imread(str(img_path))
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        annotate(img, boxes, classes, r.names)
        out = OUT_DIR / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        counts = Counter(r.names[int(c)] for c in classes)
        print(f"  {img_path.name:20s} -> {out.name:28s} | {len(boxes):3d} boxes  {dict(counts)}")


if __name__ == "__main__":
    main()
