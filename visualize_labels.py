from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import cv2


def load_names(names_path: Path) -> List[str]:
    if names_path.exists():
        return [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ["class0", "class1"]


def yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    # Clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def draw_for_pair(img_path: Path, label_path: Path, names_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to read image: {img_path}"
    h, w = img.shape[:2]

    names = load_names(names_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if label_path.exists():
        for ln, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping malformed line {label_path.name}:{ln}: {line}")
                continue
            cls = int(float(parts[0]))
            x, y, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            color = (255, 0, 0) if cls == 0 else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names[cls] if 0 <= cls < len(names) else str(cls)
            ((tw, th), _) = cv2.getTextSize(label, font, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(out_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    # Use first training image as default example
    img_dir = root / "datasets" / "clip" / "images" / "train"
    lbl_dir = root / "datasets" / "clip" / "labels" / "train"
    names_path = root / "labels" / "obj.names"
    out_dir = root / "runs" / "labels_viz"

    images = sorted(img_dir.glob("*.jpg"))
    assert images, f"No images found in {img_dir}"
    img_path = images[0]
    label_path = lbl_dir / f"{img_path.stem}.txt"
    out_path = out_dir / f"{img_path.stem}.jpg"

    draw_for_pair(img_path, label_path, names_path, out_path)


if __name__ == "__main__":
    main()


