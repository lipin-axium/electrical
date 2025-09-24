from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Tuple

import cv2


def load_names(names_path: Path) -> List[str]:
    if names_path.exists():
        return [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ["Symbol A", "Symbol B"]


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


def resolve_label_path(img_path: Path, img_root: Path, labels_root: Path) -> Path | None:
    """Map an image path to its expected label path.

    - Mirrors folder structure under labels
    - Supports augmented filenames by stripping suffix starting with '__'
    - Searches across labels/train, labels/val, labels/test if needed
    """
    candidates: List[Path] = []

    try:
        rel = img_path.relative_to(img_root)
    except Exception:
        rel = Path(img_path.name)

    # derive potential stems
    stems = [rel.stem, re.sub(r"__.*$", "", rel.stem)]
    stems = [s for i, s in enumerate(stems) if s and s not in stems[:i]]

    # search in provided labels_root and its siblings train/val/test
    label_bases = [labels_root]
    parent = labels_root.parent
    for sub in ["train", "val", "test"]:
        p = parent / sub
        if p not in label_bases:
            label_bases.append(p)

    for base in label_bases:
        # same subfolder structure if exists
        rel_dir = rel.parent if rel.parent != Path(".") else Path()
        for stem in stems:
            candidates.append((base / rel_dir / f"{stem}.txt").resolve())
            # also try without rel_dir (flat)
            candidates.append((base / f"{stem}.txt").resolve())

    for c in candidates:
        if c.exists():
            return c

    # Print first few attempted for diagnostics
    print(f"Label not found for {img_path.name}. Tried:")
    for c in candidates[:5]:
        print(f"  - {c}")
    return None


def draw_for_pair(img_path: Path, label_path: Path | None, names_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to read image: {img_path}"
    h, w = img.shape[:2]

    names = load_names(names_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    drawn = 0
    if label_path and label_path.exists():
        for ln, raw in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping malformed line {label_path.name}:{ln}: {raw}")
                continue
            try:
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])
            except Exception:
                print(f"Skipping unparsable line {label_path.name}:{ln}: {raw}")
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            color = (0, 255, 0) if cls == 0 else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names[cls] if 0 <= cls < len(names) else str(cls)
            ((tw, th), _) = cv2.getTextSize(label, font, 0.6, 1)
            y_top = max(0, y1 - th - 6)
            cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
            cv2.putText(img, label, (x1 + 3, y_top + th + 1), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            drawn += 1
    else:
        print(f"No label for {img_path.name}")

    if drawn == 0:
        print(f"No boxes drawn for {img_path.name}. Checked label: {label_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(out_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    img_dir = root / "datasets" / "final" / "images" / "train"
    labels_root = root / "datasets" / "final" / "labels" / "train"
    names_path = root / "labels" / "obj.names"
    out_dir = root / "runs" / "labels_viz"

    images = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    assert images, f"No images found in {img_dir}"

    for img_path in images:
        label_path = resolve_label_path(img_path, img_dir, labels_root)
        out_rel = img_path.relative_to(img_dir)
        out_path = out_dir / out_rel.with_suffix(".jpg")
        draw_for_pair(img_path, label_path, names_path, out_path)


if __name__ == "__main__":
    main()


