import argparse
import hashlib
import sys
from pathlib import Path
import random

from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def compute_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Return SHA-256 hex digest for the file at file_path."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def discover_images(source_dirs: list[Path]) -> list[Path]:
    """Find image files under given directories, deduplicating by file hash."""
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    for directory in source_dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in supported_ext:
                continue
            try:
                file_hash = compute_sha256(path)
            except Exception:
                continue
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)
            unique_paths.append(path)
    return unique_paths


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        if img.mode == "RGBA":
            return img.convert("RGB")
        return img
    return img.convert("RGB")


def augment_image(img: Image.Image) -> dict[str, Image.Image]:
    """Return a dict of operation name -> augmented PIL Image.
    
    Operations:
      - r90   : rotate 90° CCW
      - r180  : rotate 180°
      - r270  : rotate 270° CCW (90° CW)
      - flip_lr: left-right mirror
      - flip_ud: upside-down flip
      - blur  : Gaussian blur
      - sharpen: sharpen filter
      - bright: brightness increase
      - dark  : brightness decrease
      - contrast: contrast increase
      - sat   : saturation increase
      - desat : saturation decrease
    """
    img = ImageOps.exif_transpose(img)
    rgb = ensure_rgb(img)
    aug: dict[str, Image.Image] = {}
    
    # Geometric transforms
    aug["r90"] = rgb.rotate(90, expand=True)
    aug["r180"] = rgb.rotate(180, expand=True)
    aug["r270"] = rgb.rotate(270, expand=True)
    aug["flip_lr"] = ImageOps.mirror(rgb)
    aug["flip_ud"] = ImageOps.flip(rgb)
    
    # Blur transforms
    aug["blur"] = rgb.filter(ImageFilter.GaussianBlur(radius=1.5))
    aug["blur_heavy"] = rgb.filter(ImageFilter.GaussianBlur(radius=3.0))
    
    # Sharpening
    aug["sharpen"] = rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Color/brightness transforms
    bright_enhancer = ImageEnhance.Brightness(rgb)
    aug["bright"] = bright_enhancer.enhance(1.3)
    aug["dark"] = bright_enhancer.enhance(0.7)
    
    contrast_enhancer = ImageEnhance.Contrast(rgb)
    aug["contrast"] = contrast_enhancer.enhance(1.3)
    
    color_enhancer = ImageEnhance.Color(rgb)
    aug["sat"] = color_enhancer.enhance(1.4)
    aug["desat"] = color_enhancer.enhance(0.6)
    
    # Combined transforms (rotation + blur, flip + brightness, etc.)
    aug["r90_blur"] = aug["r90"].filter(ImageFilter.GaussianBlur(radius=1.5))
    aug["flip_lr_bright"] = bright_enhancer.enhance(1.2)
    aug["r180_sharpen"] = aug["r180"].filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return aug


def read_yolo_labels(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO labels and return list of (cls, x, y, w, h)."""
    boxes: list[tuple[int, float, float, float, float]] = []
    try:
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    c = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append((c, x, y, w, h))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return boxes


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def transform_box(op: str, box: tuple[int, float, float, float, float]) -> tuple[int, float, float, float, float]:
    """Transform a normalized YOLO box for a given operation.

    Box format: (cls, x_center, y_center, width, height) all in [0,1].
    """
    c, x, y, w, h = box
    
    # Geometric transforms that change box coordinates
    if op == "r90":  # 90° CCW
        x2, y2, w2, h2 = y, 1.0 - x, h, w
    elif op == "r180":
        x2, y2, w2, h2 = 1.0 - x, 1.0 - y, w, h
    elif op == "r270":  # 270° CCW = 90° CW
        x2, y2, w2, h2 = 1.0 - y, x, h, w
    elif op == "flip_lr":  # mirror horizontally
        x2, y2, w2, h2 = 1.0 - x, y, w, h
    elif op == "flip_ud":  # flip vertically
        x2, y2, w2, h2 = x, 1.0 - y, w, h
    elif op == "r90_blur":  # 90° CCW + blur (same as r90)
        x2, y2, w2, h2 = y, 1.0 - x, h, w
    elif op == "flip_lr_bright":  # flip horizontally + brightness (same as flip_lr)
        x2, y2, w2, h2 = 1.0 - x, y, w, h
    elif op == "r180_sharpen":  # 180° + sharpen (same as r180)
        x2, y2, w2, h2 = 1.0 - x, 1.0 - y, w, h
    else:
        # Color/brightness/blur transforms don't change geometry
        x2, y2, w2, h2 = x, y, w, h
    
    x2 = _clamp01(x2)
    y2 = _clamp01(y2)
    w2 = _clamp01(w2)
    h2 = _clamp01(h2)
    eps = 1e-6
    if w2 <= 0:
        w2 = eps
    if h2 <= 0:
        h2 = eps
    return c, x2, y2, w2, h2


def save_augmented(
    source_image: Path,
    output_images: Path,
    output_labels: Path | None,
    label_sources: list[Path] | None,
) -> tuple[list[Path], list[Path]]:
    """Save augmented image variants and optional transformed label files.

    Returns (saved_image_paths, saved_label_paths).
    """
    saved_paths: list[Path] = []
    saved_label_paths: list[Path] = []
    try:
        with Image.open(source_image) as img:
            variants = augment_image(img)
    except Exception as e:
        print(f"Warning: failed to open {source_image}: {e}")
        return saved_paths, saved_label_paths

    stem = source_image.stem
    # find source label file if any
    src_label: Path | None = None
    if label_sources:
        for ls in label_sources:
            candidate = ls / f"{stem}.txt"
            if candidate.exists():
                src_label = candidate
                break
    src_boxes: list[tuple[int, float, float, float, float]] = []
    if src_label and src_label.exists():
        src_boxes = read_yolo_labels(src_label)

    for name, im in variants.items():
        out_name = f"{stem}__{name}.jpg"
        out_path = output_images / out_name
        if out_path.exists():
            continue
        try:
            im.save(out_path, format="JPEG", quality=95, subsampling=1, optimize=True)
            saved_paths.append(out_path)
        except Exception as e:
            print(f"Warning: failed to save {out_path}: {e}")
            continue

        # write transformed labels if available
        if output_labels is not None and src_boxes:
            out_label = output_labels / f"{stem}__{name}.txt"
            try:
                with out_label.open("w", encoding="utf-8") as f:
                    for box in src_boxes:
                        c, x2, y2, w2, h2 = transform_box(name, box)
                        f.write(f"{c} {x2:.6f} {y2:.6f} {w2:.6f} {h2:.6f}\n")
                saved_label_paths.append(out_label)
            except Exception as e:
                print(f"Warning: failed to write labels {out_label}: {e}")

    return saved_paths, saved_label_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate rotated/flipped variants of images and (optionally) matching YOLO labels."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["datasets/clip/images/train"],
        help="One or more source directories to scan for images.",
    )
    parser.add_argument(
        "--label-sources",
        nargs="*",
        default=["datasets/clip/labels/train", "datasets/clip/labels/val"],
        help=(
            "Optional label directories to read YOLO .txt files from (match by stem). "
            "If omitted or not found, labels are not written."
        ),
    )
    parser.add_argument(
        "--output-images",
        default="datasets/augmented/images",
        help="Directory to write augmented images into.",
    )
    parser.add_argument(
        "--output-labels",
        default="datasets/augmented/labels",
        help="Directory to write transformed YOLO labels into (if available).",
    )
    args = parser.parse_args()

    source_dirs = [Path(s).resolve() for s in args.sources]
    label_sources = [Path(s).resolve() for s in (args.label_sources or [])]
    output_images = Path(args.output_images).resolve()
    output_labels = Path(args.output_labels).resolve() if args.output_labels else None
    output_images.mkdir(parents=True, exist_ok=True)
    if output_labels is not None:
        output_labels.mkdir(parents=True, exist_ok=True)

    print("Scanning sources:")
    for d in source_dirs:
        print(f" - {d}")

    images = discover_images(source_dirs)
    if not images:
        print("No images found. Nothing to do.")
        return 0

    print(f"Found {len(images)} unique images. Generating augmentations...")
    total_saved = 0
    total_label_saved = 0
    for idx, src in enumerate(images, start=1):
        imgs, lbls = save_augmented(
            src,
            output_images,
            output_labels,
            label_sources,
        )
        total_saved += len(imgs)
        total_label_saved += len(lbls)
        if idx % 10 == 0 or imgs or lbls:
            print(
                f"[{idx}/{len(images)}] {src.name}: +{len(imgs)} images, +{len(lbls)} labels"
            )

    print(f"Done. Wrote {total_saved} augmented images to {output_images}")
    if output_labels is not None:
        print(f"Also wrote {total_label_saved} label files to {output_labels}")
    else:
        print("No label directory specified; only images were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


