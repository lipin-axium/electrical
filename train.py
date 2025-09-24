import os
import shutil
from pathlib import Path

from ultralytics import YOLO
import math
from dataset_split import build_final_dataset


def detect_apple_mps() -> str:
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def prepare_dataset(root: Path, use_all_for_train: bool = True) -> Path:
    """Create a YOLOv8-friendly dataset layout under datasets/clip/ by symlinking existing files.

    Expected existing layout:
      - data/*.jpg                # images
      - labels/obj_train_data/*.txt  # YOLO labels
      - labels/obj.names             # class names (Symbol A, Symbol B)

    We create:
      datasets/clip/
        images/train/..., images/val/...
        labels/train/..., labels/val/...
    Using an 80/20 split (deterministic by sorted filenames).
    """

    dataset_root = root / "datasets" / "clip"
    # If the prepared dataset already exists, fix any stale paths in data.yaml and reuse it
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        # Extract existing names to preserve class order
        names: list[str] = []
        try:
            lines = data_yaml.read_text(encoding="utf-8").splitlines()
            in_names = False
            for line in lines:
                s = line.strip()
                if s.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if s.startswith("-"):
                        names.append(s.split("-", 1)[1].strip())
                    elif s:
                        # Any non-list line ends names block
                        break
        except Exception:
            pass
        if not names:
            # Fallback to a generic two-class list if not found; overwritten below if obj.names exists
            names = ["Symbol A", "Symbol B"]

        # If raw labels with obj.names exist, prefer them for canonical class names
        class_names_file = root / "labels" / "obj.names"
        if class_names_file.exists():
            try:
                with open(class_names_file, "r", encoding="utf-8") as f:
                    file_names = [line.strip() for line in f if line.strip()]
                if file_names:
                    names = file_names
            except Exception:
                pass

        yaml_lines = [
            f"path: {dataset_root}",
            "train: images/train",
            "val: images/train" if use_all_for_train else "val: images/val",
            "names:",
            *[f"  - {n}" for n in names],
            "",
        ]
        data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
        return dataset_root

    images_dir = root / "data"
    label_txt_dir = root / "labels" / "obj_train_data"
    class_names_file = root / "labels" / "obj.names"

    assert images_dir.exists(), f"Missing images directory: {images_dir}"
    assert label_txt_dir.exists(), f"Missing labels directory: {label_txt_dir}"
    assert class_names_file.exists(), f"Missing class names file: {class_names_file}"

    # dataset_root defined above
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.glob("*.jpg")])
    # Match label files by same stem under labels/obj_train_data
    pairs = []
    for img in image_files:
        lbl = label_txt_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"Warning: missing label for image: {img.name}")

    if not pairs:
        raise RuntimeError("No image-label pairs found. Check file naming and locations.")

    if use_all_for_train:
        train_pairs = pairs
        val_pairs = pairs  # validate on train set when dataset is tiny
    else:
        split_index = int(len(pairs) * 0.8)
        train_pairs = pairs[:split_index]
        val_pairs = pairs[split_index:]

    def link(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            return
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def sanitize_label_file(src: Path, dst: Path) -> None:
        """Copy 'src' to 'dst' ensuring YOLO labels are finite and within [0,1]."""
        if dst.exists():
            return
        lines_out = []
        try:
            with open(src, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: skipping malformed label line {src.name}:{ln}: '{line}'")
                        continue
                    try:
                        cls = int(float(parts[0]))
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except Exception:
                        print(f"Warning: non-numeric label values {src.name}:{ln}: '{line}'")
                        continue
                    # Replace non-finite with 0.5 defaults
                    if not math.isfinite(x):
                        x = 0.5
                    if not math.isfinite(y):
                        y = 0.5
                    if not math.isfinite(w) or w <= 0:
                        w = 1e-6
                    if not math.isfinite(h) or h <= 0:
                        h = 1e-6
                    # Clamp to [0, 1]
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    w = max(1e-6, min(1.0, w))
                    h = max(1e-6, min(1.0, h))
                    lines_out.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"Warning: failed to read labels from {src}: {e}")
        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(lines_out)

    for img, lbl in train_pairs:
        link(img, images_train / img.name)
        sanitize_label_file(lbl, labels_train / lbl.name)

    for img, lbl in val_pairs:
        link(img, images_val / img.name)
        sanitize_label_file(lbl, labels_val / lbl.name)

    # also write data.yaml at dataset root for standalone use
    data_yaml = dataset_root / "data.yaml"
    with open(class_names_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    yaml_lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/train" if use_all_for_train else "val: images/val",
        "names:",
        *[f"  - {n}" for n in names],
        "",
    ]
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")

    return dataset_root


def main() -> None:
    project_root = Path(__file__).resolve().parent
    # Build/refresh final dataset split with train/val/test, including augmented data in train
    dataset_root = build_final_dataset(project_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, include_augmented=True)

    data_cfg = dataset_root / "data.yaml"

    device = detect_apple_mps()
    print(f"Using device: {device}")

    run_name = "symbol-detector-poc"
    model = YOLO("yolov8m.pt")
    model.train(
        data=str(data_cfg),
        epochs=300,  # More epochs for proof of concept
        imgsz=1280,
        batch=8,  # Larger batch size for better convergence
        device=device,
        name=run_name,
        patience=50,  # Much higher patience to ensure convergence
        augment=False,  # Disable augmentation since we have plenty of augmented data
        # Minimal augmentation to avoid overfitting
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        scale=0.0,
        translate=0.0,
        flipud=0.0,
        fliplr=0.0,
        copy_paste=0.0,
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        auto_augment="none",
        # Optimized loss weights for perfect training accuracy
        cls=10.0,     # Much higher classification loss weight
        box=10.0,     # Higher box loss weight
        dfl=2.0,      # Higher DFL loss weight
        optimizer="AdamW",
        lr0=0.0001,   # Lower learning rate for stable convergence
        lrf=0.001,    # Very low final learning rate
        momentum=0.937,
        weight_decay=0.0001,
        warmup_epochs=10, # More warmup
        # Training settings optimized for accuracy
        rect=True,
        workers=4,    # More workers
        plots=True,
        amp=True,     # Enable mixed precision
        project=str(project_root / "runs"),
    )

    # Evaluate on the held-out test split using the best weights saved by training
    # Find the newest run directory matching run_name*
    runs_dir = project_root / "runs"
    candidates = sorted([p for p in runs_dir.glob(f"{run_name}*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    best_weights = None
    if candidates:
        cand = candidates[0]
        maybe = cand / "weights" / "best.pt"
        if maybe.exists():
            best_weights = maybe

    eval_model = YOLO(str(best_weights)) if best_weights else model
    eval_model.val(data=str(data_cfg), split="test", device=device, workers=0, plots=True, save_json=False)


if __name__ == "__main__":
    main()


