import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageItem:
    image_path: Path
    label_path: Path | None
    stem: str


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def _compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def _discover_images(directories: Iterable[Path]) -> list[Path]:
    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    for directory in directories:
        for p in sorted(directory.rglob("*")):
            if not _is_image(p):
                continue
            try:
                h = _compute_sha256(p)
            except Exception:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            unique_paths.append(p)
    return unique_paths


def _match_label(image_path: Path, label_dirs: Iterable[Path]) -> Path | None:
    stem = image_path.stem
    for d in label_dirs:
        p = d / f"{stem}.txt"
        if p.exists():
            return p
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yaml(dataset_root: Path, names: list[str], use_train_for_val: bool = False) -> None:
    yaml_lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/train" if use_train_for_val else "val: images/val",
        "test: images/test",
        "names:",
        *[f"  - {n}" for n in names],
        "",
    ]
    (dataset_root / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")


def _read_names(project_root: Path) -> list[str]:
    # Prefer labels/obj.names if present
    names_path = project_root / "labels" / "obj.names"
    if names_path.exists():
        try:
            return [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            pass
    # Fallback to datasets/clip/data.yaml if available
    clip_yaml = project_root / "datasets" / "clip" / "data.yaml"
    if clip_yaml.exists():
        try:
            lines = clip_yaml.read_text(encoding="utf-8").splitlines()
            names: list[str] = []
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
                        break
            if names:
                return names
        except Exception:
            pass
    return ["Symbol A", "Symbol B"]


def build_final_dataset(
    project_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    include_augmented: bool = True,
) -> Path:
    """Create datasets/final with images/labels train/val/test and data.yaml.

    - Splits are made from ORIGINAL images only (no augmented) using deterministic shuffle.
    - Augmented images/labels are INCLUDED ONLY in the train split for originals assigned to train.
    - Existing files are left in place (symlink/copy skipped if already exists).
    """

    assert 0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01, "Ratios must sum to 1.0"

    clip_images = [project_root / "datasets" / "clip" / "images" / "train",
                   project_root / "datasets" / "clip" / "images" / "val"]
    clip_labels = [project_root / "datasets" / "clip" / "labels" / "train",
                   project_root / "datasets" / "clip" / "labels" / "val"]
    aug_images = project_root / "datasets" / "augmented" / "images"
    aug_labels = project_root / "datasets" / "augmented" / "labels"

    originals = _discover_images([p for p in clip_images if p.exists()])
    if not originals:
        raise RuntimeError("No original images found under datasets/clip/images.")

    # Build list of original items with labels
    original_items: list[ImageItem] = []
    for img in originals:
        lbl = _match_label(img, [p for p in clip_labels if p.exists()])
        original_items.append(ImageItem(image_path=img, label_path=lbl, stem=img.stem))

    # Deterministic split
    rng = random.Random(seed)
    original_items.sort(key=lambda it: it.stem)
    rng.shuffle(original_items)
    n = len(original_items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_items = original_items[:n_train]
    val_items = original_items[n_train:n_train + n_val]
    test_items = original_items[n_train + n_val:]

    dataset_root = project_root / "datasets" / "final"
    img_train = dataset_root / "images" / "train"
    img_val = dataset_root / "images" / "val"
    img_test = dataset_root / "images" / "test"
    lbl_train = dataset_root / "labels" / "train"
    lbl_val = dataset_root / "labels" / "val"
    lbl_test = dataset_root / "labels" / "test"
    for d in [img_train, img_val, img_test, lbl_train, lbl_val, lbl_test]:
        d.mkdir(parents=True, exist_ok=True)

    # Helper to copy one item
    def add_item(item: ImageItem, img_dst: Path, lbl_dst_dir: Path) -> None:
        _link_or_copy(item.image_path, img_dst / item.image_path.name)
        if item.label_path is not None and item.label_path.exists():
            _link_or_copy(item.label_path, lbl_dst_dir / item.label_path.name)

    for it in train_items:
        add_item(it, img_train, lbl_train)
    for it in val_items:
        add_item(it, img_val, lbl_val)
    for it in test_items:
        add_item(it, img_test, lbl_test)

    # Include augmented only for train items
    if include_augmented and aug_images.exists() and aug_labels.exists():
        train_stems = {it.stem for it in train_items}
        for p in sorted(aug_images.glob("*.jpg")):
            # augmented filenames look like "{stem}__op.jpg"
            stem_base = p.stem.split("__", 1)[0]
            if stem_base not in train_stems:
                continue
            lbl = aug_labels / f"{p.stem}.txt"
            _link_or_copy(p, img_train / p.name)
            if lbl.exists():
                _link_or_copy(lbl, lbl_train / lbl.name)

    # Write YAML
    names = _read_names(project_root)
    _write_yaml(dataset_root, names, use_train_for_val=False)

    return dataset_root


