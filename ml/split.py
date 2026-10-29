"""Dataset indexing + leak-free train/val/test split (pure Python, no torch needed).

Expected layout (as in your archive):  <root>/fake/<videoid>_<n>.png  and  <root>/real/<videoid>_<n>.png
Label convention matches the backend: fake = 1 (forged), real = 0.
The split is done per VIDEO, so frames of one video never appear in two splits.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_TO_LABEL = {"real": 0, "fake": 1}


def scan_dataset(root: str | Path) -> list[tuple[str, int, str]]:
    """Return [(path, label, video_id)]. video_id is the filename minus its trailing _<frame>."""
    root = Path(root)
    items = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = CLASS_TO_LABEL.get(cls_dir.name.lower())
        if label is None:
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                video_id = f.stem.rsplit("_", 1)[0]
                items.append((str(f), label, f"{cls_dir.name.lower()}/{video_id}"))
    if not items:
        raise FileNotFoundError(f"No images found under {root}/real and {root}/fake")
    return items


def split_by_video(items, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Stratified (per class) split at video level. Returns (train, val, test) item lists."""
    by_class = defaultdict(set)
    for _, label, vid in items:
        by_class[label].add(vid)

    rng = random.Random(seed)
    video_split: dict[str, str] = {}
    for label, vids in by_class.items():
        vids = sorted(vids)
        rng.shuffle(vids)
        n = len(vids)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for i, v in enumerate(vids):
            video_split[v] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[video_split[it[2]]].append(it)

    # safety check: no video in two splits
    seen = {}
    for name, rows in out.items():
        for _, _, vid in rows:
            assert seen.setdefault(vid, name) == name, f"leak: {vid} in {seen[vid]} and {name}"
    return out["train"], out["val"], out["test"]


def summarize(name: str, rows) -> str:
    n_fake = sum(1 for _, label, _ in rows if label == 1)
    vids = {v for _, _, v in rows}
    return f"{name}: {len(rows)} frames, {len(vids)} videos, fake frames {n_fake} ({n_fake / max(len(rows), 1):.0%})"


if __name__ == "__main__":
    import sys

    tr, va, te = split_by_video(scan_dataset(sys.argv[1]))
    for n, r in (("train", tr), ("val", va), ("test", te)):
        print(summarize(n, r))
