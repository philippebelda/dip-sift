"""Small generic helpers (IO, seeds, aggregation)."""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}


def set_seed(seed):
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def load_image(path, max_side=None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def synthetic_image(size=(600, 800), seed=0):
    """Textured random image, used when no real picture is available."""
    rng = np.random.default_rng(seed)
    h, w = size
    low = rng.normal(128, 45, (h // 10, w // 10, 3)).clip(0, 255).astype(np.float32)
    img = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)
    for _ in range(80):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.integers(5, 40)), color, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))), color, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, w)), int(rng.integers(0, h))), color, int(rng.integers(1, 5)))
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def save_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not k.startswith("_"):
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate(rows, group_keys, metric_keys):
    """Group rows and return mean / std / n of each metric (NaN-aware).

    Spread is computed across *images*, which is what we want to know:
    how much does the result depend on the picture?
    """
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    out = []
    for key, members in groups.items():
        row = dict(zip(group_keys, key))
        row["n"] = len(members)
        for m in metric_keys:
            vals = np.array([x.get(m, np.nan) for x in members], dtype=float)
            ok = vals[~np.isnan(vals)]
            row[f"{m}_mean"] = float(ok.mean()) if len(ok) else math.nan
            row[f"{m}_std"] = float(ok.std()) if len(ok) else math.nan
        out.append(row)
    return out
