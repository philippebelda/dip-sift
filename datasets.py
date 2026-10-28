"""Dataset construction: real pictures, synthetic transformations, HPatches loader."""
import json
from pathlib import Path

import cv2
import numpy as np

import config
import tools
from transformations import Transformed


def collect_images(assets_dir=config.ASSETS_DIR, max_side=config.MAX_SIDE, limit=None):
    """Return [(name, BGR image)]. Falls back on synthetic images when the folder is empty."""
    paths = tools.list_images(assets_dir)[:limit]
    images = [(p.stem, tools.load_image(p, max_side)) for p in paths]
    if not images:
        print(f"[datasets] no image found in {assets_dir}/, using synthetic images")
        images = [(f"synthetic_{i}", tools.synthetic_image(seed=i)) for i in range(3)]
    return images


def build_dataset(images, experiments, out_dir, rng=None):
    """Write every transformed image + its homography on disk, so that the exact same
    dataset can be shared, inspected or fed to other algorithms."""
    rng = rng or np.random.default_rng(config.SEED)
    out_dir = tools.ensure_dir(out_dir)
    manifest = []
    for name, img in images:
        for exp in experiments.values():
            for v in exp.values:
                t = exp.apply(img, v, rng)
                rel = Path(name) / exp.name / f"{v}.png"
                tools.ensure_dir(out_dir / rel.parent)
                cv2.imwrite(str(out_dir / rel), t.image)
                manifest.append(dict(image=name, experiment=exp.name, value=v,
                                     file=str(rel), H=t.H.tolist()))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[datasets] wrote {len(manifest)} images to {out_dir}")


def load_hpatches(root, max_side=None):
    """HPatches sequences: folders i_* (illumination) / v_* (viewpoint) holding
    1.ppm ... 6.ppm and homographies H_1_2 ... H_1_6.
    Returns [(sequence_name, reference_image, [Transformed, ...])]."""
    root = Path(root)
    out = []
    for seq in sorted(p for p in root.iterdir() if p.is_dir()):
        ref_path = seq / "1.ppm"
        if not ref_path.exists():
            continue
        ref = tools.load_image(ref_path, None)
        tests = []
        for i in range(2, 7):
            img_path, h_path = seq / f"{i}.ppm", seq / f"H_1_{i}"
            if img_path.exists() and h_path.exists():
                img = tools.load_image(img_path)
                mask = np.full(img.shape[:2], 255, np.uint8)
                tests.append(Transformed(img, np.loadtxt(h_path), mask, seq.name[:1], float(i)))
        out.append((seq.name, ref, tests))
    return out
