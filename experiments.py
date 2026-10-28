"""Experiment definitions and runners."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import config
import transformations as T
from detectors import detect_and_describe, get_detector
from metrics import evaluate_pair
from tools import to_gray

METRICS = ["n_kp_test", "repeatability", "localisation_error", "scale_error", "orientation_error",
           "precision", "recall", "matching_score", "h_error", "h_success", "inlier_ratio"]


@dataclass
class Experiment:
    name: str
    label: str
    values: list
    apply: Callable          # (img, value, rng) -> Transformed
    reverse_x: bool = False  # plot with decreasing x (e.g. scale)


EXPERIMENTS = {
    "scale": Experiment("scale", "scaling factor", [2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1],
                        lambda im, v, rng: T.scale(im, v)),
    "rotation": Experiment("rotation", "rotation (deg)", list(range(0, 181, 15)),
                           lambda im, v, rng: T.rotate(im, v)),
    "perspective": Experiment("perspective", "viewpoint strength", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                              lambda im, v, rng: T.perspective(im, v)),
    "brightness": Experiment("brightness", "brightness offset", [-80, -40, -20, 0, 20, 40, 80],
                             lambda im, v, rng: T.brightness(im, v)),
    "contrast": Experiment("contrast", "contrast gain", [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
                           lambda im, v, rng: T.contrast(im, v)),
    "noise": Experiment("noise", "gaussian noise sigma", [0, 2, 5, 10, 20, 30, 50],
                        lambda im, v, rng: T.gaussian_noise(im, v, rng)),
    "blur": Experiment("blur", "blur sigma (px)", [0, 0.5, 1, 1.5, 2, 3, 4],
                       lambda im, v, rng: T.blur(im, v)),
    "jpeg": Experiment("jpeg", "JPEG quality", [100, 80, 60, 40, 20, 10, 5],
                       lambda im, v, rng: T.jpeg(im, v), reverse_x=True),
}


def run_experiment(exp, images, detector_names, verbose=True):
    """For each image, detector and parameter value: detect on the original and on the
    transformed image, then compare using the ground-truth homography."""
    rows = []
    for i_img, (img_name, img) in enumerate(images):
        gray_ref = to_gray(img)
        for det_name in detector_names:
            det = get_detector(det_name)
            ref = detect_and_describe(det, gray_ref)
            for i_val, v in enumerate(exp.values):
                rng = np.random.default_rng([config.SEED, i_img, i_val])
                t = exp.apply(img, v, rng)
                test = detect_and_describe(det, to_gray(t.image))
                m = evaluate_pair(det, ref, test, t.H, t.mask)
                m.update(experiment=exp.name, image=img_name, detector=det_name, value=v,
                         t_test_ms=1e3 * (test.t_detect + test.t_describe))
                rows.append(m)
        if verbose:
            print(f"  [{exp.name}] {img_name} done")
    return rows


# --------------------------------------------------------------------------- SIFT parameters
SIFT_SWEEPS = {
    "contrastThreshold": [0.01, 0.02, 0.04, 0.08, 0.12],
    "edgeThreshold": [5, 10, 20, 40],
    "nOctaveLayers": [2, 3, 4, 5],
    "sigma": [1.0, 1.4, 1.6, 2.0, 2.4],
}
# a small battery of "typical" nuisances to average over
SIFT_BATTERY = [("scale", 0.5), ("rotation", 30), ("perspective", 0.3), ("noise", 10), ("blur", 2)]


def run_sift_param_sweep(images, sweeps=SIFT_SWEEPS, battery=SIFT_BATTERY, verbose=True):
    rows = []
    for param, values in sweeps.items():
        for value in values:
            det = get_detector("sift", **{param: value})
            for i_img, (img_name, img) in enumerate(images):
                ref = detect_and_describe(det, to_gray(img))
                for exp_name, v in battery:
                    rng = np.random.default_rng([config.SEED, i_img])
                    t = EXPERIMENTS[exp_name].apply(img, v, rng)
                    test = detect_and_describe(det, to_gray(t.image))
                    m = evaluate_pair(det, ref, test, t.H, t.mask)
                    m.update(param=param, param_value=value, transform=f"{exp_name}={v}", image=img_name,
                             t_test_ms=1e3 * (test.t_detect + test.t_describe))
                    rows.append(m)
        if verbose:
            print(f"  [sift sweep] {param} done")
    return rows


# --------------------------------------------------------------------------- resolution study
def resolution_study(img, ratios, detector_name="sift"):
    """Generalisation of the original script: keypoints found at several resolutions.
    Sizes are expressed in *original* pixels (size / ratio): a scale-invariant detector
    should give comparable features, only fewer small ones when resolution drops."""
    import cv2
    det = get_detector(detector_name)
    rows, vis = [], []
    for r in ratios:
        small = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        d = detect_and_describe(det, to_gray(small))
        sizes = d.sizes / r if len(d) else np.array([np.nan])
        rows.append(dict(ratio=r, n_kp=len(d), width=small.shape[1], height=small.shape[0],
                         kp_per_megapixel=len(d) / (small.shape[0] * small.shape[1] / 1e6),
                         min_size_orig=float(np.nanmin(sizes)), median_size_orig=float(np.nanmedian(sizes)),
                         max_size_orig=float(np.nanmax(sizes)), mean_response=float(d.responses.mean()) if len(d) else np.nan))
        vis.append((small, d))
    return rows, vis


# --------------------------------------------------------------------------- HPatches
def run_hpatches(root, detector_names, max_sequences=None, verbose=True):
    from datasets import load_hpatches
    rows = []
    for k, (seq, ref_img, tests) in enumerate(load_hpatches(root)[:max_sequences]):
        gray = to_gray(ref_img)
        for det_name in detector_names:
            det = get_detector(det_name)
            ref = detect_and_describe(det, gray)
            for t in tests:
                test = detect_and_describe(det, to_gray(t.image))
                m = evaluate_pair(det, ref, test, t.H, t.mask)
                m.update(experiment="hpatches_" + t.name, image=seq, detector=det_name, value=t.value)
                rows.append(m)
        if verbose:
            print(f"  [hpatches] {seq}")
    return rows
