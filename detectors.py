"""Uniform wrapper around the feature detectors being compared."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cv2
import numpy as np


@dataclass
class Detector:
    name: str
    algo: Any
    norm: int
    params: dict = field(default_factory=dict)


def get_detector(name, **params):
    key = name.lower()
    if key == "sift":
        return Detector("sift", cv2.SIFT_create(**params), cv2.NORM_L2, params)
    if key == "orb":
        params.setdefault("nfeatures", 2000)   # default (500) would cap repeatability
        return Detector("orb", cv2.ORB_create(**params), cv2.NORM_HAMMING, params)
    if key == "akaze":
        return Detector("akaze", cv2.AKAZE_create(**params), cv2.NORM_HAMMING, params)
    if key == "brisk":
        return Detector("brisk", cv2.BRISK_create(**params), cv2.NORM_HAMMING, params)
    raise ValueError(f"unknown detector {name!r}")


def decode_octave(kp):
    """OpenCV packs (octave, layer, scale) in kp.octave."""
    octave = kp.octave & 255
    layer = (kp.octave >> 8) & 255
    if octave >= 128:
        octave -= 256
    return octave, layer


@dataclass
class Detection:
    keypoints: tuple
    descriptors: Any
    shape: tuple
    t_detect: float
    t_describe: float

    def __post_init__(self):
        kps = self.keypoints
        self.pts = np.array([k.pt for k in kps], float).reshape(-1, 2)
        self.sizes = np.array([k.size for k in kps], float)
        self.angles = np.array([k.angle for k in kps], float)
        self.responses = np.array([k.response for k in kps], float)
        self.octaves = np.array([decode_octave(k)[0] for k in kps], int)

    def __len__(self):
        return len(self.keypoints)


def detect_and_describe(det, gray):
    t0 = perf_counter()
    kps = det.algo.detect(gray, None)
    t1 = perf_counter()
    desc = None
    if len(kps):
        kps, desc = det.algo.compute(gray, kps)   # may drop border keypoints
    t2 = perf_counter()
    return Detection(tuple(kps), desc, gray.shape, t1 - t0, t2 - t1)
