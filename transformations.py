"""Image transformations with *known ground truth*.

Every function returns a `Transformed` holding the new image, the 3x3 homography H
mapping original pixel coordinates to the new image, and a mask of the pixels that
contain real image content (borders created by warping are excluded).
Photometric / noise transformations have H = identity.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Transformed:
    image: np.ndarray
    H: np.ndarray
    mask: np.ndarray          # uint8, 255 where the pixel is valid
    name: str
    value: float


def _full_mask(img):
    return np.full(img.shape[:2], 255, np.uint8)


def _warp_mask(shape, H, out_size):
    """Warp an all-white image and erode it, to know where real content is."""
    ones = np.full(shape[:2], 255, np.uint8)
    m = cv2.warpPerspective(ones, H, out_size, flags=cv2.INTER_NEAREST)
    k = 2 * config.MASK_MARGIN + 1
    return cv2.erode(m, np.ones((k, k), np.uint8))


def identity(img, name="identity", value=0.0):
    return Transformed(img.copy(), np.eye(3), _full_mask(img), name, value)


def scale(img, s):
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    sx, sy = out.shape[1] / w, out.shape[0] / h   # real factors after rounding
    # OpenCV maps pixel centres: x' = s * (x + 0.5) - 0.5
    H = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]], float)
    return Transformed(out, H, _full_mask(out), "scale", s)


def rotate(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, 1.0)
    H = np.vstack([M, [0, 0, 1]])
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return Transformed(out, H, _warp_mask(img.shape, H, (w, h)), "rotation", angle_deg)


def perspective(img, strength):
    """Fake viewpoint change: the right edge is shortened by `strength` (0 .. ~0.6)
    and the image is squeezed horizontally, like looking at a plane from the side."""
    h, w = img.shape[:2]
    t = strength * h / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w * (1 - strength / 2), t], [w * (1 - strength / 2), h - t], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst).astype(float)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
    return Transformed(out, H, _warp_mask(img.shape, H, (w, h)), "perspective", strength)


def brightness(img, beta):
    out = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return Transformed(out, np.eye(3), _full_mask(out), "brightness", beta)


def contrast(img, alpha):
    out = np.clip((img.astype(np.float32) - 128) * alpha + 128, 0, 255).astype(np.uint8)
    return Transformed(out, np.eye(3), _full_mask(out), "contrast", alpha)


def gaussian_noise(img, sigma, rng):
    noise = rng.normal(0, sigma, img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Transformed(out, np.eye(3), _full_mask(out), "noise", sigma)


def blur(img, sigma):
    out = img.copy() if sigma <= 0 else cv2.GaussianBlur(img, (0, 0), sigma)
    return Transformed(out, np.eye(3), _full_mask(out), "blur", sigma)


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Transformed(out, np.eye(3), _full_mask(out), "jpeg", quality)
