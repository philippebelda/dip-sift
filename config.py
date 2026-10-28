"""Global settings shared by every module."""
from pathlib import Path

ASSETS_DIR = Path("assets")
RESULTS_DIR = Path("results")

SEED = 0
MAX_SIDE = 800            # base images are downscaled to this size for speed

# --- evaluation protocol (see README, "Metrics") ---
EPS_PIXELS = 3.0          # max localisation error (px) for a keypoint to be "repeated"
SCALE_TOLERANCE = 1.5     # accepted ratio between predicted and observed keypoint size
LOWE_RATIO = 0.8          # ratio test threshold (Lowe 2004)
RANSAC_THRESHOLD = 3.0    # px, homography estimation
H_SUCCESS_PX = 3.0        # homography is "correct" if mean corner error < this (px, in reference frame)
MASK_MARGIN = 5           # px eroded from the valid area of warped images

DEFAULT_DETECTORS = ["sift", "orb", "akaze", "brisk"]
