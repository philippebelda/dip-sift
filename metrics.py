"""Quality metrics computed from a ground-truth homography H (reference -> test).

Protocol follows Mikolajczyk & Schmid, "A performance evaluation of local descriptors"
(2005) and the HPatches benchmark (Balntas et al., 2017). See README for the maths.
"""
import math

import cv2
import numpy as np

import config


def project_points(pts, H):
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def jacobian(pts, H):
    """Local 2x2 jacobian [[du/dx, du/dy],[dv/dx, dv/dy]] of the homography at each point.
    For a homography (u,v) = (h1.p / w, h2.p / w), w = h3.p, so du/dx = (h11 - u*h31)/w ..."""
    x, y = pts[:, 0], pts[:, 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    a = (H[0, 0] - u * H[2, 0]) / w
    b = (H[0, 1] - u * H[2, 1]) / w
    c = (H[1, 0] - v * H[2, 0]) / w
    d = (H[1, 1] - v * H[2, 1]) / w
    return a, b, c, d


def _in_bounds(pts, shape, mask):
    h, w = shape[:2]
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if mask is not None and ok.any():
        xi = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
        ok &= mask[yi, xi] > 0
    return ok


def _wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def _gt_pairs(projA, sizeA, scaleA, ptsB, sizeB, eps, tol, chunk=512):
    """Ground-truth correspondences: nearest test keypoint within `eps` px whose size is
    compatible with the predicted one. Chunked to keep memory bounded."""
    ia, ib = [], []
    b2 = (ptsB ** 2).sum(1)[None, :]
    for s in range(0, len(projA), chunk):
        a = projA[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2 - 2 * a @ ptsB.T
        r = sizeB[None, :] / (sizeA[s:s + chunk, None] * scaleA[s:s + chunk, None])
        d2[(r > tol) | (r < 1 / tol) | (d2 > eps ** 2)] = np.inf
        j = d2.argmin(1)
        best = d2[np.arange(len(a)), j]
        ok = np.isfinite(best)
        ia.append(np.flatnonzero(ok) + s)
        ib.append(j[ok])
    if not ia:
        return np.array([], int), np.array([], int)
    # No one-to-one dedupe on purpose: SIFT may emit several keypoints at the same
    # location with different orientations (up to 2-3 per blob), all legitimate.
    return np.concatenate(ia), np.concatenate(ib)


def corner_error(H_est, H_gt, shape):
    """Mean distance between the 4 image corners mapped by H_est and H_gt,
    expressed in the *reference* frame (so it is comparable across image scales)."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    est = project_points(corners, H_est)
    gt = project_points(corners, H_gt)
    back_est = project_points(est, np.linalg.inv(H_gt))
    back_gt = project_points(gt, np.linalg.inv(H_gt))
    return float(np.linalg.norm(back_est - back_gt, axis=1).mean())


def evaluate_pair(det, A, B, H, mask, return_details=False):
    """A: Detection on the reference image; B: Detection on the transformed image."""
    eps, tol = config.EPS_PIXELS, config.SCALE_TOLERANCE
    nan = math.nan
    out = dict(n_kp_ref=len(A), n_kp_test=len(B), repeatability=nan, localisation_error=nan,
               scale_error=nan, orientation_error=nan, n_matches=0, precision=nan,
               recall=nan, matching_score=nan, h_error=nan, h_success=0.0, inlier_ratio=nan)
    if len(A) == 0 or len(B) == 0:
        return out

    projA = project_points(A.pts, H)
    validA = _in_bounds(projA, B.shape, mask)
    validB = _in_bounds(project_points(B.pts, np.linalg.inv(H)), A.shape, None) & _in_bounds(B.pts, B.shape, mask)
    idxA, idxB = np.flatnonzero(validA), np.flatnonzero(validB)
    out["n_valid_ref"], out["n_valid_test"] = len(idxA), len(idxB)
    if len(idxA) == 0 or len(idxB) == 0:
        return out
    denom = min(len(idxA), len(idxB))

    # ---- 1. detector level: repeatability and geometric accuracy
    a, b, c, d = jacobian(A.pts[idxA], H)
    scaleA = np.sqrt(np.abs(a * d - b * c))
    pa, pb = _gt_pairs(projA[idxA], A.sizes[idxA], scaleA, B.pts[idxB], B.sizes[idxB], eps, tol)
    # matched keypoints counted in both directions (ref->test, test->ref), so duplicates at one location are not penalised
    Hinv = np.linalg.inv(H)
    a2, b2_, c2, d2_ = jacobian(B.pts[idxB], Hinv)
    qb, _ = _gt_pairs(project_points(B.pts[idxB], Hinv), B.sizes[idxB], np.sqrt(np.abs(a2 * d2_ - b2_ * c2)),
                      A.pts[idxA], A.sizes[idxA], eps, tol)
    n_rep = min(len(np.unique(pa)), len(np.unique(qb)))
    out["repeatability"] = n_rep / denom
    if len(pa):
        iA, iB = idxA[pa], idxB[pb]
        out["localisation_error"] = float(np.linalg.norm(projA[iA] - B.pts[iB], axis=1).mean())
        out["scale_error"] = float(np.median(np.abs(np.log2(B.sizes[iB] / (A.sizes[iA] * scaleA[pa])))))
        th = np.radians(A.angles[iA])
        pred = np.degrees(np.arctan2(c[pa] * np.cos(th) + d[pa] * np.sin(th),
                                     a[pa] * np.cos(th) + b[pa] * np.sin(th)))
        out["orientation_error"] = float(np.median(np.abs(_wrap_deg(B.angles[iB] - pred))))

    # ---- 2. descriptor level: matching with Lowe's ratio test
    if A.descriptors is None or B.descriptors is None or len(idxB) < 2:
        return out
    bf = cv2.BFMatcher(det.norm)
    knn = bf.knnMatch(A.descriptors[idxA], B.descriptors[idxB], k=2)
    mi, mj = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < config.LOWE_RATIO * pair[1].distance:
            mi.append(idxA[pair[0].queryIdx])
            mj.append(idxB[pair[0].trainIdx])
    mi, mj = np.array(mi, int), np.array(mj, int)
    out["n_matches"] = len(mi)
    if len(mi) == 0:
        out.update(precision=0.0, recall=0.0, matching_score=0.0)
        return out
    correct = np.linalg.norm(projA[mi] - B.pts[mj], axis=1) < eps
    out["precision"] = float(correct.mean())
    out["matching_score"] = float(correct.sum() / denom)
    out["recall"] = float(min(1.0, correct.sum() / max(n_rep, 1)))

    # ---- 3. task level: can we recover the transformation from the matches?
    if len(mi) >= 4:
        H_est, inl = cv2.findHomography(A.pts[mi], B.pts[mj], cv2.RANSAC, config.RANSAC_THRESHOLD)
        if H_est is not None:
            out["inlier_ratio"] = float(inl.mean())
            err = corner_error(H_est, H, A.shape)
            out["h_error"] = err
            out["h_success"] = float(err < config.H_SUCCESS_PX)
    if return_details:
        out["_details"] = dict(mi=mi, mj=mj, correct=correct)
    return out
