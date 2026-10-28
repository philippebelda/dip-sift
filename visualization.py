"""All figures. Saved to disk; shown only if show=True."""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tools import ensure_dir

LABELS = {
    "repeatability": "repeatability", "matching_score": "matching score", "precision": "precision",
    "recall": "recall", "n_kp_test": "# keypoints", "localisation_error": "localisation error (px)",
    "scale_error": "scale error (log2)", "orientation_error": "orientation error (deg)",
    "h_success": "homography success rate", "h_error": "homography error (px)",
    "inlier_ratio": "RANSAC inlier ratio",
}


def _finish(fig, path, show):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def plot_experiment(agg, exp, metrics, path, show=False):
    """agg: output of tools.aggregate(rows, ['detector','value'], metrics)."""
    cols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.6 * nrows), squeeze=False)
    detectors = sorted({r["detector"] for r in agg})
    for ax, m in zip(axes.ravel(), metrics):
        for det in detectors:
            pts = sorted((r for r in agg if r["detector"] == det), key=lambda r: r["value"])
            x = [r["value"] for r in pts]
            y = np.array([r[f"{m}_mean"] for r in pts])
            s = np.array([r[f"{m}_std"] for r in pts])
            ax.plot(x, y, marker="o", ms=3, label=det)
            ax.fill_between(x, y - s, y + s, alpha=0.12)
        ax.set_title(LABELS.get(m, m))
        ax.set_xlabel(exp.label)
        ax.grid(alpha=0.3)
        if exp.reverse_x:
            ax.invert_xaxis()
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    axes[0, 0].legend()
    fig.suptitle(f"Experiment: {exp.name}  (mean ± std over images)")
    _finish(fig, path, show)


def plot_keypoint_grid(vis, ratios, path, per_line=3, show=False):
    """Original script's figure: keypoints drawn on black canvases at several resolutions."""
    lines = int(np.ceil(len(vis) / per_line))
    fig, axes = plt.subplots(lines, per_line, figsize=(4.5 * per_line, 3.6 * lines), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, d), r in zip(axes.ravel(), vis, ratios):
        canvas = cv2.drawKeypoints(np.zeros(img.shape[:2], np.uint8), d.keypoints, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(canvas)
        ax.set_title(f"scaling: {r:.3f} ({len(d)} kp)")
    fig.suptitle("SIFT at different resolutions")
    _finish(fig, path, show)


def plot_resolution_study(rows, path, show=False):
    r = [x["ratio"] for x in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(r, [x["n_kp"] for x in rows], "o-")
    axes[0].set_title("number of keypoints")
    axes[1].plot(r, [x["max_size_orig"] for x in rows], "o-", label="max")
    axes[1].plot(r, [x["median_size_orig"] for x in rows], "o-", label="median")
    axes[1].set_title("keypoint size in original pixels")
    axes[1].legend()
    axes[2].plot(r, [x["kp_per_megapixel"] for x in rows], "o-")
    axes[2].set_title("keypoints per megapixel")
    for ax in axes:
        ax.set_xlabel("scaling ratio")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    _finish(fig, path, show)


def plot_property_histograms(det_results, path, show=False):
    """det_results: {detector_name: Detection} on the same image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, d in det_results.items():
        if len(d) == 0:
            continue
        axes[0].hist(np.log2(d.sizes), bins=30, alpha=0.5, label=name)
        axes[1].hist(d.angles, bins=36, alpha=0.5, label=name)
        axes[2].hist(np.log10(np.abs(d.responses) + 1e-9), bins=30, alpha=0.5, label=name)
    axes[0].set_title("log2(size)"); axes[1].set_title("orientation (deg)"); axes[2].set_title("log10(response)")
    axes[0].legend()
    _finish(fig, path, show)


def plot_matches(img_a, det_a, img_b, det_b, details, path, show=False, max_draw=150):
    """Green = correct match according to ground truth, red = wrong match."""
    mi, mj, ok = details["mi"], details["mj"], details["correct"]
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    canvas = np.zeros((max(ha, hb), wa + wb, 3), np.uint8)
    canvas[:ha, :wa], canvas[:hb, wa:] = img_a, img_b
    for k in np.random.default_rng(0).permutation(len(mi))[:max_draw]:
        p, q = det_a.pts[mi[k]], det_b.pts[mj[k]] + [wa, 0]
        cv2.line(canvas, tuple(int(v) for v in p), tuple(int(v) for v in q),
                 (0, 200, 0) if ok[k] else (0, 0, 255), 1, cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas[..., ::-1])
    ax.axis("off")
    ax.set_title(f"{len(mi)} matches, {int(ok.sum())} correct (green) / {int((~ok).sum())} wrong (red)")
    _finish(fig, path, show)


def plot_param_sweep(agg, path, show=False):
    """agg: aggregate(rows, ['param','param_value'], [...]) for the SIFT parameter study."""
    params = sorted({r["param"] for r in agg})
    fig, axes = plt.subplots(3, len(params), figsize=(4.2 * len(params), 9), squeeze=False)
    for j, p in enumerate(params):
        pts = sorted((r for r in agg if r["param"] == p), key=lambda r: r["param_value"])
        x = [r["param_value"] for r in pts]
        for i, (m, title) in enumerate([("n_kp_test", "# keypoints"), ("matching_score", "matching score"),
                                        ("t_test_ms", "time (ms)")]):
            y = np.array([r[f"{m}_mean"] for r in pts]); s = np.array([r[f"{m}_std"] for r in pts])
            axes[i, j].errorbar(x, y, yerr=s, marker="o", capsize=3)
            axes[i, j].set_title(f"{title} vs {p}")
            axes[i, j].grid(alpha=0.3)
    _finish(fig, path, show)


def plot_benchmark(rows, thread_rows, path, show=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for det in sorted({r["detector"] for r in rows}):
        pts = [r for r in rows if r["detector"] == det]
        mp = [r["megapixels"] for r in pts]
        axes[0].plot(mp, [r["detect_ms"] + r["describe_ms"] for r in pts], "o-", label=det)
        axes[1].plot(mp, [r["peak_rss_mb"] for r in pts], "o-", label=det)
    axes[0].set_title("time (ms) vs megapixels"); axes[1].set_title("peak RSS (MB)")
    axes[0].legend()
    for r_det in sorted({r["detector"] for r in thread_rows}):
        pts = [r for r in thread_rows if r["detector"] == r_det]
        axes[2].plot([r["threads"] for r in pts], [r["speedup"] for r in pts], "o-", label=r_det)
    axes[2].set_title("speed-up vs CPU threads"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    _finish(fig, path, show)
