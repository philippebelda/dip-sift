"""Run the SIFT evaluation.

python main.py                       # everything, with images in assets/
python main.py --quick               # small & fast (2 images, fewer detectors)
python main.py --only scale rotation # chosen experiments
python main.py --hpatches path/to/hpatches-sequences-release
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config
import experiments as E
import profiling
import tools
import visualization as V
from datasets import build_dataset, collect_images
from detectors import detect_and_describe, get_detector
from tools import aggregate, ensure_dir, save_csv, to_gray

PLOT_METRICS = ["repeatability", "matching_score", "precision", "n_kp_test",
                "localisation_error", "scale_error", "orientation_error", "h_success", "h_error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(config.ASSETS_DIR))
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    ap.add_argument("--detectors", nargs="+", default=config.DEFAULT_DETECTORS)
    ap.add_argument("--only", nargs="+", default=None, help="experiment names, or: resolution sweep benchmark")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=config.MAX_SIDE)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--build-dataset", action="store_true", help="also write transformed images to disk")
    ap.add_argument("--hpatches", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    if args.quick:
        args.max_images, args.max_side = args.max_images or 2, 500
        args.detectors = args.detectors[:2] if args.detectors == config.DEFAULT_DETECTORS else args.detectors

    out = ensure_dir(args.out)
    tools.set_seed(config.SEED)
    images = collect_images(args.assets, args.max_side, args.max_images)
    wanted = lambda name: args.only is None or name in args.only
    print(f"{len(images)} image(s), detectors: {args.detectors}")

    (out / "system.json").write_text(json.dumps(profiling.gpu_info(), indent=1))
    if args.build_dataset:
        build_dataset(images, E.EXPERIMENTS, out / "dataset")

    # 1. resolution study (generalises the original script) ------------------------------
    if wanted("resolution"):
        ratios = np.linspace(1, 0.01, 6)
        rows, vis = E.resolution_study(images[0][1], ratios)
        save_csv(rows, out / "resolution_study.csv")
        V.plot_keypoint_grid(vis, ratios, out / "resolution_keypoints.png", show=args.show)
        V.plot_resolution_study(rows, out / "resolution_study.png", show=args.show)

    # 2. keypoint properties + a match visualisation ------------------------------------
    name0, img0 = images[0]
    dets = {n: detect_and_describe(get_detector(n), to_gray(img0)) for n in args.detectors}
    V.plot_property_histograms(dets, out / "keypoint_properties.png", show=args.show)
    from metrics import evaluate_pair
    import transformations as T
    t = T.rotate(img0, 30)
    sift = get_detector("sift")
    ra, rb = dets.get("sift") or detect_and_describe(sift, to_gray(img0)), detect_and_describe(sift, to_gray(t.image))
    res = evaluate_pair(sift, ra, rb, t.H, t.mask, return_details=True)
    if "_details" in res:
        V.plot_matches(img0, ra, t.image, rb, res["_details"], out / "matches_rotation30.png", show=args.show)

    # 3. transformation experiments ------------------------------------------------------
    for name, exp in E.EXPERIMENTS.items():
        if not wanted(name):
            continue
        print(f"Experiment {name}")
        rows = E.run_experiment(exp, images, args.detectors)
        save_csv(rows, out / f"{name}.csv")
        agg = aggregate(rows, ["detector", "value"], E.METRICS + ["t_test_ms"])
        save_csv(agg, out / f"{name}_summary.csv")
        V.plot_experiment(agg, exp, PLOT_METRICS, out / f"{name}.png", show=args.show)

    # 4. SIFT hyper-parameters -----------------------------------------------------------
    if wanted("sweep"):
        print("SIFT parameter sweep")
        sweeps = E.SIFT_SWEEPS if not args.quick else {"contrastThreshold": [0.02, 0.04, 0.08]}
        rows = E.run_sift_param_sweep(images, sweeps)
        save_csv(rows, out / "sift_param_sweep.csv")
        agg = aggregate(rows, ["param", "param_value"], ["n_kp_test", "matching_score", "repeatability", "t_test_ms"])
        save_csv(agg, out / "sift_param_sweep_summary.csv")
        V.plot_param_sweep(agg, out / "sift_param_sweep.png", show=args.show)

    # 5. performance ---------------------------------------------------------------------
    if wanted("benchmark"):
        print("Benchmark (time / CPU / memory / GPU)")
        dets_obj = [get_detector(n) for n in args.detectors]
        b = profiling.benchmark_detectors(dets_obj, img0, repeat=3 if args.quick else 5)
        th = [r for d in dets_obj for r in profiling.benchmark_threads(d, img0, repeat=3)]
        save_csv(b, out / "benchmark.csv")
        save_csv(th, out / "benchmark_threads.csv")
        V.plot_benchmark(b, th, out / "benchmark.png", show=args.show)

    # 6. real benchmark (optional) --------------------------------------------------------
    if args.hpatches:
        rows = E.run_hpatches(args.hpatches, args.detectors, max_sequences=10 if args.quick else None)
        save_csv(rows, out / "hpatches.csv")
        save_csv(aggregate(rows, ["experiment", "detector"], E.METRICS), out / "hpatches_summary.csv")

    print(f"Done. Results in {Path(out).resolve()}")


if __name__ == "__main__":
    main()
