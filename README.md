# Digital Image Processing and Computer Vision

## Objective
Evaluate the SIFT algorithm: establish methods to measure its efficiency and correctness.
- compare with other methods (ORB, AKAZE, BRISK) to get reference points
- build datasets with known ground truth
- justify choices (data, maths behind the formulas, ...)
- monitor performance: time, CPU, memory, GPU

## Project layout
| file                 | role                                                                                                                               |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `config.py`          | all thresholds of the protocol (ε, ratio test, ...)                                                                                |
| `tools.py`           | IO, seeds, synthetic image, CSV, aggregation (mean/std over images)                                                                |
| `transformations.py` | scale, rotation, perspective, brightness, contrast, noise, blur, JPEG — each returns the image **and its ground-truth homography** |
| `detectors.py`       | uniform wrapper (SIFT, ORB, AKAZE, BRISK) + `Detection` (arrays of size/angle/response/octave)                                     |
| `metrics.py`         | repeatability, localisation / scale / orientation error, precision, recall, matching score, homography error                       |
| `profiling.py`       | timing vs image size, CPU threads scaling, CPU time, peak RSS, GPU (nvidia-smi)                                                    |
| `datasets.py`        | image collection, on-disk dataset builder, HPatches loader                                                                         |
| `experiments.py`     | experiment registry + runners (transformations, SIFT hyper-parameters, resolution study)                                           |
| `visualization.py`   | all figures                                                                                                                        |
| `main.py`            | CLI entry point                                                                                                                    |

```bash
pip install -r requirements.txt
# put some pictures in assets/ (mallard.jpg ...). Without any, synthetic images are used.
python main.py --quick                 # ~1 min sanity run
python main.py                         # full run -> results/*.csv + *.png
python main.py --only scale rotation   # chosen experiments (also: resolution sweep benchmark)
python main.py --hpatches /path/to/hpatches-sequences-release   # real viewpoint/illumination data
python main.py --build-dataset         # also dump every transformed image + homography
```

## Methodology

### Data (and why)
1. **Controlled transformations of your own pictures.** Each transform has an *exactly known* homography
   `H` mapping reference pixels to transformed pixels, so correctness can be computed without manual labelling.
   One nuisance at a time → we can attribute failures to a cause.
   - scale uses OpenCV's pixel-centre convention `x' = s(x+½) − ½` (a naive `x' = s·x` would add a
     systematic half-pixel error that looks like a detector flaw);
   - warped images have black borders: a mask (eroded by 5 px) excludes them, otherwise artificial corners
     would pollute the results.
2. **HPatches** (optional): real viewpoint and illumination changes with ground-truth homographies, to check that
   conclusions drawn on synthetic transforms hold on real data.
3. Averages are taken **over images**, and the shaded areas in plots are the standard deviation across images:
   a result that depends on one lucky picture is visible.

### Metrics
Let `A` be keypoints of the reference image, `B` those of the transformed image, `H` the ground truth.
Only keypoints in the *common visible area* are considered (projected inside the other image and its mask).

* **Correspondence** `a ↔ b`: `‖H·a − b‖ < ε` (ε = 3 px) **and** the sizes agree: `σ_b / (σ_a · s(a)) ∈ [1/1.5, 1.5]`,
  where `s(a) = √|det J_H(a)|` is the local scale change given by the Jacobian of the homography
  (for a homography `(u,v) = (h₁·p, h₂·p)/w`, so `∂u/∂x = (h₁₁ − u·h₃₁)/w`, etc.).
* **Repeatability** `= min(#A matched, #B matched) / min(#A visible, #B visible)` (Mikolajczyk & Schmid).
  Matched keypoints are counted on both sides because SIFT can output several keypoints at the same
  position with different orientations.
* **Localisation error**: mean `‖H·a − b‖` over correspondences (sub-pixel accuracy).
* **Scale error**: median `|log₂(σ_b / (σ_a · s(a)))|` (0 = perfect scale covariance; log so that ×2 and ÷2 weigh the same).
* **Orientation error**: median `|wrap(θ_b − θ_pred)|` with `θ_pred` the direction of `J_H·(cosθ_a, sinθ_a)`.
* **Descriptor matching**: brute force kNN (k = 2) in the descriptor space (L2 for SIFT, Hamming for binary),
  Lowe's ratio test `d₁ < 0.8·d₂`. A match is *correct* if it lands within ε of the projection.
  - **precision** = correct / matches
  - **matching score** = correct / min(#A visible, #B visible) — captures detector *and* descriptor quality
  - **recall** = correct / #ground-truth correspondences
* **Homography estimation (task level)**: RANSAC (3 px) on the matches; error = mean displacement of the 4 image corners
  between estimated and true `H`, **measured in the reference frame** so that it doesn't shrink artificially
  when the test image is downscaled. Success if < 3 px. This is the metric that matters for the applications
  listed in "Why" below.

### Performance monitoring
`profiling.py` reports median detect / describe time, ms per keypoint, CPU time (→ average number of busy cores),
peak RSS, and GPU utilisation / memory if `nvidia-smi` exists. Also: speed-up versus `cv2.setNumThreads`.
Note: OpenCV's SIFT is **CPU only** (`cv2.cuda` has no SIFT), so the GPU numbers are expected to be zero; they
are monitored to compare against GPU implementations (e.g. PyTorch/Kornia SIFT, `cv2.cuda.ORB`) if added later.

## Tests
| experiment | what it checks | expectation for SIFT |
|---|---|---|
| `resolution` | keypoint count / sizes vs resolution (original script, with sizes expressed in **original pixels**: `size / ratio`) | fewer small keypoints at low resolution, large ones preserved |
| `scale` | scale invariance (×0.1 … ×2) | high repeatability, scale error ≈ 0, slow decay when the image gets tiny |
| `rotation` | rotation invariance (0–180°) | flat curves, orientation error ≈ 0 |
| `perspective` | partial affine/viewpoint invariance | graceful degradation (SIFT is only *locally* affine invariant) |
| `brightness`, `contrast` | photometric robustness | near invariant (gradient-based, normalised descriptor) |
| `noise`, `blur`, `jpeg` | degradation robustness | gradual decay; blur removes fine scales |
| `sweep` | SIFT hyper-parameters (`contrastThreshold`, `edgeThreshold`, `nOctaveLayers`, `sigma`) | trade-off # keypoints / quality / time |
| `benchmark` | time, CPU, memory, threads | time ∝ pixels |

Sanity checks built in (observed on the quick run): identity transform gives repeatability = 1 and all errors = 0;
rotation orientation error is below 1°, which validates the geometry code before trusting any other result.

## Caveats / things to keep in mind
- Detectors return different numbers of keypoints (SIFT unlimited, ORB capped at 2000 here). Compare
  *rates* (repeatability, precision) but remember counts differ; use `n_kp_test` plots.
- The ground truth of `perspective` assumes a planar scene, which is exactly true for a warped image.
- `ε`, the ratio threshold and the scale tolerance are in `config.py`; changing them changes absolute values, so
  report them with results.
- Results on a handful of pictures say little: use ≥ 10 varied images (textures, objects, repetitive patterns)
  or HPatches.

## Steps/Roadmap
- [x] Understand SIFT (see [SIFT](#sift))
- [x] Ground-truth transformations and metrics
- [x] Comparison with ORB / AKAZE / BRISK
- [x] Performance monitoring
- [ ] Run on HPatches and a personal dataset (repetitive textures, low light, ...)
- [ ] Compare with a GPU SIFT implementation
- [ ] Look at descriptor distinctiveness (distance distributions of true vs false matches, ROC curves)

## SIFT

### What it does (purpose of the algorithm)
The SIFT Algorithm detects features in a picture. As from its name, it is stable under scaling changes, but also rotation and viewpoint.

### Why (reasons, interest of the feature)
Feature detection allows object detection, for example to identify the contents of a picture. It also allows matching the same object in different pictures to recreate a trajectory in a temporal dimension, or identify some camera transformations.

### How (internal working of the algorithm)
The algorithm relies on a feature detection on the target image at multiple scales.
First, keypoint candidates are obtained from a set of gaussian blurred versions of the image (difference of Gaussians).
Then the candidates are sorted and unstable/weak keypoints are removed (low contrast, edge responses).
Each keypoint is then assigned a direction.
Finally, a finer description of the keypoint is generated:
- 16×16 neighborhood around the keypoint is selected.
- The neighborhood is divided into sixteen 4×4 subregions.
- An 8-bin orientation histogram is computed for each subregion.
- The resulting 128-dimensional descriptor uniquely represents the keypoint.

### Sources/Documentation
[algo](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Algorithm)  
[gfg opencv](https://www.geeksforgeeks.org/machine-learning/sift-interest-point-detector-using-python-opencv/)  
[OpenCV KeyPoint](https://docs.opencv.org/3.4.20/d2/d29/classcv_1_1KeyPoint.html)  
Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004  
Mikolajczyk & Schmid, *A performance evaluation of local descriptors*, TPAMI 2005  
Balntas et al., *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*, CVPR 2017
