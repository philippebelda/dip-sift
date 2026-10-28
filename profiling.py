"""Timing and resource monitoring (CPU, memory, optional NVIDIA GPU)."""
import resource
import shutil
import subprocess
import threading
from time import perf_counter, process_time

import cv2
import numpy as np

from detectors import detect_and_describe
from tools import to_gray

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None


def gpu_info():
    """What does this machine / this OpenCV build offer?
    Note: OpenCV's SIFT is CPU-only. cv2.cuda has no SIFT implementation, so the
    GPU is only monitored (and would matter for e.g. cv2.cuda.ORB or a PyTorch SIFT)."""
    info = {"opencv": cv2.__version__, "cuda_devices_in_opencv": 0, "nvidia_smi": None,
            "threads": cv2.getNumThreads()}
    try:
        info["cuda_devices_in_opencv"] = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=3)
            info["nvidia_smi"] = r.stdout.strip()
        except Exception:
            pass
    return info


def _gpu_sample():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        util, mem = r.stdout.strip().splitlines()[0].split(",")
        return float(util), float(mem)
    except Exception:
        return None


def _rss_mb():
    if psutil:
        return psutil.Process().memory_info().rss / 2 ** 20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux


class ResourceMonitor:
    """Context manager: wall time, CPU time (-> average number of busy cores),
    peak RSS and, if available, peak GPU utilisation / memory."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.has_gpu = shutil.which("nvidia-smi") is not None
        self.result = {}

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._rss = max(self._rss, _rss_mb())
            if self.has_gpu:
                s = _gpu_sample()
                if s:
                    self._gpu_util = max(self._gpu_util, s[0])
                    self._gpu_mem = max(self._gpu_mem, s[1])

    def __enter__(self):
        self._stop = threading.Event()
        self._rss, self._gpu_util, self._gpu_mem = _rss_mb(), 0.0, 0.0
        self._t0, self._c0 = perf_counter(), process_time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        wall, cpu = perf_counter() - self._t0, process_time() - self._c0
        self._stop.set()
        self._thread.join()
        self._rss = max(self._rss, _rss_mb())
        self.result = dict(wall_s=wall, cpu_s=cpu, avg_busy_cores=cpu / wall if wall else np.nan,
                           peak_rss_mb=self._rss)
        if self.has_gpu:
            self.result.update(gpu_util_max=self._gpu_util, gpu_mem_max_mb=self._gpu_mem)


def benchmark_detectors(detectors, base_img, scales=(0.25, 0.5, 1.0, 1.5), repeat=5):
    """Time vs image size, per detector."""
    rows = []
    for det in detectors:
        for s in scales:
            img = cv2.resize(base_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            gray = to_gray(img)
            detect_and_describe(det, gray)               # warm-up
            t_det, t_desc, n = [], [], 0
            with ResourceMonitor() as mon:
                for _ in range(repeat):
                    d = detect_and_describe(det, gray)
                    t_det.append(d.t_detect)
                    t_desc.append(d.t_describe)
                    n = len(d)
            mpx = gray.shape[0] * gray.shape[1] / 1e6
            rows.append(dict(detector=det.name, scale=s, megapixels=mpx, n_kp=n,
                             detect_ms=1e3 * np.median(t_det), describe_ms=1e3 * np.median(t_desc),
                             ms_per_kp=1e3 * (np.median(t_det) + np.median(t_desc)) / max(n, 1),
                             **mon.result))
    return rows


def benchmark_threads(det, base_img, thread_counts=(1, 2, 4, 8), repeat=5):
    """Does the detector scale with the number of CPU threads?"""
    gray = to_gray(base_img)
    original = cv2.getNumThreads()
    rows = []
    for n in thread_counts:
        cv2.setNumThreads(n)
        detect_and_describe(det, gray)
        times = [sum(x for x in (lambda d: (d.t_detect, d.t_describe))(detect_and_describe(det, gray)))
                 for _ in range(repeat)]
        rows.append(dict(detector=det.name, threads=n, total_ms=1e3 * np.median(times)))
    cv2.setNumThreads(original)
    base = rows[0]["total_ms"]
    for r in rows:
        r["speedup"] = base / r["total_ms"]
    return rows
