# =============================================================================
# benchmark_face_detection.py  —  MTCNN implementation benchmark (Stage 2 prep)
# =============================================================================
# PURPOSE:
#   FR-02 specifies the MTCNN algorithm, not a specific package. This repo
#   currently ships the TensorFlow-based `mtcnn` package (requirements.txt),
#   which drags in tensorflow_cpu alongside the torch stack already required
#   for the CV/NLP classifiers — two deep learning frameworks for one
#   pipeline. facenet-pytorch ships its own MTCNN implementation on top of
#   torch, which we already depend on.
#
#   This script benchmarks both on real frames pulled from this project's
#   own video corpus (not synthetic test images) — same frame in, same
#   frame out, per-frame CPU wall-clock time and whether the two detectors
#   agree on whether/where a face is present. Whichever wins gets written
#   into requirements.txt; the loser is dropped so we don't carry both.
#
# USAGE:
#   python benchmark_face_detection.py
# =============================================================================

import time
from pathlib import Path

import cv2
import numpy as np

FRAMES_DIR = Path(
    "/tmp/claude-1000/-home-arukwe-watchdog/32334010-de4e-4707-8c53-ff3a6b9b3099"
    "/scratchpad/bench_frames"
)


def iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes, for agreement scoring."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def run_tf_mtcnn(frames_rgb):
    from mtcnn import MTCNN
    detector = MTCNN(min_face_size=40)

    times, boxes = [], []
    for rgb in frames_rgb:
        t0 = time.perf_counter()
        detections = detector.detect_faces(rgb)
        times.append(time.perf_counter() - t0)
        if detections:
            best = max(detections, key=lambda d: d.get("confidence", 0))
            boxes.append(tuple(best["box"]))
        else:
            boxes.append(None)
    return times, boxes


def run_facenet_pytorch_mtcnn(frames_rgb):
    import torch
    from facenet_pytorch import MTCNN

    torch.set_num_threads(1)   # match single-threaded CPU assumption used elsewhere
    detector = MTCNN(keep_all=False, device="cpu")

    times, boxes = [], []
    for rgb in frames_rgb:
        t0 = time.perf_counter()
        box, _ = detector.detect(rgb)
        times.append(time.perf_counter() - t0)
        if box is not None:
            x1, y1, x2, y2 = box[0]
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        else:
            boxes.append(None)
    return times, boxes


def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {FRAMES_DIR} — extract them first.")

    frames_bgr = [cv2.imread(str(p)) for p in frame_paths]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    names = [p.name for p in frame_paths]

    print(f"Benchmarking on {len(frames_rgb)} real frames from the video corpus:")
    for n, f in zip(names, frames_bgr):
        print(f"  {n}: {f.shape[1]}x{f.shape[0]}")
    print()

    print("Loading + warming up TF mtcnn...")
    # Warm-up call (first call pays model-load + TF graph-build cost — exclude
    # that one-time cost from the per-frame numbers, but report it separately)
    warm_t0 = time.perf_counter()
    tf_times_warm, _ = run_tf_mtcnn(frames_rgb[:1])
    tf_warmup_s = time.perf_counter() - warm_t0
    tf_times, tf_boxes = run_tf_mtcnn(frames_rgb)

    print("Loading + warming up facenet-pytorch MTCNN...")
    warm_t0 = time.perf_counter()
    fp_times_warm, _ = run_facenet_pytorch_mtcnn(frames_rgb[:1])
    fp_warmup_s = time.perf_counter() - warm_t0
    fp_times, fp_boxes = run_facenet_pytorch_mtcnn(frames_rgb)

    print()
    print(f"{'frame':<16} {'tf_mtcnn(s)':>12} {'facenet_pt(s)':>14} {'agree(IoU>0.5)':>16}")
    agreements = []
    for n, tt, ft, tb, fb in zip(names, tf_times, fp_times, tf_boxes, fp_boxes):
        if tb is None and fb is None:
            agree = "both:none"
        elif tb is None or fb is None:
            agree = "DISAGREE(one none)"
        else:
            score = iou(tb, fb)
            agree = f"iou={score:.2f}"
            agreements.append(score)
        print(f"{n:<16} {tt:>12.3f} {ft:>14.3f} {agree:>16}")

    print()
    print(f"First-call warmup (model load, one-time): "
          f"tf_mtcnn={tf_warmup_s:.2f}s  facenet_pytorch={fp_warmup_s:.2f}s")
    print(f"Per-frame CPU time — tf_mtcnn:      "
          f"mean={np.mean(tf_times):.3f}s  median={np.median(tf_times):.3f}s")
    print(f"Per-frame CPU time — facenet_pytorch: "
          f"mean={np.mean(fp_times):.3f}s  median={np.median(fp_times):.3f}s")
    if agreements:
        print(f"Mean IoU on frames where both detected a face: {np.mean(agreements):.3f} "
              f"({len(agreements)}/{len(names)} frames)")

    winner = "facenet-pytorch" if np.median(fp_times) < np.median(tf_times) else "tf_mtcnn"
    print()
    print(f"WINNER (lower median per-frame CPU time): {winner}")


if __name__ == "__main__":
    main()
