#!/usr/bin/env python3
"""
Von Kamerabild zu CNC-Pfad (Demo-Pipeline)

Dieser Prototyp uebertraegt die relevanten C++-Ideen aus libromi nach Python:
- filter_mask: Erosion gefolgt von Dilatation
- compute_connected_components: Label-Bild aus Binarmaske
- calculate_centers: Superpixel-basierte Zentren (SLIC), mit Fallback
- sort_centers: Zentren nach Connected-Component gruppieren

Ausgaben:
- 01_mask_raw.png
- 02_mask_filtered.png
- 03_components.png
- 03b_component_filter.png
- 04_centers.png
- 05_cnc_path.png
- cnc_path.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from skimage.segmentation import slic

    HAS_SLIC = True
except Exception:
    HAS_SLIC = False


Point2D = Tuple[int, int]  # (x, y)
Point3D = Tuple[float, float, float]


@dataclass
class PipelineConfig:
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    morph_iterations: int
    max_centers: int
    z_depth: float
    use_slic: bool
    min_weed_area_px: int
    max_weed_area_px: int
    max_weed_major_axis_px: float
    depth_image_path: str | None
    max_weed_height_rel: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kamerabild -> Unkrautmaske -> CNC-Pfad")
    parser.add_argument("--input", required=True, help="Pfad zum Eingangsbild")
    parser.add_argument("--outdir", default="output_cnc_path", help="Ausgabeordner")

    parser.add_argument("--h-low", type=int, default=30, help="HSV H min")
    parser.add_argument("--s-low", type=int, default=35, help="HSV S min")
    parser.add_argument("--v-low", type=int, default=25, help="HSV V min")
    parser.add_argument("--h-high", type=int, default=95, help="HSV H max")
    parser.add_argument("--s-high", type=int, default=255, help="HSV S max")
    parser.add_argument("--v-high", type=int, default=255, help="HSV V max")

    parser.add_argument("--morph-iterations", type=int, default=2, help="Erode/Dilate-Iterationen")
    parser.add_argument("--max-centers", type=int, default=120, help="Maximale Anzahl Zielpunkte")
    parser.add_argument("--z-depth", type=float, default=-0.01, help="Z-Tiefe fuer CNC-Punkte")
    parser.add_argument("--min-weed-area", type=int, default=40, help="Minimale Flaeche einer Unkraut-Instanz in Pixeln")
    parser.add_argument("--max-weed-area", type=int, default=3500, help="Maximale Flaeche einer Unkraut-Instanz in Pixeln")
    parser.add_argument(
        "--max-weed-major-axis",
        type=float,
        default=120.0,
        help="Maximale Hauptachsenlaenge einer Unkraut-Instanz in Pixeln",
    )
    parser.add_argument(
        "--depth-image",
        default=None,
        help="Optionales Tiefenbild fuer zusaetzliche Hoehenfilterung",
    )
    parser.add_argument(
        "--max-weed-height-rel",
        type=float,
        default=0.12,
        help="Maximale relative Komponentenhoehe (0..1) gegenueber Bodenreferenz",
    )
    parser.add_argument(
        "--no-slic",
        action="store_true",
        help="SLIC deaktivieren und KMeans-Fallback verwenden",
    )
    return parser.parse_args()


def segment_weed_mask_bgr(image_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def filter_mask(mask_u8: np.ndarray, iterations: int) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=iterations)
    filtered = cv2.dilate(eroded, kernel, iterations=iterations)
    return filtered


def compute_connected_components(mask_u8: np.ndarray) -> Tuple[np.ndarray, int]:
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    return labels, num_labels


def compute_major_axis_length(coords_xy: np.ndarray) -> float:
    if len(coords_xy) < 2:
        return 0.0
    centered = coords_xy.astype(np.float32) - np.mean(coords_xy, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigenvals, _ = np.linalg.eigh(cov)
    major_var = float(np.max(eigenvals))
    if major_var < 0:
        major_var = 0.0
    return 4.0 * math.sqrt(major_var)


def load_depth_map(depth_image_path: str | None, shape_hw: Tuple[int, int]) -> np.ndarray | None:
    if depth_image_path is None:
        return None

    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Konnte Tiefenbild nicht laden: {depth_image_path}")

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)
    dmin = float(np.min(depth))
    dmax = float(np.max(depth))
    if dmax - dmin < 1e-8:
        depth = np.zeros_like(depth, dtype=np.float32)
    else:
        depth = (depth - dmin) / (dmax - dmin)

    h, w = shape_hw
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth


def filter_components_for_weeding(
    labels: np.ndarray,
    num_labels: int,
    cfg: PipelineConfig,
    depth_map: np.ndarray | None,
) -> Tuple[np.ndarray, Dict[int, str], List[int]]:
    kept_mask = np.zeros(labels.shape, dtype=np.uint8)
    rejected_reasons: Dict[int, str] = {}
    kept_labels: List[int] = []

    background_depth_ref = None
    if depth_map is not None:
        background = depth_map[labels == 0]
        if background.size > 0:
            background_depth_ref = float(np.median(background))

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        area = len(xs)
        if area == 0:
            continue

        if area < cfg.min_weed_area_px:
            rejected_reasons[label] = "too_small"
            continue
        if area > cfg.max_weed_area_px:
            rejected_reasons[label] = "too_large_area"
            continue

        coords = np.column_stack([xs, ys])
        major_axis = compute_major_axis_length(coords)
        if major_axis > cfg.max_weed_major_axis_px:
            rejected_reasons[label] = "too_large_major_axis"
            continue

        if depth_map is not None and background_depth_ref is not None:
            comp_depth = depth_map[ys, xs]
            comp_depth_median = float(np.median(comp_depth))
            rel_height = comp_depth_median - background_depth_ref
            if rel_height > cfg.max_weed_height_rel:
                rejected_reasons[label] = "too_tall"
                continue

        kept_mask[ys, xs] = 255
        kept_labels.append(label)

    return kept_mask, rejected_reasons, kept_labels


def calculate_centers_slic(mask_u8: np.ndarray, max_centers: int) -> List[Point2D]:
    mask_rgb = np.dstack([mask_u8, mask_u8, mask_u8])
    labels = slic(
        mask_rgb,
        n_segments=max_centers,
        compactness=20.0,
        start_label=1,
        convert2lab=False,
        channel_axis=-1,
    )

    centers: List[Point2D] = []
    for seg_id in np.unique(labels):
        ys, xs = np.where(labels == seg_id)
        if len(xs) == 0:
            continue

        segment_values = mask_u8[ys, xs]
        if float(np.mean(segment_values)) < 127.0:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        centers.append((cx, cy))

    return centers


def calculate_centers_kmeans(mask_u8: np.ndarray, max_centers: int) -> List[Point2D]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    k = min(max_centers, len(points))
    if k <= 0:
        return []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
    _compactness, _labels, centers = cv2.kmeans(
        data=points,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=2,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    return [(int(c[0]), int(c[1])) for c in centers]


def calculate_centers(mask_u8: np.ndarray, max_centers: int, prefer_slic: bool) -> List[Point2D]:
    if prefer_slic and HAS_SLIC:
        return calculate_centers_slic(mask_u8, max_centers)
    return calculate_centers_kmeans(mask_u8, max_centers)


def sort_centers_by_component(centers: Sequence[Point2D], component_labels: np.ndarray) -> List[List[Point2D]]:
    grouped: Dict[int, List[Point2D]] = {}
    h, w = component_labels.shape[:2]

    for (x, y) in centers:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        label = int(component_labels[y, x])
        if label == 0:
            continue
        grouped.setdefault(label, []).append((x, y))

    return list(grouped.values())


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def nearest_neighbor_order(points: Sequence[Point2D], start: Point2D | None = None) -> List[Point2D]:
    if not points:
        return []

    remaining = list(points)
    if start is None:
        current = remaining.pop(0)
    else:
        nearest_idx = int(np.argmin([distance(start, p) for p in remaining]))
        current = remaining.pop(nearest_idx)

    ordered = [current]
    while remaining:
        nearest_idx = int(np.argmin([distance(current, p) for p in remaining]))
        current = remaining.pop(nearest_idx)
        ordered.append(current)

    return ordered


def build_cnc_path(grouped_centers: Sequence[Sequence[Point2D]], z_depth: float) -> List[Point3D]:
    path2d: List[Point2D] = []
    current: Point2D | None = None

    groups = [list(g) for g in grouped_centers if len(g) > 0]

    while groups:
        if current is None:
            group_idx = int(np.argmax([len(g) for g in groups]))
        else:
            group_idx = int(
                np.argmin(
                    [min(distance(current, p) for p in g) if g else float("inf") for g in groups]
                )
            )

        selected_group = groups.pop(group_idx)
        ordered_group = nearest_neighbor_order(selected_group, start=current)
        path2d.extend(ordered_group)
        current = ordered_group[-1]

    return [(float(x), float(y), float(z_depth)) for (x, y) in path2d]


def render_components(component_labels: np.ndarray) -> np.ndarray:
    labels_u8 = np.uint8((component_labels % 256))
    colored = cv2.applyColorMap(labels_u8, cv2.COLORMAP_TURBO)
    colored[component_labels == 0] = (0, 0, 0)
    return colored


def render_component_filter_overlay(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    kept_labels: Sequence[int],
) -> np.ndarray:
    out = image_bgr.copy()
    kept_set = set(kept_labels)
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        color = (0, 200, 0) if int(label) in kept_set else (0, 0, 220)
        out[ys, xs] = (0.35 * out[ys, xs] + 0.65 * np.array(color, dtype=np.float32)).astype(np.uint8)

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        tag = "K" if int(label) in kept_set else "X"
        cv2.putText(
            out,
            tag,
            (cx - 5, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_centers(base_bgr: np.ndarray, centers: Sequence[Point2D]) -> np.ndarray:
    out = base_bgr.copy()
    for (x, y) in centers:
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out


def draw_path(base_bgr: np.ndarray, path_xyz: Sequence[Point3D]) -> np.ndarray:
    out = base_bgr.copy()
    points2d = [(int(p[0]), int(p[1])) for p in path_xyz]

    for i, p in enumerate(points2d):
        cv2.circle(out, p, 3, (0, 255, 255), -1)
        if i > 0:
            cv2.line(out, points2d[i - 1], p, (255, 64, 64), 2)

    for i, (x, y) in enumerate(points2d[:200]):
        if i % 5 == 0:
            cv2.putText(
                out,
                str(i),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def save_path_csv(path_csv: str, path_xyz: Sequence[Point3D]) -> None:
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "z"])
        for i, (x, y, z) in enumerate(path_xyz):
            writer.writerow([i, f"{x:.3f}", f"{y:.3f}", f"{z:.4f}"])


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        hsv_lower=(args.h_low, args.s_low, args.v_low),
        hsv_upper=(args.h_high, args.s_high, args.v_high),
        morph_iterations=args.morph_iterations,
        max_centers=args.max_centers,
        z_depth=args.z_depth,
        use_slic=(not args.no_slic),
        min_weed_area_px=args.min_weed_area,
        max_weed_area_px=args.max_weed_area,
        max_weed_major_axis_px=args.max_weed_major_axis,
        depth_image_path=args.depth_image,
        max_weed_height_rel=args.max_weed_height_rel,
    )

    ensure_outdir(args.outdir)

    image_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Konnte Bild nicht laden: {args.input}")

    mask_raw = segment_weed_mask_bgr(image_bgr, cfg)
    mask_filtered = filter_mask(mask_raw, cfg.morph_iterations)

    depth_map = load_depth_map(cfg.depth_image_path, mask_filtered.shape[:2])

    labels_all, num_labels_all = compute_connected_components(mask_filtered)
    mask_selected, rejected_reasons, kept_labels = filter_components_for_weeding(
        labels_all, num_labels_all, cfg, depth_map
    )

    component_labels, num_labels = compute_connected_components(mask_selected)
    centers = calculate_centers(mask_selected, cfg.max_centers, cfg.use_slic)
    grouped_centers = sort_centers_by_component(centers, component_labels)
    path_xyz = build_cnc_path(grouped_centers, cfg.z_depth)

    components_viz = render_components(component_labels)
    filter_overlay = render_component_filter_overlay(image_bgr, labels_all, kept_labels)
    centers_viz = draw_centers(image_bgr, centers)
    path_viz = draw_path(image_bgr, path_xyz)

    cv2.imwrite(os.path.join(args.outdir, "01_mask_raw.png"), mask_raw)
    cv2.imwrite(os.path.join(args.outdir, "02_mask_filtered.png"), mask_filtered)
    cv2.imwrite(os.path.join(args.outdir, "03_components.png"), components_viz)
    cv2.imwrite(os.path.join(args.outdir, "03b_component_filter.png"), filter_overlay)
    cv2.imwrite(os.path.join(args.outdir, "04_centers.png"), centers_viz)
    cv2.imwrite(os.path.join(args.outdir, "05_cnc_path.png"), path_viz)
    save_path_csv(os.path.join(args.outdir, "cnc_path.csv"), path_xyz)

    print("Pipeline abgeschlossen")
    print(f"Connected Components gesamt (inkl. Hintergrund): {num_labels_all}")
    print(f"Akzeptierte Komponenten: {len(kept_labels)}")
    print(f"Verworfene Komponenten: {len(rejected_reasons)}")
    if rejected_reasons:
        by_reason: Dict[str, int] = {}
        for reason in rejected_reasons.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print(f"Verwerfungsgruende: {by_reason}")
    print(f"Anzahl Zentren: {len(centers)}")
    print(f"Anzahl CNC-Wegpunkte: {len(path_xyz)}")
    print(f"Ausgabeordner: {args.outdir}")
    if cfg.use_slic and not HAS_SLIC:
        print("Hinweis: skimage nicht gefunden, KMeans-Fallback verwendet.")


if __name__ == "__main__":
    main()
