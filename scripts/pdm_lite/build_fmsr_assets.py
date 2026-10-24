#!/usr/bin/env python3
"""Build speed-aware FMSR anchors and a PDM-Lite key-frame manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

GARAGE_CODE = Path(__file__).resolve().parents[2] / "3rd_party/carla_garage/team_code"
sys.path.insert(0, str(GARAGE_CODE))
from route_geometry import prepare_route, smooth_route
from config import GlobalConfig

INPUT_ROUTE_POINTS = GlobalConfig().num_route_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="CARLA Garage PDM-Lite data directory containing scenario folders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=60)
    parser.add_argument("--route-points", type=int, default=10)
    parser.add_argument("--ordinary-keep-ratio", type=float, default=0.14)
    parser.add_argument("--speed-change", type=float, default=0.1, help="m/s")
    parser.add_argument("--angle-change", type=float, default=0.5, help="degrees")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="Debug limit; zero processes all valid routes.")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def is_trainable_route(route: Path) -> bool:
    result_file = route / "results.json.gz"
    if route.name.startswith("FAILED_") or not result_file.is_file():
        return False
    result = load_json_gz(result_file)
    scores = result.get("scores", {})
    infractions = result.get("infractions", {})
    min_speed = infractions.get("min_speed_infractions", [])
    imperfect = scores.get("score_composed", 0.0) < 100.0
    only_min_speed = result.get("num_infractions", -1) == len(min_speed)
    failed_states = {
        "Failed", "Failed - Agent couldn't be set up", "Failed - Simulation crashed",
        "Failed - Agent crashed",
    }
    return (not imperfect or only_min_speed) and result.get("status") not in failed_states


def resample_route(route: np.ndarray, count: int) -> np.ndarray:
    """Unaugmented training label, with the same input horizon as CARLA_Data."""
    horizon = INPUT_ROUTE_POINTS
    if not 1 <= count <= horizon:
        raise ValueError(f"route-points must be between 1 and {horizon}")
    return smooth_route(prepare_route(route, horizon), horizon)[:count].astype(np.float32)


def route_angles(route: np.ndarray) -> np.ndarray:
    delta = np.diff(np.vstack((np.zeros((1, 2), dtype=route.dtype), route)), axis=0)
    return np.arctan2(delta[:, 1], delta[:, 0])


def wrapped_angle_delta(left: np.ndarray, right: np.ndarray) -> float:
    delta = (left - right + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.max(np.abs(delta)))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ordinary_keep_ratio <= 1.0:
        raise ValueError("--ordinary-keep-ratio must be in [0, 1]")
    routes = sorted(path for path in args.data_root.glob("*/*") if path.is_dir())
    routes = [route for route in routes if is_trainable_route(route)]
    if args.max_routes:
        routes = routes[: args.max_routes]
    if not routes:
        raise FileNotFoundError(f"No valid PDM-Lite routes under {args.data_root}")

    rng = np.random.default_rng(args.seed)
    angle_threshold = math.radians(args.angle_change)
    samples: list[np.ndarray] = []
    manifest: list[str] = []
    scanned = 0
    key_frames = 0

    for route_index, route_dir in enumerate(routes, 1):
        previous_speed = None
        previous_angles = None
        measurements = sorted((route_dir / "measurements").glob("*.json.gz"))
        for measurement_path in measurements:
            measurement = load_json_gz(measurement_path)
            if "route" not in measurement or "target_speed" not in measurement:
                continue
            route = resample_route(np.asarray(measurement["route"]), args.route_points)
            speed = max(float(measurement["target_speed"]), 0.0)
            angles = route_angles(route)
            is_key = previous_speed is None
            if previous_speed is not None:
                is_key = abs(speed - previous_speed) > args.speed_change
                is_key = is_key or wrapped_angle_delta(angles, previous_angles) > angle_threshold
            keep = is_key or rng.random() < args.ordinary_keep_ratio
            previous_speed = speed
            previous_angles = angles
            scanned += 1
            if not keep:
                continue
            state = np.zeros((args.route_points + 1, 2), dtype=np.float32)
            state[: args.route_points] = route
            state[-1, 0] = speed
            samples.append(state)
            if is_key:
                key_frames += 1
            manifest.append(
                f"{route_dir.parent.name}/{route_dir.name}/{measurement_path.name[:4]}"
            )
        if route_index % 100 == 0:
            print(f"processed {route_index}/{len(routes)} routes; kept {len(samples)}/{scanned} frames",
                  flush=True)

    if len(samples) < args.clusters:
        raise ValueError(f"Only {len(samples)} frames were retained for {args.clusters} clusters")
    states = np.stack(samples)
    flat = states.reshape(len(states), -1)
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init=10,
        reassignment_ratio=0.01,
    ).fit(flat)
    anchors = kmeans.cluster_centers_.reshape(args.clusters, args.route_points + 1, 2)
    anchors[:, -1, 1] = 0.0
    mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = states.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = args.output_dir / f"pdm_cp_kmeans_spd_{args.clusters}_{args.route_points}.npy"
    stats_path = args.output_dir / f"pdm_cp_stats_spd_{args.route_points}.npz"
    manifest_path = args.output_dir / "pdm_lite_keyframes.txt"
    metadata_path = args.output_dir / "pdm_lite_assets.json"
    np.save(anchor_path, anchors.astype(np.float32))
    np.savez(stats_path, mean=mean, std=std)
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    metadata = {
        "data_root": str(args.data_root.resolve()),
        "valid_routes": len(routes),
        "scanned_frames": scanned,
        "retained_frames": len(states),
        "key_frames": key_frames,
        "retained_ratio": len(states) / scanned,
        "clusters": args.clusters,
        "route_points": args.route_points,
        "ordinary_keep_ratio": args.ordinary_keep_ratio,
        "speed_change_mps": args.speed_change,
        "angle_change_degrees": args.angle_change,
        "seed": args.seed,
        "route_geometry": "carla_garage_shared_v1",
        "input_route_points": INPUT_ROUTE_POINTS,
        "inertia": float(kmeans.inertia_),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"anchors: {anchor_path}\nstats: {stats_path}\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
