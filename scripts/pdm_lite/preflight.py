#!/usr/bin/env python3
"""Validate the assets needed for CARLA Garage + FMSR training."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.data_root, args.checkpoint, args.anchors, args.stats, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    route = next((p for p in args.data_root.glob("*/*")
                  if (p / "measurements").is_dir() and (p / "lidar").is_dir()), None)
    if route is None:
        raise ValueError(
            f"{args.data_root} is not a raw CARLA Garage dataset root; expected "
            "SCENARIO/ROUTE/{measurements,lidar,rgb}"
        )
    measurement = next((route / "measurements").glob("*.json.gz"), None)
    if measurement is None:
        raise ValueError(f"No measurement JSON files found under {route}")
    with gzip.open(measurement, "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    missing = {"route", "target_speed", "speed"} - sample.keys()
    if missing:
        raise ValueError(f"PDM-Lite measurement is missing keys: {sorted(missing)}")

    anchors = np.load(args.anchors)
    if anchors.shape != (60, 11, 2) or not np.isfinite(anchors).all():
        raise ValueError(f"Expected finite anchors [60, 11, 2], got {anchors.shape}")
    stats = np.load(args.stats)
    if stats["mean"].shape != (11, 2) or stats["std"].shape != (11, 2):
        raise ValueError("Expected mean/std with shape [11, 2]")
    if (stats["std"] <= 0).any() or not np.isfinite(stats["std"]).all():
        raise ValueError("Trajectory std must be finite and positive")
    manifest_count = sum(1 for line in args.manifest.open(encoding="utf-8")
                         if line.strip() and not line.startswith("#"))
    if manifest_count < 60:
        raise ValueError(f"Key-frame manifest contains only {manifest_count} frames")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint is not a non-empty state_dict")
    backbone_keys = [key for key in state if key.startswith("backbone.")]
    planner_keys = [key for key in state if key.startswith(("join.", "checkpoint_query"))]
    if not backbone_keys or not planner_keys:
        raise ValueError(
            "Checkpoint does not look like a CARLA Garage TF++ checkpoint. "
            "FMSR tfv6 checkpoints are not compatible."
        )

    free_gib = shutil.disk_usage(args.data_root).free / 1024**3
    print(f"dataset sample: {route}")
    print(f"manifest frames: {manifest_count:,}")
    print(f"checkpoint tensors: {len(state):,} ({len(backbone_keys):,} backbone)")
    print(f"anchors/stats: {anchors.shape} / {stats['mean'].shape}")
    print(f"free space on dataset filesystem: {free_gib:.1f} GiB")
    print("PDM-Lite FMSR asset preflight passed.")


if __name__ == "__main__":
    main()
