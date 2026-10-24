#!/usr/bin/env python3
"""Summarize per-frame FMSR inference latency measurements."""

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    "preprocess_ms",
    "tensor_transfer_ms",
    "model_planning_ms",
    "control_postprocess_ms",
    "image_to_control_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()

    with args.csv_file.open(newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))

    if not rows:
        raise SystemExit(f"No latency samples found in {args.csv_file}")

    print(f"samples={len(rows)} routes={len({row['route'] for row in rows})}")
    print(f"{'stage':28s} {'mean':>10s} {'p50':>10s} {'p95':>10s} {'p99':>10s} {'max':>10s}")
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        print(
            f"{metric:28s} "
            f"{values.mean():9.2f}ms "
            f"{np.percentile(values, 50):9.2f}ms "
            f"{np.percentile(values, 95):9.2f}ms "
            f"{np.percentile(values, 99):9.2f}ms "
            f"{values.max():9.2f}ms"
        )

    end_to_end = np.asarray([float(row["image_to_control_ms"]) for row in rows])
    print(f"mean throughput={1000.0 / end_to_end.mean():.2f} FPS")


if __name__ == "__main__":
    main()
