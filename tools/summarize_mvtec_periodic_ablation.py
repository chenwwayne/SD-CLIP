#!/usr/bin/env python3
"""Aggregate completed MVTec-Periodic C0-C4 metrics."""

import argparse
import csv
import json
import statistics
from pathlib import Path

METHODS = ("c0_aa_clip", "c1_generic_prompts", "c2_apsf", "c3_apsf_kan_no_memory", "c4_sd_clip_snprb")
METRICS = ("image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "pixel_f1_max")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    raw = []
    for method in METHODS:
        for metrics_path in sorted((root / method).glob("seed*/**/metrics.json")):
            seed = int(metrics_path.parts[-3][4:])
            direction, shot_name = metrics_path.parts[-2].rsplit("_shot", 1)
            source, target = direction.split("_to_")
            item = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw.append({"method": method, "seed": seed, "shot": int(shot_name), "direction": direction, "source": source, "target": target, **{k: item[k] for k in METRICS}})
    expected = len(METHODS) * 3 * 6 * 2
    if len(raw) != expected:
        raise SystemExit(f"expected {expected} metrics, found {len(raw)}")
    fields = ["method", "seed", "shot", "direction", "source", "target", *METRICS]
    write_csv(root / "records.csv", raw, fields)

    direction_rows = []
    for method in METHODS:
        for shot in sorted({r["shot"] for r in raw}):
            for direction in sorted({r["direction"] for r in raw}):
                group = [r for r in raw if r["method"] == method and r["shot"] == shot and r["direction"] == direction]
                row = {"method": method, "shot": shot, "direction": direction, "n_seeds": len(group)}
                row.update({metric: statistics.mean(r[metric] for r in group) for metric in METRICS})
                direction_rows.append(row)
    write_csv(root / "summary_by_direction.csv", direction_rows, ["method", "shot", "direction", "n_seeds", *METRICS])

    seed_rows = []
    for method in METHODS:
        for seed in sorted({r["seed"] for r in raw}):
            for shot in sorted({r["shot"] for r in raw}):
                group = [r for r in raw if r["method"] == method and r["seed"] == seed and r["shot"] == shot]
                row = {"method": method, "seed": seed, "shot": shot, "n_directions": len(group)}
                row.update({metric: statistics.mean(r[metric] for r in group) for metric in METRICS})
                seed_rows.append(row)
    write_csv(root / "summary_by_seed.csv", seed_rows, ["method", "seed", "shot", "n_directions", *METRICS])

    final_rows = []
    for method in METHODS:
        for shot in sorted({r["shot"] for r in raw}):
            group = [r for r in seed_rows if r["method"] == method and r["shot"] == shot]
            row = {"method": method, "shot": shot, "n_seeds": len(group), "aggregation": "mean_over_directions_then_seed_mean_std"}
            for metric in METRICS:
                values = [r[metric] for r in group]
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            final_rows.append(row)
    write_csv(root / "summary_bidirectional.csv", final_rows, ["method", "shot", "n_seeds", "aggregation", *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")]])
    print(json.dumps({"records": len(raw), "outputs": ["records.csv", "summary_by_direction.csv", "summary_by_seed.csv", "summary_bidirectional.csv"]}, indent=2))


if __name__ == "__main__":
    main()
