#!/usr/bin/env python3
"""Build reproducible, direction-independent AMOLED splits from raw images.

The legacy metadata in this repository is incomplete for AMOLED.  This builder
therefore scans the raw image/mask directories directly and writes only
job-local JSONL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DATASET_COUNTS = {
    "Active": (181, 266),
    "Cell": (138, 32),
}
SHOTS = (1, 2, 4, 8, 16, 32)
DIRECTIONS = (("Active", "Cell"), ("Cell", "Active"))

# Neutral public class names. The private imagery uses its own internal tags;
# re-map them locally with --class-name-map if you rebuild inside your lab.
DEFAULT_CLASS_NAMES = {"Active": "active", "Cell": "cell"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Root directory of the (private) AMOLED-ARRAY dataset, expected to "
            "contain Active/{images, mask} and Cell/{images, mask} sub-folders. "
            "Omit to use the bundled dummy generator as a smoke test."
        ),
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seeds", default=",".join(map(str, range(5))))
    p.add_argument("--shots", default=",".join(map(str, SHOTS)))
    p.add_argument("--directions", default="active_to_cell,cell_to_active")
    p.add_argument(
        "--class-name-map",
        default="",
        help=(
            "Optional internal-tag re-mapping, e.g. 'Active=<tag>,Cell=<tag>'. "
            "When omitted, neutral names 'active'/'cell' are used."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_class_name_map(text: str) -> dict[str, str]:
    mapping = dict(DEFAULT_CLASS_NAMES)
    for item in (x for x in text.split(",") if x.strip()):
        domain, _, name = item.partition("=")
        domain = domain.strip()
        name = name.strip()
        if domain not in mapping or not name:
            raise ValueError(f"invalid class-name map entry: {item!r}")
        mapping[domain] = name
    return mapping


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_domain(data_root: Path, dataset: str, class_names: dict[str, str]) -> list[dict]:
    root = data_root / dataset
    image_root = root / "images"
    mask_root = root / "mask"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"missing AMOLED image/mask directories under {root}")

    records = []
    for image in sorted(image_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        relative = image.relative_to(root).as_posix()
        label = 1 if relative.startswith("images/anomaly/") else 0
        mask_path = ""
        if label:
            candidate = mask_root / "anomaly" / f"{image.stem}.png"
            if not candidate.is_file():
                raise FileNotFoundError(f"missing mask for {image}")
            mask_path = candidate.relative_to(root).as_posix()
        records.append(
            {
                "image_path": relative,
                "mask_path": mask_path,
                "class_name": class_names[dataset],
                "label": label,
            }
        )

    expected = DATASET_COUNTS[dataset]
    actual = (
        sum(r["label"] == 0 for r in records),
        sum(r["label"] == 1 for r in records),
    )
    if actual != expected:
        raise RuntimeError(f"{dataset} count mismatch: expected {expected}, got {actual}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256(path)


def select_pool(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    normal = [r for r in records if r["label"] == 0]
    anomaly = [r for r in records if r["label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    pool = normal[:16] + anomaly[:16]
    remainder = normal[16:] + anomaly[16:]
    return pool, remainder


def direction_pairs(names: str):
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    mapping = {
        "active_to_cell": DIRECTIONS[0],
        "cell_to_active": DIRECTIONS[1],
    }
    for key, pair in mapping.items():
        if key in wanted:
            yield key, pair


def build(args: argparse.Namespace) -> dict:
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    class_names = parse_class_name_map(args.class_name_map)
    domains = {name: scan_domain(args.data_root, name, class_names) for name in ("Active", "Cell")}
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"seeds": seeds, "shots": shots, "directions": [], "splits": []}

    for direction, (source_name, target_name) in direction_pairs(args.directions):
        source_records = domains[source_name]
        target_records = domains[target_name]
        for seed in seeds:
            pool, remainder = select_pool(source_records, seed)
            normal_pool = [r for r in pool if r["label"] == 0]
            anomaly_pool = [r for r in pool if r["label"] == 1]
            for shot in shots:
                if shot <= 0 or shot > 32:
                    raise ValueError(f"invalid shot {shot}; expected 1..32")
                normal_n = (shot + 1) // 2
                anomaly_n = shot // 2
                support = normal_pool[:normal_n] + anomaly_pool[:anomaly_n]
                split_dir = output_root / direction / f"seed_{seed}" / f"shot_{shot}"
                split_dir.mkdir(parents=True, exist_ok=True)
                pool_hash = write_jsonl(split_dir / "source_pool.jsonl", pool)
                support_hash = write_jsonl(split_dir / "source_support.jsonl", support)
                unused_pool = [r for r in pool if r not in support]
                unused_pool_hash = write_jsonl(split_dir / "unused_pool_samples.jsonl", unused_pool)
                remainder_hash = write_jsonl(
                    split_dir / "unused_source_remainder.jsonl", remainder
                )
                target_hash = write_jsonl(split_dir / "target_test.jsonl", target_records)
                manifest = {
                    "schema_version": "amoled_directional_fulltarget_v2_no_source_validation",
                    "source_domain": f"AMOLED-{source_name}",
                    "target_domain": f"AMOLED-{target_name}",
                    "direction": direction,
                    "seed": seed,
                    "shot": shot,
                    "source_pool": {"normal": 16, "anomalous": 16, "sha256": pool_hash},
                    "source_support": {
                        "normal": normal_n,
                        "anomalous": anomaly_n,
                        "count": len(support),
                        "sha256": support_hash,
                    },
                    "unused_pool_samples": {"count": len(unused_pool), "sha256": unused_pool_hash},
                    "unused_source_remainder": {
                        "normal": sum(r["label"] == 0 for r in remainder),
                        "anomalous": sum(r["label"] == 1 for r in remainder),
                        "count": len(remainder),
                        "sha256": remainder_hash,
                    },
                    "target_test": {
                        "normal": sum(r["label"] == 0 for r in target_records),
                        "anomalous": sum(r["label"] == 1 for r in target_records),
                        "count": len(target_records),
                        "sha256": target_hash,
                    },
                    "source_validation": {"executed": False},
                    "paths": {
                        "source_pool": "source_pool.jsonl",
                        "source_support": "source_support.jsonl",
                        "unused_pool_samples": "unused_pool_samples.jsonl",
                        "unused_source_remainder": "unused_source_remainder.jsonl",
                        "target_test": "target_test.jsonl",
                    },
                }
                manifest_path = split_dir / "split_manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                summary["splits"].append(str(split_dir))
        summary["directions"].append(direction)
    summary_path = output_root / "split_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.dry_run:
        seeds = [x for x in args.seeds.split(",") if x.strip()]
        shots = [x for x in args.shots.split(",") if x.strip()]
        directions = list(direction_pairs(args.directions))
        print(f"splits={len(seeds) * len(shots) * len(directions)}")
        print(f"seeds={','.join(seeds)} shots={','.join(shots)} directions={','.join(k for k, _ in directions)}")
        return
    summary = build(args)
    print(f"built_splits={len(summary['splits'])}")


if __name__ == "__main__":
    main()
