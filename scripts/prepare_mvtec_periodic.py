#!/usr/bin/env python3
"""Prepare the MVTec-Periodic benchmark from the official MVTec-AD dataset
(carpet and grid classes only).

The paper evaluates cross-domain few-shot anomaly detection on two periodic
texture classes. This script converts the official MVTec-AD layout into the
flat layout consumed by this repository:

    data/MVTec-Periodic/carpet/
        images/normal/000.png ... 279.png      <- train/good (280)
        images/normal/t000.png ... t027.png    <- test/good  (28)
        images/anomaly/color_*.png ...         <- test/{color,cut,hole,metal_contamination,thread} (89)
        mask/anomaly/color_*.png ...           <- ground_truth/{...} (89)
        license.txt / readme.txt
    data/MVTec-Periodic/grid/
        similar flat layout (264 + 21 normal, 57 anomalous)

Anomaly images (official: bare ``<id>.png``) and their masks (official:
``<id>_mask.png``) are copied flat as ``<class>_<id>.png`` so the layout stays
collision-free. Normal ``test/good`` images get a ``t`` prefix to avoid
clashing with ``train/good`` images.

Usage:
    python scripts/prepare_mvtec_periodic.py --source-dir <mvtec_root>
    python scripts/prepare_mvtec_periodic.py --download   # needs pip pkg

Source: MVTec-AD (carpet, grid), Copyright MVTec Software GmbH,
licensed under CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CLASS_SPECS = {
    "carpet": (
        ("color", 19), ("cut", 17), ("hole", 17),
        ("metal_contamination", 17), ("thread", 19),
    ),
    "grid": (
        ("bent", 12), ("broken", 12), ("glue", 11),
        ("metal_contamination", 11), ("thread", 11),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir", type=Path,
        help="Root of the unzipped official MVTec-AD dataset (contains carpet/ and grid/).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/MVTec-Periodic"),
        help="Where the flat MVTec-Periodic layout is written (default: <repo>/data/MVTec-Periodic).",
    )
    p.add_argument("--classes", default="carpet,grid")
    p.add_argument(
        "--download", action="store_true",
        help="Download the two MVTec-AD classes with mvtec_anomaly_detection_downloader.",
    )
    return p.parse_args()


def download_class(download_root: Path, class_name: str) -> None:
    try:
        from mvtec_anomaly_detection_downloader import MVTecDownloader
    except ImportError:
        sys.exit(
            "mvtec_anomaly_detection_downloader is required for --download:\n"
            "    pip install mvtec_anomaly_detection_downloader\n"
            "Alternatively run offline with an existing official MVTec-AD root."
        )
    MVTecDownloader(dataset_name=class_name, root_dir=str(download_root)).download()


def build_class(source_cls_dir: Path, out_cls_dir: Path) -> None:
    out_cls_dir.mkdir(parents=True, exist_ok=True)
    normal_dir = out_cls_dir / "images" / "normal"
    anomaly_dir = out_cls_dir / "images" / "anomaly"
    mask_dir = out_cls_dir / "mask" / "anomaly"
    for d in (normal_dir, anomaly_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_good = source_cls_dir / "train" / "good"
    test_good = source_cls_dir / "test" / "good"
    if not train_good.is_dir() or not test_good.is_dir():
        raise FileNotFoundError(f"expected train/good and test/good under {source_cls_dir}")

    n_train = n_test = 0
    for img in sorted(train_good.glob("*.png")):
        shutil.copy2(img, normal_dir / img.name)
        n_train += 1
    for img in sorted(test_good.glob("*.png")):
        shutil.copy2(img, normal_dir / f"t{img.name}")
        n_test += 1

    test_dir = source_cls_dir / "test"
    gt_dir = source_cls_dir / "ground_truth"
    n_anom = 0
    for cls_name, _expected in CLASS_SPECS[source_cls_dir.name]:
        for img in sorted((test_dir / cls_name).glob("*.png")):
            # Official anomaly images are bare "<id>.png"; masks are
            # "<id>_mask.png". The benchmark uses "<class>_<id>.png" for both,
            # which keeps the flat layout collision-free.
            flat_name = f"{cls_name}_{img.name}"
            shutil.copy2(img, anomaly_dir / flat_name)
            mask_src = gt_dir / cls_name / f"{img.stem}_mask.png"
            if not mask_src.is_file():
                raise FileNotFoundError(f"missing ground-truth mask for {img}")
            shutil.copy2(mask_src, mask_dir / flat_name)
            n_anom += 1

    for meta in ("license.txt", "readme.txt"):
        src = source_cls_dir / meta
        if src.is_file():
            shutil.copy2(src, out_cls_dir / meta)

    print(
        f"[{source_cls_dir.name}] normal={n_train + n_test} "
        f"(train_good={n_train}, test_good={n_test}) anomaly={n_anom}"
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    if args.download:
        source_dir = output_dir.parent / "downloads"
        source_dir.mkdir(parents=True, exist_ok=True)
        for cls_name in args.classes.split(","):
            download_class(source_dir, cls_name)
    elif args.source_dir is None:
        sys.exit("either --source-dir or --download is required")
    else:
        source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        sys.exit(f"source dir does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in args.classes.split(","):
        source_cls_dir = source_dir / cls_name
        if not source_cls_dir.is_dir():
            print(f"[{cls_name}] not found at {source_cls_dir}; skipping.", file=sys.stderr)
            continue
        build_class(source_cls_dir, output_dir / cls_name)

    print(f"\nDone. MVTec-Periodic is ready at {output_dir}")
    print("Source: MVTec-AD (carpet, grid), Copyright MVTec Software GmbH, "
          "licensed under CC BY-NC-SA 4.0.")


if __name__ == "__main__":
    main()