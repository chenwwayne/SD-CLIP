#!/usr/bin/env python3
"""Generate synthetic AMOLED-style data to demonstrate the private dataset's
*organizational format* without exposing any real images.

The AMOLED-ARRAY data itself is private and is NOT redistributed here.  This
generator produces placeholder images following the exact same layout so the
public split builder and pipeline can be exercised end-to-end (smoke test).

Generated layout (identical to the real dataset):

    <output>/Active/images/normal/TDAOI_0001.png ...          (181)
            /Active/images/anomaly/TDAOI_xxxx.png ...         (266)
            /Active/mask/anomaly/TDAOI_xxxx.png ...           (266, same stems)
            /Cell/images/normal/TDAOH_0001.png ...            (138)
            /Cell/images/anomaly/TDAOH_xxxx.png ...           (32)
            /Cell/mask/anomaly/TDAOH_xxxx.png ...             (32, same stems)

Filenames embed the class tag ``TDAOI`` (Active) / ``TDAOH`` (Cell); the split
builder derives ``class_name`` from this tag.  Masks share the exact basename
of their anomaly image so ``--data-root`` scanning finds them one-to-one.

After generation, feed the output directory to the split builder:

    python scripts/build_amoled_directional_splits.py \
        --data-root dummy_data/amoled_array --output-root splits/amoled_dummy

Requirements: numpy, Pillow (both installed by environment.yml).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

# (normal, anomalous) per domain, matching the private AMOLED-ARRAY counts.
DOMAIN_COUNTS = {
    "Active": (181, 266),
    "Cell": (138, 32),
}
CLASS_TAG = {"Active": "TDAOI", "Cell": "TDAOH"}
IMG_SIZE = 256


def make_background(rng: random.Random, anomaly: bool) -> np.ndarray:
    """Synthetic display-panel background: base color + low-frequency noise.

    Anomalous images get a bright, compact defect blob inscribed in a black
    mask (the mask is written separately, not drawn into the image).
    """
    base = np.full((IMG_SIZE, IMG_SIZE, 3), 90 + rng.randint(0, 30), dtype=np.float32)
    # smooth low-frequency variation on the 64x64 grid
    coarse = np.array(
        [[[rng.randint(-25, 25) for _ in range(3)] for _ in range(4)] for _ in range(4)],
        dtype=np.float32,
    )
    big = np.kron(coarse, np.ones((64, 64, 1), dtype=np.float32))
    img = np.clip(base + big, 0, 255)
    if anomaly:
        cx, cy = rng.randint(40, IMG_SIZE - 40), rng.randint(40, IMG_SIZE - 40)
        rx, ry = rng.randint(8, 22), rng.randint(8, 22)
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        blob = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        img[blob] = np.clip(img[blob] + 120 + rng.randint(0, 60), 0, 255)
    return img.astype(np.uint8)


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)


def save_mask(path: Path, anomaly: bool, rng: random.Random) -> None:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if anomaly:
        cx, cy = rng.randint(40, IMG_SIZE - 40), rng.randint(40, IMG_SIZE - 40)
        rx, ry = rng.randint(8, 22), rng.randint(8, 22)
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def build_domain(output: Path, domain: str, rng: random.Random) -> tuple[int, int]:
    normal_n, anomaly_n = DOMAIN_COUNTS[domain]
    tag = CLASS_TAG[domain]
    normal_dir = output / domain / "images" / "normal"
    anomaly_dir = output / domain / "images" / "anomaly"
    mask_dir = output / domain / "mask" / "anomaly"
    idx = 0
    for _ in range(normal_n):
        idx += 1
        save_image(normal_dir / f"{tag}_{idx:04d}.png", make_background(rng, anomaly=False))
    for _ in range(anomaly_n):
        idx += 1
        name = f"{tag}_{idx:04d}.png"
        save_image(anomaly_dir / name, make_background(rng, anomaly=True))
        save_mask(mask_dir / name, anomaly=True, rng=rng)
    return normal_n, anomaly_n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dummy_data/amoled_array"),
        help="Directory that will contain Active/ and Cell/ (default: <repo>/dummy_data/amoled_array).",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    total_normal = total_anomaly = 0
    for domain, (n_norm, n_anom) in DOMAIN_COUNTS.items():
        build_domain(args.output_dir, domain, rng)
        total_normal += n_norm
        total_anomaly += n_anom
        print(f"[{domain}] normal={n_norm} anomaly={n_anom}")
    print(f"\nDummy AMOLED layout ready at {args.output_dir} "
          f"(normal={total_normal}, anomalous={total_anomaly}).")
    print("Next step: build reproducible splits, e.g.")
    print(f"    python scripts/build_amoled_directional_splits.py \\")
    print(f"        --data-root {args.output_dir} \\")
    print("        --output-root splits/amoled_dummy --dry-run")


if __name__ == "__main__":
    main()