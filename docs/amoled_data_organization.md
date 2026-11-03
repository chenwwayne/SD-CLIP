# AMOLED Data Organization (public format, private data)

## Data access policy

The AMOLED-ARRAY dataset used in the paper is **private** and is **not
redistributed** in this repository. This document openly describes its
*organizational format* so that:

1. the split builder can be audited and re-run on the private data;
2. anyone can generate synthetic placeholder data and run the whole pipeline
   end-to-end (smoke test).

## Directory layout

Both domains (`Active`, `Cell`) follow the same flat layout:

```
<amoled_root>/
  Active/
    images/normal/            # 181 normal images
    images/anomaly/           # 266 anomalous images
    mask/anomaly/             # 266 binary masks (same basenames as the images)
  Cell/
    images/normal/            # 138 normal images
    images/anomaly/           # 32 anomalous images
    mask/anomaly/             # 32 binary masks (same basenames as the images)
```

### File naming

- Normal images: `<TAG>_<id>.png`, e.g. `TDAOI_0001.png`.
- Anomaly images: `<TAG>_<id>.png`.
- Masks: exactly the same basename as their anomaly image (`.png`).
- `TAG` is `TDAOI` for `Active` and `TDAOH` for `Cell`. The split builder
  derives the record-level `class_name` from this tag via
  `scripts/build_amoled_directional_splits.py::class_name_for`.

Masks are binary: white defect pixels, black background.

## JSONL metadata schema

Every split file is a JSON Lines file. Each line is one image record:

```json
{
  "image_path": "images/anomaly/TDAOI_0123.png",
  "mask_path": "mask/anomaly/TDAOI_0123.png",
  "class_name": "TDAOI",
  "label": 1
}
```

| field        | type   | description                                              |
|--------------|--------|----------------------------------------------------------|
| `image_path` | string | path relative to the domain root (`<amoled_root>/<domain>`) |
| `mask_path`  | string | empty (`""`) for normal images; relative mask path for anomalies |
| `class_name` | string | `"TDAOI"` (Active) or `"TDAOH"` (Cell)                   |
| `label`      | int    | `0` = normal, `1` = anomalous                             |

Training jobs consume `source_support.jsonl`; test jobs consume
`target_test.jsonl`. The two files are intentionally kept separate.

## Split protocol (AMOLED v2)

- Two directions: `Active → Cell` and `Cell → Active`.
- Seeds: `0, 1, 9`.
- Shots: `1, 2, 4, 8, 16, 32`.
- For every `(direction, seed, shot)`:
  - **source pool**: 32 fixed images per seed = 16 normal + 16 anomalous,
    drawn via two deterministic, seed-keyed shuffles (see `select_pool`).
  - **source support**: strictly nested, `ceil(k/2)` normal and `floor(k/2)`
    anomalous for shot `k` (so a k-shot support is always a subset of the
    `(k-1)`-shot... i.e. every pool is a superset of the previous support).
  - **unused**: pool images not selected into the support are logged as
    `unused_pool_samples.jsonl`; the rest of the source domain is logged as
    `unused_source_remainder.jsonl`.
  - **target test**: the *full* target domain (never changes across
    seed/shot and never overlaps the source support).
  - **no source validation** is executed in v2.

### Per-split output files

```
splits/<direction>/seed_<s>/shot_<k>/
  source_pool.jsonl              # 32 images (16+16)
  source_support.jsonl           # ceil(k/2) normal + floor(k/2) anomalous
  unused_pool_samples.jsonl      # pool - support
  unused_source_remainder.jsonl  # source domain  - pool
  target_test.jsonl              # full target domain
  split_manifest.json            # counts + sha256 of every file above
```

Each JSONL file's `sha256` is recorded in `split_manifest.json`, making every
split reproducible and auditable.

## Smoke test with synthetic data

```bash
# 1. generate placeholder images/masks (no real AMOLED data involved)
python scripts/generate_amoled_dummy_data.py --output-dir dummy_data/amoled_array

# 2. build all splits (dry-run prints the job count first)
python scripts/build_amoled_directional_splits.py \
    --data-root dummy_data/amoled_array --output-root splits/amoled_dummy --dry-run
python scripts/build_amoled_directional_splits.py \
    --data-root dummy_data/amoled_array --output-root splits/amoled_dummy
```

The dummy generator writes the same `TDAOI`/`TDAOH` naming, the same
`images/`, `mask/` split, and the same counts, so every stage of the pipeline
can be validated on a machine that never sees the private imagery.