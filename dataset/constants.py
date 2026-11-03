import os
import re
from pathlib import Path

# Repository root is the parent of the `dataset/` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Only the two periodic subsets used in the paper are shipped as open data
# recipes. Their layout is prepared by `scripts/prepare_mvtec_periodic.py`
# (downloadable public MVTec-AD data), see:
#   data/MVTec-Periodic/carpet/{images/{normal,anomaly}, mask/anomaly}
#   data/MVTec-Periodic/grid/{images/{normal,anomaly}, mask/anomaly}
DATA_PATH = {
    "MVTec-Periodic-carpet": str(REPO_ROOT / "data" / "MVTec-Periodic" / "carpet"),
    "MVTec-Periodic-grid": str(REPO_ROOT / "data" / "MVTec-Periodic" / "grid"),
}

CLASS_NAMES = {
    "MVTec-Periodic-carpet": ["carpet"],
    "MVTec-Periodic-grid": ["grid"],
}

DOMAINS = {
    "MVTec-Periodic-carpet": "Industrial",
    "MVTec-Periodic-grid": "Industrial",
}

REAL_NAMES = {
    "MVTec-Periodic-carpet": {"carpet": "carpet surface"},
    "MVTec-Periodic-grid": {"grid": "grid surface"},
}


def _discover_classes(dataset_name: str):
    """Auto-discover class labels for a dataset whose root contains class
    sub-folders (or an `images/{normal,anomaly}/<class>*` layout). This is
    optional; users may skip it by pre-populating CLASS_NAMES / REAL_NAMES.
    """
    root = DATA_PATH.get(dataset_name, "")
    if not root or not os.path.isdir(root):
        return
    if os.path.isdir(os.path.join(root, "images")):
        candidates = CLASS_NAMES.get(dataset_name, [])
        regex = None
        if candidates:
            regex = re.compile(r"(" + "|".join(re.escape(c) for c in candidates) + r")")
        classes = set()
        for status in ("normal", "anomaly"):
            folder = os.path.join(root, "images", status)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if regex is None:
                    continue
                match = regex.search(name)
                if match:
                    classes.add(match.group(1))
        classes = sorted(classes)
    else:
        classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
    if classes:
        CLASS_NAMES[dataset_name] = classes
        REAL_NAMES.setdefault(dataset_name, {})
        for class_name in classes:
            REAL_NAMES[dataset_name].setdefault(
                class_name, f"{dataset_name.lower()} panel"
            )


# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

# Generic semantic prompts (AA-CLIP text-style route, baseline).
PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}

# MVFA provides the generic visual-fault semantics used for source adaptation.
# APSF adds a separate process-specific periodicity-aware branch at inference.
MVFA_PROMPTS = {
    "prompt_normal": [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ],
    "prompt_abnormal": [
        "damaged {}",
        "broken {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
    ],
    "prompt_templates": [
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    ],
}

# Periodicity-aware prompts describe local periodic-process evidence rather than
# display-domain names, so they transfer across periodic texture domains
# (e.g. MVTec carpet/grid and other periodic patterns).
PERIODICITY_AWARE_PROMPTS = {
    "prompt_normal": [
        "a clean periodic pattern with continuous local structure",
        "a uniform periodic texture with consistent spacing",
        "a regular repeated pattern without local interruption",
        "a periodic visual structure with stable local repetition",
        "a clean repetitive texture with preserved local regularity",
        "a repeated pattern with no contaminant mark",
    ],
    "prompt_abnormal": [
        "a periodic pattern with a local structural disruption",
        "a repeated pattern with a local periodic interruption",
        "a repeated pattern with a locally missing element",
        "a repeated pattern with a locally shifted element",
        "a repeated pattern with uneven local spacing",
        "a repeated pattern with broken local continuity",
        "a repeated pattern with a distorted local unit",
        "a regular local structure with broken periodic continuity",
        "a periodic pattern with a local contamination mark",
        "a periodic texture with a small stain disrupting local regularity",
        "a repeated pattern with a residue-like spot",
        "a periodic visual structure with a foreign spot",
        "a clean repeated texture interrupted by local contamination",
        "a repeated texture with a local stain-like region",
        "a periodic texture with residue breaking local continuity",
        "a repeated pattern with a small dirty spot",
        "a periodic structure partially covered by contamination",
        "a repetitive texture with an isolated intensity anomaly",
        "a periodic pattern with local contrast inconsistency",
        "a repeated visual structure with a small irregular blob",
        "a periodic texture crossed by a thin contaminant trace",
        "a regular periodic background with a localized surface stain",
    ],
    "prompt_templates": [
        "{}.",
        "a close-up image of {}.",
        "a cropped image of {}.",
        "a high resolution inspection image of {}.",
    ],
}

# Descriptive prompts for fully-periodic array-style images. Kept for
# completeness; the paper's periodic experiments use PERIODICITY_AWARE_PROMPTS.
PERIODIC_ARRAY_PROMPTS = {
    "prompt_normal": [
        "a normal periodic array image",
        "a defect-free regular repeated cell pattern",
        "a normal industrial image with consistent row and column structures",
        "a clean image with uniform periodic texture",
    ],
    "prompt_abnormal": [
        "an anomalous periodic array image",
        "a defective image with broken periodic structure",
        "an image with local structural inconsistency",
        "a periodic texture image with defects",
        "an image with disrupted row or column pattern",
    ],
    "prompt_templates": ["{}"],
}

# Dataset-specific prompt overrides. The experiments in the paper run the
# periodic benchmark with either PROMPTS (AA-CLIP baseline) or
# MVFA_PROMPTS / PERIODICITY_AWARE_PROMPTS, so no override is needed here.
DATASET_PROMPTS = {}