#!/usr/bin/env python3
"""Run the MVTec-Periodic five-level ablation under the registered protocol.

This runner deliberately keeps source-support and target-test metadata separate.
It does not run source validation or request visualizations.
"""

import argparse
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS = {
    "carpet": "MVTec-Periodic/carpet",
    "grid": "MVTec-Periodic/grid",
}
METHODS = {
    "c0_aa_clip": ("aa", "simple", "simple"),
    "c1_generic_prompts": ("mvfa", "simple", "simple"),
    "c2_apsf": ("apsf", "simple", "simple"),
    "c3_apsf_kan_no_memory": ("apsf", "kan_residual", "simple"),
    "c4_sd_clip_snprb": ("apsf", "kan_residual", "simple"),
}
DIRECTIONS = (("carpet", "grid"), ("grid", "carpet"))
SEEDS = (0, 1, 9)
SHOTS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Item:
    image_path: str
    label: int
    mask_path: str
    class_name: str


def stable_seed(seed, *parts):
    raw = "|".join(map(str, (seed, *parts))).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def image_files(root, status):
    return sorted(p.relative_to(root).as_posix() for p in (root / "images" / status).glob("*.png"))


def make_items(root, name, paths):
    out = []
    for rel in paths:
        label = 0 if "/normal/" in rel else 1
        mask = ""
        if label:
            mask = (root / "mask" / "anomaly" / Path(rel).name).relative_to(root).as_posix()
        out.append(Item(rel, label, mask, name))
    return out


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(x), ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


def prepare_split(data_root, split_root, seed, source, target, shot):
    source_root = data_root / DATASETS[source]
    target_root = data_root / DATASETS[target]
    normal = image_files(source_root, "normal")
    anomaly = image_files(source_root, "anomaly")
    rng_n = random.Random(stable_seed(seed, source, "normal")); rng_a = random.Random(stable_seed(seed, source, "anomaly"))
    rng_n.shuffle(normal); rng_a.shuffle(anomaly)
    if len(normal) < 16 or len(anomaly) < 16:
        raise ValueError(f"{source} lacks 16 normal and 16 anomaly source-pool images")
    pool_n, pool_a = normal[:16], anomaly[:16]
    n_n, n_a = (shot + 1) // 2, shot // 2
    support_n = pool_n[:n_n]; support_a = pool_a[:n_a]
    support = make_items(source_root, source, support_n + support_a)
    target_test = make_items(target_root, target, image_files(target_root, "normal") + image_files(target_root, "anomaly"))
    base = split_root / f"seed{seed}" / f"{source}_to_{target}_shot{shot}"
    write_jsonl(base / "source_support.jsonl", support)
    write_jsonl(base / "target_test.jsonl", target_test)
    (base / "split_summary.json").write_text(json.dumps({
        "seed": seed, "shot": shot, "source": source, "target": target,
        "source_pool": {"normal": 16, "anomaly": 16},
        "support": {"normal": n_n, "anomaly": n_a},
        "target_test": len(target_test), "protocol": "MVTec-Periodic full-target few-shot v2",
    }, indent=2) + "\n")
    return base


def command(root, dataset, shot, seed, metadata, save_path, method, metrics, image_epoch, test_batch, apsf_weight=0.5, memory_weight=0.0, support_metadata=None, support_dataset=None):
    prompt, adapter, projection = METHODS[method]
    common = ["--text_adapt_weight", "0.1", "--image_adapt_weight", "0.1", "--image_adapter_type", adapter, "--image_proj_type", projection,
              "--kan_bottleneck", "16", "--kan_adapter_bottleneck", "16", "--kan_proj_bottleneck", "16", "--kan_grid_size", "2", "--kan_spline_order", "2"]
    return [sys.executable, str(root / ("train.py" if metrics is None else "test.py")), "--dataset", dataset, "--shot", str(shot), "--seed", str(seed), "--metadata-path", str(metadata), "--save_path", str(save_path), "--prompt_mode", prompt, *common] + ([] if metrics is None else ["--batch_size", str(test_batch), "--checkpoint_epoch", str(image_epoch), "--apsf_weight", str(apsf_weight), "--memory-weight", str(memory_weight), "--metrics_json_path", str(metrics)] + (["--support-metadata-path", str(support_metadata), "--support-dataset", str(support_dataset)] if support_metadata else []))


def run_job(args, root, gpu, seed, shot, source, target):
    split = prepare_split(args.data_root, args.output_root / "splits", seed, source, target, shot)
    source_dataset = f"MVTec-Periodic-{source}"
    target_dataset = f"MVTec-Periodic-{target}"
    base = args.output_root / "shared" / f"seed{seed}" / f"{source}_to_{target}_shot{shot}"
    base.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu)})
    aa_ckpt, mvfa_ckpt, sd_ckpt = base / "aa_clip", base / "generic", base / "kan_residual"
    jobs = [("c0_aa_clip", aa_ckpt, "c0_aa_clip"), ("c1_generic_prompts", mvfa_ckpt, "c1_generic_prompts"), ("c3_apsf_kan_no_memory", sd_ckpt, "c3_apsf_kan_no_memory")]
    for label, ckpt, method in jobs:
        ckpt.mkdir(exist_ok=True)
        if not (ckpt / f"image_adapter_{args.epoch}.pth").exists():
            log = base / f"train_{label}_gpu{gpu}.log"
            train = command(root, source_dataset, shot, seed, split / "source_support.jsonl", ckpt, method, None, args.epoch, args.image_batch)
            with log.open("w", encoding="utf-8") as fh:
                subprocess.run(train, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    lambda_dir = base / "lambda"
    lambda_file = lambda_dir / "learned_weight.json"
    if not lambda_file.exists():
        lambda_dir.mkdir(exist_ok=True)
        learn = [sys.executable, str(root / "tools" / "learn_apsf_weight.py"), "--source", source_dataset, "--shot", str(shot), "--seed", str(seed), "--checkpoint-dir", str(mvfa_ckpt), "--output-dir", str(lambda_dir), "--metadata-path", str(split / "source_support.jsonl"), "--epochs", "5", "--batch-size", "64"]
        with (base / f"learn_lambda_gpu{gpu}.log").open("w", encoding="utf-8") as fh:
            subprocess.run(learn, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    weight = float(json.loads(lambda_file.read_text())["final_weight"])
    checkpoints = {"c0_aa_clip": aa_ckpt, "c1_generic_prompts": mvfa_ckpt, "c2_apsf": mvfa_ckpt, "c3_apsf_kan_no_memory": sd_ckpt, "c4_sd_clip_snprb": sd_ckpt}
    test_settings = {"c0_aa_clip": ("aa", 0.0), "c1_generic_prompts": ("mvfa", 0.0), "c2_apsf": ("apsf", 0.0), "c3_apsf_kan_no_memory": ("apsf", 0.0), "c4_sd_clip_snprb": ("apsf", 0.75)}
    for method in METHODS:
        run = args.output_root / method / f"seed{seed}" / f"{source}_to_{target}_shot{shot}"; metrics = run / "metrics.json"
        if metrics.is_file() and not args.force: continue
        run.mkdir(parents=True, exist_ok=True)
        prompt, memory_weight = test_settings[method]
        test = command(root, target_dataset, shot, seed, split / "target_test.jsonl", checkpoints[method], method, metrics, args.epoch, args.test_batch, weight, memory_weight, split / "source_support.jsonl" if memory_weight else None, source_dataset if memory_weight else None)
        with (run / f"run_gpu{gpu}.log").open("w", encoding="utf-8") as fh:
            subprocess.run(test, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, default=Path("data/MVTec-Periodic")); p.add_argument("--output-root", type=Path, default=Path("results/mvtec_periodic_ablation_seed0_seed1_seed9_epoch5_full")); p.add_argument("--seeds", default="0,1,9"); p.add_argument("--shots", default="1,2,4,8,16,32"); p.add_argument("--gpus", default="0,1,2"); p.add_argument("--slots-per-gpu", type=int, default=1); p.add_argument("--epoch", type=int, default=5); p.add_argument("--image-batch", type=int, default=8); p.add_argument("--test-batch", type=int, default=128); p.add_argument("--dry-run", action="store_true"); p.add_argument("--smoke", action="store_true"); p.add_argument("--force", action="store_true")
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]; a.output_root = (root / a.output_root).resolve(); seeds = [0] if a.smoke else [int(x) for x in a.seeds.split(",")]; shots = [2] if a.smoke else [int(x) for x in a.shots.split(",")]; gpus = [int(x) for x in a.gpus.split(",")]
    manifest = {"methods": list(METHODS), "seeds": seeds, "shots": shots, "directions": DIRECTIONS, "gpus": gpus, "total_jobs": len(METHODS) * len(seeds) * len(shots) * len(DIRECTIONS), "visualize": False, "source_validation": False}
    if a.dry_run: print(json.dumps(manifest, indent=2)); return
    a.output_root.mkdir(parents=True, exist_ok=True); (a.output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    q = queue.Queue(); [q.put((s, k, x, y)) for s in seeds for k in shots for x, y in DIRECTIONS]
    errors = []; lock = threading.Lock()
    def worker(gpu):
        while True:
            try: job = q.get_nowait()
            except queue.Empty: return
            try: run_job(a, root, gpu, *job)
            except Exception as exc:
                with lock: errors.append((job, repr(exc)))
            finally: q.task_done()
    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads: t.start()
    for t in threads: t.join()
    if errors: raise SystemExit(json.dumps(errors, indent=2))


if __name__ == "__main__": main()
