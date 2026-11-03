import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def build_normal_patch_memory(model, dataset_dict, device, batch_size):
    """Build a normal-only patch memory bank from job-local source support."""
    memory_features = None
    loader_kwargs = {"num_workers": 4, "pin_memory": True}
    for class_dataset in dataset_dict.values():
        loader = torch.utils.data.DataLoader(
            class_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )
        for input_data in loader:
            normal = input_data["label"] == 0
            if not bool(normal.any()):
                continue
            image = input_data["image"][normal].to(device)
            with torch.no_grad():
                patch_features, _ = model(image)
            if memory_features is None:
                memory_features = [[] for _ in patch_features]
            for layer, features in enumerate(patch_features):
                features = torch.nn.functional.normalize(features, dim=-1)
                memory_features[layer].append(features.reshape(-1, features.shape[-1]))
    if memory_features is None:
        return None
    return [torch.cat(layer_features, dim=0) for layer_features in memory_features]


def minmax_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = values.min(), values.max()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
    memory_features=None,
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    preds_memory = []
    preds_memory_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_feature = class_text_embeddings
        # forward image
        patch_features, det_feature = model(image)
        # calculate similarity and get prediction
        # cls_preds = []
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            # f: bs,patch_num,768
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1)
        patch_preds = patch_preds.cpu().numpy()
        preds.append(patch_preds)
        if memory_features is not None:
            memory_map = torch.zeros_like(torch.from_numpy(patch_preds)).to(device)
            for features, memory in zip(patch_features, memory_features):
                query = torch.nn.functional.normalize(features, dim=-1)
                similarity = torch.matmul(
                    torch.nn.functional.normalize(memory, dim=-1), query.transpose(1, 2)
                ).amax(dim=1)
                height = int(np.sqrt(similarity.shape[-1]))
                layer_map = (1.0 - similarity).reshape(
                    similarity.shape[0], 1, height, height
                )
                layer_map = torch.nn.functional.interpolate(
                    layer_map, size=img_size, mode="bilinear", align_corners=True
                )[:, 0]
                memory_map += layer_map
            memory_map = memory_map.cpu().numpy()
            preds_memory.append(memory_map)
            preds_memory_image.append(memory_map.reshape(memory_map.shape[0], -1).max(axis=1))
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, preds_memory, preds_memory_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="job-local target JSONL metadata; overrides the shared dataset metadata",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--metrics_json_path", type=str, default="")
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="evaluate only image_adapter_<epoch>.pth; default evaluates all checkpoints",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize_dir",
        type=str,
        default=None,
        help="visualization output root; defaults to save_path",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="test log path; defaults to save_path/test.log",
    )
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument(
        "--adapter_type",
        choices=["simple", "kan", "kan_residual", "mlp_residual"],
        default=None,
    )
    parser.add_argument(
        "--image_adapter_type",
        choices=["simple", "kan", "kan_residual", "mlp_residual"],
        default=None,
    )
    parser.add_argument(
        "--image_proj_type",
        choices=["simple", "kan", "kan_residual", "mlp_residual"],
        default=None,
    )
    parser.add_argument("--kan_bottleneck", type=int, default=None)
    parser.add_argument("--kan_adapter_bottleneck", type=int, default=None)
    parser.add_argument("--kan_proj_bottleneck", type=int, default=None)
    parser.add_argument("--kan_grid_size", type=int, default=None)
    parser.add_argument("--kan_spline_order", type=int, default=None)
    parser.add_argument(
        "--predictions_npz_path",
        type=str,
        default=None,
        help="optionally save masks, labels, pixel maps, and image scores",
    )
    parser.add_argument(
        "--support-metadata-path",
        type=str,
        default=None,
        help="job-local source-support JSONL for the optional normal patch memory bank",
    )
    parser.add_argument(
        "--support-dataset",
        type=str,
        default=None,
        help="source dataset name used to interpret source-support class names",
    )
    parser.add_argument(
        "--memory-weight",
        type=float,
        default=0.0,
        help="weight of the normal-support patch memory branch; 0 keeps the original path",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=os.environ.get("PROMPT_MODE", "mvfa"),
        choices=[
            "aa",
            "mvfa",
            "apsf",
        ],
    )
    parser.add_argument(
        "--apsf_weight",
        type=float,
        default=float(os.environ.get("APSF_WEIGHT", "0.5")),
    )

    args = parser.parse_args()
    if not 0.0 <= args.apsf_weight <= 1.0:
        parser.error("--apsf_weight must be in [0, 1]")
    if not 0.0 <= args.memory_weight <= 1.0:
        parser.error("--memory-weight must be in [0, 1]")
    if args.memory_weight > 0.0 and not args.support_metadata_path:
        parser.error("--support-metadata-path is required when --memory-weight > 0")
    os.environ["PROMPT_MODE"] = args.prompt_mode
    os.environ["APSF_WEIGHT"] = str(args.apsf_weight)
    if args.checkpoint_epoch is None:
        files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    else:
        files = glob(
            os.path.join(
                args.save_path, f"image_adapter_{args.checkpoint_epoch}.pth"
            )
        )
    assert len(files) > 0, "image adapter checkpoint not found"
    checkpoint_preview = torch.load(files[0], map_location="cpu")
    saved_config = checkpoint_preview.get("model_config", {})
    fallback_type = args.adapter_type or "simple"
    args.image_adapter_type = (
        args.image_adapter_type
        or saved_config.get("image_adapter_type")
        or fallback_type
    )
    args.image_proj_type = (
        args.image_proj_type or saved_config.get("image_proj_type") or fallback_type
    )
    for name, default in (
        ("kan_bottleneck", 16),
        ("kan_grid_size", 3),
        ("kan_spline_order", 3),
    ):
        if getattr(args, name) is None:
            setattr(args, name, saved_config.get(name, default))
    for name in ("kan_adapter_bottleneck", "kan_proj_bottleneck"):
        if getattr(args, name) is None:
            setattr(args, name, saved_config.get(name, args.kan_bottleneck))
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    if args.visualize_dir:
        os.makedirs(args.visualize_dir, exist_ok=True)
    log_path = args.log_path or os.path.join(args.save_path, "test.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=log_path,
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        image_adapter_type=args.image_adapter_type,
        image_proj_type=args.image_proj_type,
        kan_bottleneck=args.kan_bottleneck,
        kan_adapter_bottleneck=args.kan_adapter_bottleneck,
        kan_proj_bottleneck=args.kan_proj_bottleneck,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_seed=args.seed,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    for file in files:
        checkpoint = torch.load(file)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        # ========================================================
        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
            metadata_path=args.metadata_path,
        )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )
        memory_features = None
        if args.memory_weight > 0.0:
            support_datasets = get_dataset(
                args.support_dataset or args.dataset,
                args.img_size,
                None,
                args.shot,
                "test",
                logger=logger,
                metadata_path=args.support_metadata_path,
            )
            memory_features = build_normal_patch_memory(
                model, support_datasets, device, max(1, min(args.batch_size, 8))
            )
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "pixel F1 max",
                "image AUC",
                "image AP",
                "image F1 max",
            ]
        )
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                (
                    masks,
                    labels,
                    preds,
                    preds_image,
                    preds_memory,
                    preds_memory_image,
                    file_names,
                ) = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                    memory_features=memory_features,
                )
            if memory_features is not None:
                preds_memory = np.concatenate(preds_memory, axis=0)
                preds_memory_image = np.concatenate(preds_memory_image, axis=0)
                preds = (1.0 - args.memory_weight) * minmax_normalize(preds) + args.memory_weight * minmax_normalize(preds_memory)
                preds_image = (1.0 - args.memory_weight) * minmax_normalize(preds_image) + args.memory_weight * minmax_normalize(preds_memory_image)
            if args.predictions_npz_path:
                payload = dict(
                    masks=masks,
                    labels=labels,
                    preds=preds,
                    preds_image=preds_image,
                )
                np.savez_compressed(args.predictions_npz_path, **payload)
            # ========================================================
            if args.visualize:
                visualize(
                    masks,
                    preds,
                    file_names,
                    args.visualize_dir or args.save_path,
                    args.dataset,
                    class_name=class_name,
                )
            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)
        avg_row = {
            "class name": "Average",
            "pixel AUC": pd.to_numeric(df["pixel AUC"], errors="coerce").mean(),
            "pixel AP": pd.to_numeric(df["pixel AP"], errors="coerce").mean(),
            "pixel F1 max": pd.to_numeric(df["pixel F1 max"], errors="coerce").mean(),
            "image AUC": pd.to_numeric(df["image AUC"], errors="coerce").mean(),
            "image AP": pd.to_numeric(df["image AP"], errors="coerce").mean(),
            "image F1 max": pd.to_numeric(df["image F1 max"], errors="coerce").mean(),
        }
        df.loc[len(df)] = avg_row
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
        if args.metrics_json_path:
            metrics = {
                "checkpoint_epoch": int(test_epoch),
                "image_auroc": float(avg_row["image AUC"]) / 100.0,
                "image_ap": float(avg_row["image AP"]) / 100.0,
                "image_f1_max": float(avg_row["image F1 max"]) / 100.0,
                "pixel_auroc": float(avg_row["pixel AUC"]) / 100.0,
                "pixel_ap": float(avg_row["pixel AP"]) / 100.0,
                "pixel_f1_max": float(avg_row["pixel F1 max"]) / 100.0,
            }
            metrics_dir = os.path.dirname(args.metrics_json_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(args.metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
