"""Evaluate a checkpoint using the same prediction pipeline as inference."""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import load_config, normalize_config, resolve_data_path
from src.dataset import VesselDataset
from src.metrics import calculate_dice, calculate_iou, calculate_precision, calculate_recall
from src.models import build_model
from src.prediction import predictions_from_logits
from src.transforms import get_val_transforms
from src.visualize import save_overlay_image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Vessel-Seg-UNet Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--config", default=os.path.join(ROOT_DIR, "configs", "default.yaml"),
        help="Config that supplies evaluation dataset paths",
    )
    parser.add_argument("--visualize", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--postprocess", action="store_true", help="Evaluate deployed postprocessed masks")
    parser.add_argument("--threshold", type=float, default=None, help="Override sigmoid threshold")
    parser.add_argument("--device", default=None, help="Device to evaluate on (auto by default; e.g. cuda, cuda:1, cpu)")
    parser.add_argument("--output-dir", default="results/eval", help="Directory for reports and images")
    args = parser.parse_args()

    data_config = load_config(args.config)
    try:
        device = (
            torch.device(args.device)
            if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except RuntimeError as exc:
        parser.error(f"Invalid --device value: {exc}")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    saved_config = checkpoint.get("config")
    runtime_config = normalize_config(saved_config) if isinstance(saved_config, dict) else data_config
    model_cfg = checkpoint_model_config(checkpoint, runtime_config)

    model = build_model(
        model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
    ).to(device)
    load_model_state(model, checkpoint)
    model.eval()

    img_size = runtime_config["inference"]["img_size"] or runtime_config["dataset"]["img_size"]
    dataset_cfg = data_config["dataset"]
    val_dataset = VesselDataset(
        image_dir=str(resolve_data_path(dataset_cfg["val_image_dir"], ROOT_DIR)),
        mask_dir=str(resolve_data_path(dataset_cfg["val_mask_dir"], ROOT_DIR)),
        transform=get_val_transforms(img_size, runtime_config["dataset"]["keep_aspect_ratio"]),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=dataset_cfg["num_workers"])

    threshold = args.threshold if args.threshold is not None else runtime_config["evaluation"]["threshold"]
    apply_postprocess = args.postprocess or runtime_config["evaluation"]["apply_postprocess"]
    logger.info("Device: %s | Samples: %s | Postprocess: %s", device, len(val_dataset), apply_postprocess)

    metric_values = {"dice": [], "iou": [], "precision": [], "recall": []}
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for index, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            predictions = predictions_from_logits(
                model(images),
                threshold=threshold,
                apply_postprocess=apply_postprocess,
                postprocess_config=runtime_config["inference"]["postprocess"],
            )
            metric_values["dice"].append(calculate_dice(predictions, masks))
            metric_values["iou"].append(calculate_iou(predictions, masks))
            metric_values["precision"].append(calculate_precision(predictions, masks))
            metric_values["recall"].append(calculate_recall(predictions, masks))

            if args.visualize:
                save_overlay_image(
                    (images[0, 0].cpu().numpy() * 255).astype(np.uint8),
                    (masks[0, 0].cpu().numpy() * 255).astype(np.uint8),
                    (predictions[0, 0].cpu().numpy() * 255).astype(np.uint8),
                    os.path.join(
                        args.output_dir,
                        f"overlay_{Path(val_dataset.filenames[index]).stem}.png",
                    ),
                )

    report = {
        name: (float(np.mean(values)), float(np.std(values)))
        for name, values in metric_values.items()
    }
    report_path = os.path.join(args.output_dir, "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Checkpoint: {args.checkpoint}\n")
        file.write(f"Samples: {len(val_dataset)}\n")
        file.write(f"Threshold: {threshold}\n")
        file.write(f"Postprocess: {apply_postprocess}\n\n")
        for name, (mean, std) in report.items():
            file.write(f"{name.title()}: {mean:.4f} ± {std:.4f}\n")

    for name, (mean, std) in report.items():
        logger.info("%s: %.4f ± %.4f", name.title(), mean, std)
    logger.info("Report saved to: %s", report_path)


if __name__ == "__main__":
    main()
