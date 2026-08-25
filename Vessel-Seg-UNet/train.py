"""Command-line training entry point for Vessel-Seg-UNet."""

import argparse
import logging
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.config import load_config, resolve_checkpoint_dir
from src.dataset import get_dataloaders
from src.models import build_model
from src.trainer import Trainer
from src.training import build_criterion, build_optimizer, build_scheduler, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Vessel-Seg-UNet Training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "configs", "default.yaml"),
        help="Path to a YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (auto by default; e.g. cuda, cuda:1, cpu)",
    )
    args = parser.parse_args()
    if args.device:
        try:
            requested_device = torch.device(args.device)
        except RuntimeError as exc:
            parser.error(f"Invalid --device value: {exc}")
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is not available")
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))

    logger.info("Vessel-Seg-UNet Training")
    logger.info("Config: %s", args.config)
    train_loader, val_loader = get_dataloaders(config, project_root=ROOT_DIR)
    logger.info("Train: %s images | Val: %s images", len(train_loader.dataset), len(val_loader.dataset))

    model_cfg = config["model"]
    model = build_model(
        model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        phase_classes=model_cfg.get("phase_classes", 0),
    )
    logger.info(
        "Model: %s | Trainable parameters: %s",
        model_cfg["name"],
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
    )

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        checkpoint_dir=resolve_checkpoint_dir(config, ROOT_DIR),
        device=args.device,
    )
    result = trainer.run()
    logger.info("Training complete. Best Dice: %.4f", result["best_dice"])


if __name__ == "__main__":
    main()
