"""Command-line training entry point for Vessel-Seg-UNet."""

import argparse
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.config import load_config, resolve_checkpoint_dir
from src.dataset import get_dataloaders
from src.models import build_model
from src.trainer import Trainer
from src.training import build_criterion, build_optimizer, build_scheduler


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
    args = parser.parse_args()
    config = load_config(args.config)

    logger.info("Vessel-Seg-UNet Training")
    logger.info("Config: %s", args.config)
    train_loader, val_loader = get_dataloaders(config, project_root=ROOT_DIR)
    logger.info("Train: %s images | Val: %s images", len(train_loader.dataset), len(val_loader.dataset))

    model_cfg = config["model"]
    model = build_model(
        model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
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
    )
    result = trainer.run()
    logger.info("Training complete. Best Dice: %.4f", result["best_dice"])


if __name__ == "__main__":
    main()
