"""Single-image and batch inference backed by self-describing checkpoints."""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state
from src.config import normalize_config
from src.models import build_model
from src.prediction import predictions_from_logits
from src.transforms import get_val_transforms


class VesselSegmentor:
    """Load a trusted project checkpoint and produce postprocessed vessel masks."""

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        img_size: Optional[int] = None,
        threshold: Optional[float] = None,
        min_component_size: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = load_checkpoint(model_path, map_location=self.device)
        saved_config = self.checkpoint.get("config")
        self.config = normalize_config(saved_config if isinstance(saved_config, dict) else config)
        model_cfg = checkpoint_model_config(self.checkpoint, self.config)
        if model_name is not None:
            model_cfg["name"] = model_name
        self.model = build_model(
            model_cfg["name"],
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
        )
        load_model_state(self.model, self.checkpoint)
        self.model.to(self.device).eval()

        inference_cfg = self.config["inference"]
        self.img_size = img_size or inference_cfg["img_size"] or self.config["dataset"]["img_size"]
        if not isinstance(self.img_size, int) or self.img_size <= 0:
            raise ValueError("img_size must be a positive integer")
        self.keep_aspect_ratio = self.config["dataset"]["keep_aspect_ratio"]
        self.threshold = inference_cfg["threshold"] if threshold is None else threshold
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.postprocess_config = dict(inference_cfg["postprocess"])
        if min_component_size is not None:
            if min_component_size <= 0:
                raise ValueError("min_component_size must be positive")
            self.postprocess_config["min_component_size"] = min_component_size
        self.transform = get_val_transforms(self.img_size, self.keep_aspect_ratio)

    @torch.no_grad()
    def predict(self, image_path: str) -> np.ndarray:
        """Read a grayscale image and return an original-size uint8 mask {0, 255}."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")
        return self.predict_array(image)

    @torch.no_grad()
    def predict_array(self, image: np.ndarray) -> np.ndarray:
        """Segment a ``(H, W)`` grayscale uint8 image."""
        if image.ndim != 2:
            raise ValueError("predict_array expects a two-dimensional grayscale image")
        original_h, original_w = image.shape
        tensor = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        predictions = predictions_from_logits(
            self.model(tensor),
            threshold=float(self.threshold),
            apply_postprocess=bool(self.postprocess_config["enabled"]),
            postprocess_config=self.postprocess_config,
        )
        mask = predictions[0, 0].cpu().numpy().astype(np.uint8) * 255
        mask = self._restore_original_geometry(mask, original_h, original_w)
        return mask

    def _restore_original_geometry(self, mask: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
        """Remove validation/inference padding before resizing the prediction back."""
        if self.keep_aspect_ratio:
            scale = self.img_size / max(original_h, original_w)
            resized_h = max(1, int(round(original_h * scale)))
            resized_w = max(1, int(round(original_w * scale)))
            top = (self.img_size - resized_h) // 2
            left = (self.img_size - resized_w) // 2
            mask = mask[top:top + resized_h, left:left + resized_w]
        if mask.shape != (original_h, original_w):
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return mask

    @torch.no_grad()
    def predict_batch(self, image_paths: list[str]) -> list[np.ndarray]:
        """Segment a list of files while retaining each image's original geometry."""
        return [self.predict(path) for path in image_paths]


def main():
    parser = argparse.ArgumentParser(description="Vessel Segmentation Inference")
    parser.add_argument("--model", required=True, help="Path to a project checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output", default="results/inference", help="Output directory")
    parser.add_argument("--config", default=None, help="Needed only to interpret legacy raw state_dict checkpoints")
    parser.add_argument("--model-name", default=None, help="Override architecture for a legacy checkpoint")
    parser.add_argument("--img-size", type=int, default=None, help="Override preprocessing size")
    parser.add_argument("--threshold", type=float, default=None, help="Override binarization threshold")
    parser.add_argument("--min-size", type=int, default=None, help="Override min component size")
    args = parser.parse_args()

    legacy_config = None
    if args.config:
        from src.config import load_config
        legacy_config = load_config(args.config)
    segmentor = VesselSegmentor(
        model_path=args.model,
        model_name=args.model_name,
        img_size=args.img_size,
        threshold=args.threshold,
        min_component_size=args.min_size,
        config=legacy_config,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            path for path in input_path.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Processing {len(image_paths)} images...")
    for image_path in image_paths:
        out_path = output_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(out_path), segmentor.predict(str(image_path))):
            raise IOError(f"Failed to write output: {out_path}")
        print(f"  {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
