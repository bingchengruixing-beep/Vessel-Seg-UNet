"""Single-image and batch inference backed by self-describing checkpoints."""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.checkpoints import (
    checkpoint_model_config,
    infer_legacy_in_channels,
    infer_legacy_model_name,
    load_checkpoint,
    load_model_state,
)
from src.config import normalize_config
from src.models import build_model_from_config
from src.prediction import main_logits_from_output, postprocess_predictions
from src.transforms import get_val_transforms


def restore_original_geometry(
    mask: np.ndarray,
    original_h: int,
    original_w: int,
    img_size: int,
    keep_aspect_ratio: bool,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """撤销预处理中的等比例缩放 + 居中补零（letterbox），还原到原图几何。

    与 ``A.LongestMaxSize`` + ``A.PadIfNeeded`` 的居中补边顺序互逆：
    先裁掉补边区域，再缩放回原始尺寸。供推理使用，也便于单元测试。
    """
    if keep_aspect_ratio:
        scale = img_size / max(original_h, original_w)
        resized_h = max(1, int(round(original_h * scale)))
        resized_w = max(1, int(round(original_w * scale)))
        top = (img_size - resized_h) // 2
        left = (img_size - resized_w) // 2
        mask = mask[top:top + resized_h, left:left + resized_w]
    if mask.shape != (original_h, original_w):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=interpolation)
    return mask


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
        # 先在 CPU 解析 checkpoint，避免集成多个模型时把优化器状态也长期占在显存中。
        checkpoint = load_checkpoint(model_path, map_location="cpu")
        saved_config = checkpoint.get("config")
        self.config = normalize_config(saved_config if isinstance(saved_config, dict) else config)
        if not isinstance(saved_config, dict):
            detected_name = infer_legacy_model_name(checkpoint)
            if detected_name:
                self.config["model"]["name"] = detected_name
                self.config["model"]["pretrained"] = False
            detected_channels = infer_legacy_in_channels(checkpoint)
            if detected_channels:
                self.config["model"]["in_channels"] = detected_channels
                self.config["model"]["input_mode"] = "grayscale"
        model_cfg = checkpoint_model_config(checkpoint, self.config)
        if model_name is not None:
            model_cfg["name"] = model_name
        self.in_channels = int(model_cfg.get("in_channels", 1))
        self.model = build_model_from_config(model_cfg)
        load_model_state(self.model, checkpoint)
        self.model.to(self.device).eval()
        del checkpoint

        inference_cfg = self.config["inference"]
        self.img_size = img_size or inference_cfg["img_size"] or self.config["dataset"]["img_size"]
        self.keep_aspect_ratio = self.config["dataset"]["keep_aspect_ratio"]
        self.threshold = inference_cfg["threshold"] if threshold is None else threshold
        self.postprocess_config = dict(inference_cfg["postprocess"])
        if min_component_size is not None:
            self.postprocess_config["min_component_size"] = min_component_size
        self.transform = get_val_transforms(self.img_size, self.keep_aspect_ratio)
        self.patch_config = dict(inference_cfg.get("patch", {}))

    @torch.no_grad()
    def predict(self, image_path: str) -> np.ndarray:
        """Read a grayscale image and return an original-size uint8 mask {0, 255}."""
        # 使用 np.fromfile + cv2.imdecode 支持 Windows 中文路径。
        image_buf = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")
        return self.predict_array(image)

    @torch.no_grad()
    def predict_array(self, image: np.ndarray) -> np.ndarray:
        """Segment a ``(H, W)`` grayscale uint8 image."""
        probabilities = self.predict_probability_array(image)
        predictions = torch.from_numpy((probabilities > float(self.threshold)).astype(np.float32))[None, None]
        if self.postprocess_config["enabled"]:
            predictions = postprocess_predictions(predictions, self.postprocess_config)
        return predictions[0, 0].numpy().astype(np.uint8) * 255

    @torch.no_grad()
    def predict_probability_array(self, image: np.ndarray) -> np.ndarray:
        """返回与原图同尺寸的概率图，支持整图和重叠 patch 两种模式。"""
        image, original_h, original_w = self._prepare_input(image)
        if self.patch_config.get("enabled", False):
            return self._predict_sliding(image)
        tensor = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(main_logits_from_output(self.model(tensor)))
        return self._restore_original_geometry(
            probabilities[0, 0].detach().cpu().numpy(),
            original_h,
            original_w,
            cv2.INTER_LINEAR,
        )

    def _prepare_input(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """按 checkpoint 通道数把单图或前/中/后三时相整理为模型输入。"""
        if image.ndim == 2:
            height, width = image.shape
            if self.in_channels == 1:
                return image, height, width
            if self.in_channels == 3:
                return np.repeat(image[..., None], 3, axis=2), height, width
        elif image.ndim == 3 and image.shape[2] == 3:
            height, width = image.shape[:2]
            if self.in_channels == 3:
                return image, height, width
            if self.in_channels == 1:
                return image[..., 1], height, width
        raise ValueError(
            f"无法把形状 {image.shape} 的输入适配为 {self.in_channels} 通道模型"
        )

    def _predict_sliding(self, image: np.ndarray) -> np.ndarray:
        """以重叠 patch 推理并对重叠区域平均概率。"""
        size = int(self.patch_config.get("size", self.img_size))
        stride = int(self.patch_config.get("stride", max(size // 2, 1)))
        if stride <= 0 or stride > size:
            raise ValueError("inference.patch.stride 必须在 1 到 patch.size 之间")
        height, width = image.shape[:2]
        padded_height = max(height, size)
        padded_width = max(width, size)
        padded_shape = (padded_height, padded_width) + image.shape[2:]
        padded = np.zeros(padded_shape, dtype=image.dtype)
        padded[:height, :width, ...] = image
        rows = self._sliding_positions(padded_height, size, stride)
        cols = self._sliding_positions(padded_width, size, stride)
        probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
        visit_count = np.zeros((padded_height, padded_width), dtype=np.float32)
        transform = get_val_transforms(size, keep_aspect_ratio=False)
        with torch.inference_mode():
            for top in rows:
                for left in cols:
                    patch = padded[top:top + size, left:left + size]
                    tensor = transform(image=patch)["image"].unsqueeze(0).to(self.device)
                    prediction = torch.sigmoid(main_logits_from_output(self.model(tensor)))
                    probability = prediction[0, 0].detach().cpu().numpy()
                    probability_sum[top:top + size, left:left + size] += probability
                    visit_count[top:top + size, left:left + size] += 1.0
        return probability_sum[:height, :width] / np.maximum(visit_count[:height, :width], 1.0)

    @staticmethod
    def _sliding_positions(length: int, size: int, stride: int) -> list[int]:
        if length <= size:
            return [0]
        positions = list(range(0, length - size + 1, stride))
        last = length - size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _restore_original_geometry(self, mask: np.ndarray, original_h: int, original_w: int, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
        """Remove validation/inference padding before resizing the prediction back."""
        return restore_original_geometry(
            mask, original_h, original_w, self.img_size, self.keep_aspect_ratio, interpolation
        )

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
    parser.add_argument("--device", default=None, help="Device to run inference on (auto by default)")
    args = parser.parse_args()
    if args.device:
        try:
            requested_device = torch.device(args.device)
        except RuntimeError as exc:
            parser.error(f"Invalid --device value: {exc}")
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is not available")

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
        device=args.device,
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
        # 使用 imencode + 二进制写文件，支持 Windows 中文路径。
        success, encoded = cv2.imencode(".png", segmentor.predict(str(image_path)))
        if not success:
            raise IOError(f"Failed to encode output: {out_path}")
        with open(out_path, "wb") as file:
            file.write(encoded.tobytes())
        print(f"  {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
