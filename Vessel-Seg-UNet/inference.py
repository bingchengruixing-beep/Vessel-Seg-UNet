"""
[M5] 单张/批量图像推理 API
将训练好的 best_model.pth 封装为易用的黑盒接口。

使用示例:
    from inference import VesselSegmentor
    seg = VesselSegmentor('checkpoints/best_model.pth')
    mask = seg.predict('path/to/image.png')
    cv2.imwrite('output_mask.png', mask)
"""

import os
import cv2
import numpy as np
import torch

from src.models import build_model
from src.transforms import get_val_transforms
from src.postprocess import postprocess_mask


class VesselSegmentor:
    """
    端到端血管分割推理器。

    加载训练好的模型权重，对输入图像执行完整的推理管线：
    预处理 → 模型前向传播 → Sigmoid + 二值化 → 后处理去噪

    Args:
        model_path: 模型权重文件路径（.pth）
        model_name: 模型架构名称（默认 'unet_baseline'）
        device: 推理设备（自动检测 CUDA/CPU）
        img_size: 推理时 Resize 的尺寸
        threshold: 二值化阈值（Sigmoid 后）
        min_component_size: 后处理连通域最小保留面积
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = 'unet_baseline',
        device: str = None,
        img_size: int = 512,
        threshold: float = 0.5,
        min_component_size: int = 50,
    ):
        # 设备自动检测
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.img_size = img_size
        self.threshold = threshold
        self.min_component_size = min_component_size

        # 构建模型并加载权重
        self.model = build_model(model_name, in_channels=1, out_channels=1)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # 推理用的增强管线（仅 Resize + Normalize）
        self.transform = get_val_transforms(img_size)

    @torch.no_grad()
    def predict(self, image_path: str) -> np.ndarray:
        """
        端到端推理单张图像。

        流程:
            读取图像 → 预处理 (M1 逻辑) → 模型前向传播 (M2) →
            Sigmoid + 二值化 → 后处理去噪 (M5) → 返回最终掩膜

        Args:
            image_path: 输入灰度造影图像路径

        Returns:
            (H_original, W_original) uint8 二值掩膜 {0, 255}
        """
        # 读取原始图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")

        original_h, original_w = image.shape[:2]

        # 预处理
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # 模型前向传播
        logits = self.model(tensor)  # (1, 1, H, W) raw logits

        # Sigmoid + 二值化
        prob = torch.sigmoid(logits)
        pred = (prob > self.threshold).float()

        # 转回 numpy
        mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255

        # 还原到原始尺寸
        if mask.shape != (original_h, original_w):
            mask = cv2.resize(
                mask, (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 后处理去噪
        mask = postprocess_mask(
            mask,
            min_component_size=self.min_component_size,
        )

        return mask

    @torch.no_grad()
    def predict_batch(self, image_paths: list) -> list:
        """
        批量推理多张图像。

        Args:
            image_paths: 图像路径列表

        Returns:
            掩膜列表 [(H, W) uint8, ...]
        """
        return [self.predict(p) for p in image_paths]

    @torch.no_grad()
    def predict_array(self, image: np.ndarray) -> np.ndarray:
        """
        直接接受 numpy 灰度图数组推理（供 Web API 调用）。

        Args:
            image: (H, W) uint8 灰度图 numpy 数组

        Returns:
            (H_original, W_original) uint8 二值掩膜 {0, 255}
        """
        original_h, original_w = image.shape[:2]

        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        prob = torch.sigmoid(logits)
        pred = (prob > self.threshold).float()
        mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255

        if mask.shape != (original_h, original_w):
            mask = cv2.resize(mask, (original_w, original_h),
                              interpolation=cv2.INTER_NEAREST)

        mask = postprocess_mask(mask, min_component_size=self.min_component_size)
        return mask


def main():
    """命令行推理入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Vessel Segmentation Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory for predictions')
    parser.add_argument('--model-name', type=str, default='unet_baseline',
                        help='Model architecture name')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Inference image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold')
    parser.add_argument('--min-size', type=int, default=50,
                        help='Min component size for postprocessing')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    segmentor = VesselSegmentor(
        model_path=args.model,
        model_name=args.model_name,
        img_size=args.img_size,
        threshold=args.threshold,
        min_component_size=args.min_size,
    )

    # 单张或目录
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        image_paths = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        ])

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        mask = segmentor.predict(img_path)
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output, fname)

        success, buf = cv2.imencode('.png', mask)
        if success:
            with open(out_path, 'wb') as f:
                f.write(buf.tobytes())
            print(f"  ✓ {fname}")

    print(f"Done. Results saved to: {args.output}")


if __name__ == '__main__':
    main()
