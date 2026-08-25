"""血管增强滤波器，用于 DSA 脑血管造影图像的管状结构增强。

提供两种方法：
    1. Frangi vesselness — 经典多尺度 Hessian 方法，适合管状结构明显的图像
    2. Hessian vesselness — 简化的 -λ2 方法，适合 DSA 等血管边缘锐利的图像（推荐）

纯 NumPy/SciPy 实现，不依赖 GPU。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _hessian_eigenvalues(image: np.ndarray, sigma: float) -> np.ndarray:
    """计算图像在尺度 sigma 下的 Hessian 矩阵特征值。

    Args:
        image: (H, W) float64 灰度图像
        sigma: 高斯核标准差

    Returns:
        (H, W, 2) 数组，λ1=[:,:,0], λ2=[:,:,1]，|λ1| ≤ |λ2|
    """
    Lxx = gaussian_filter(image, sigma, order=(2, 0))
    Lxy = gaussian_filter(image, sigma, order=(1, 1))
    Lyy = gaussian_filter(image, sigma, order=(0, 2))

    trace = Lxx + Lyy
    discriminant = np.sqrt((Lxx - Lyy) ** 2 + 4.0 * Lxy ** 2 + 1e-12)

    lambda1 = (trace - discriminant) / 2.0
    lambda2 = (trace + discriminant) / 2.0

    return np.stack([lambda1, lambda2], axis=-1)


def hessian_vesselness(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    black_ridges: bool = False,
) -> np.ndarray:
    """基于 Hessian 矩阵第二特征值的多尺度血管增强。

    对亮血管（暗背景），取 -λ2（λ2 < 0 时为正），多尺度取最大值。
    公式简单但非常有效，尤其适合 DSA 等血管边缘锐利的图像。

    Args:
        image: (H, W) 灰度图像，值域 [0, 255]
        sigmas: 多尺度高斯核标准差序列
        black_ridges: True 检测暗血管，False 检测亮血管（DSA 默认）

    Returns:
        (H, W) float32 vesselness 概率图，值域 [0, 1]
    """
    if image.ndim != 2:
        raise ValueError(f"hessian_vesselness 需要 2D 灰度图，收到 {image.ndim}D")

    image_f = image.astype(np.float64)
    if black_ridges:
        image_f = 255.0 - image_f

    vesselness = np.zeros(image.shape, dtype=np.float32)
    for sigma in sigmas:
        eigenvalues = _hessian_eigenvalues(image_f, sigma)
        lambda2 = eigenvalues[..., 1]
        # 亮血管：λ2 < 0，取 -λ2；λ2 ≥ 0 取 0
        scale_response = np.maximum(-lambda2, 0.0).astype(np.float32)
        vesselness = np.maximum(vesselness, scale_response)

    # 用 99.5 分位数做归一化
    v_ref = float(np.percentile(vesselness, 99.5))
    if v_ref > 1e-12:
        vesselness = np.clip(vesselness / v_ref, 0.0, 1.0)

    return vesselness


def frangi_vesselness(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    beta: float = 0.5,
    c: float | None = None,
    black_ridges: bool = False,
) -> np.ndarray:
    """多尺度 Frangi 血管增强滤波器（经典方法）。

    使用 blobness 抑制项 Rb 和结构强度项 S 来区分管状结构和斑点噪声。
    注意：DSA 血管边缘锐利，Rb 值偏高，建议 beta 设为 1.5~3.0，
    或直接使用 hessian_vesselness() 获得更好的效果。

    Args:
        image: (H, W) 灰度图像，值域 [0, 255]
        sigmas: 多尺度高斯核标准差序列
        beta: blobness 抑制参数，值越大越容忍非管状结构（DSA 建议 1.5~3.0）
        c: 噪声抑制参数，None 时自动计算
        black_ridges: True 检测暗血管，False 检测亮血管（DSA 默认）

    Returns:
        (H, W) float32 vesselness 概率图，值域 [0, 1]
    """
    if image.ndim != 2:
        raise ValueError(f"frangi_vesselness 需要 2D 灰度图，收到 {image.ndim}D")

    image_f = image.astype(np.float64)
    if black_ridges:
        image_f = 255.0 - image_f

    vesselness = np.zeros(image.shape, dtype=np.float32)
    for sigma in sigmas:
        eigenvalues = _hessian_eigenvalues(image_f, sigma)
        lambda1 = eigenvalues[..., 0]
        lambda2 = eigenvalues[..., 1]

        vessel_mask = lambda2 < 0.0

        # Rb = |λ1| / |λ2|
        Rb = np.zeros_like(lambda2)
        nonzero = np.abs(lambda2) > 1e-12
        Rb[nonzero] = np.abs(lambda1[nonzero]) / np.abs(lambda2[nonzero])

        # S = sqrt(λ1² + λ2²)
        S = np.sqrt(lambda1 ** 2 + lambda2 ** 2)

        if c is None:
            if vessel_mask.any():
                c = float(np.median(S[vessel_mask])) / 2.0
            else:
                c = 1.0
            c = max(c, 0.5)

        scale_vesselness = np.zeros_like(lambda2)
        scale_vesselness[vessel_mask] = (
            np.exp(-(Rb[vessel_mask] ** 2) / (2.0 * beta ** 2))
            * (1.0 - np.exp(-(S[vessel_mask] ** 2) / (2.0 * c ** 2)))
        )
        vesselness = np.maximum(vesselness, scale_vesselness.astype(np.float32))

    v_ref = float(np.percentile(vesselness, 99.5))
    if v_ref > 1e-12:
        vesselness = np.clip(vesselness / v_ref, 0.0, 1.0)

    return vesselness


def vesselness(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    method: str = "hessian",
    beta: float = 0.5,
    c: float | None = None,
    black_ridges: bool = False,
) -> np.ndarray:
    """统一的血管增强入口。

    Args:
        image: (H, W) 灰度图像，值域 [0, 255]
        sigmas: 多尺度高斯核标准差序列
        method: "hessian"（推荐DSA）或 "frangi"
        beta: Frangi blobness 参数（仅 method="frangi" 时使用）
        c: Frangi 噪声抑制参数（仅 method="frangi" 时使用）
        black_ridges: True 检测暗血管

    Returns:
        (H, W) float32 vesselness 概率图，值域 [0, 1]
    """
    if method == "frangi":
        return frangi_vesselness(image, sigmas=sigmas, beta=beta, c=c, black_ridges=black_ridges)
    return hessian_vesselness(image, sigmas=sigmas, black_ridges=black_ridges)


def generate_frangi_map(
    image_path: str,
    output_path: str,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    method: str = "hessian",
    beta: float = 0.5,
    c: float | None = None,
) -> None:
    """读取图像，生成血管增强图并保存。

    Args:
        image_path: 输入图像路径
        output_path: 输出增强图保存路径
        sigmas: 多尺度参数
        method: "hessian" 或 "frangi"
        beta: Frangi blobness 参数
        c: Frangi 噪声抑制参数
    """
    import cv2
    import os

    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"无法读取图像: {image_path}")

    v = vesselness(image, sigmas=sigmas, method=method, beta=beta, c=c)

    v_16u = (v * 65535.0).astype(np.uint16)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    success, encoded = cv2.imencode(".png", v_16u)
    if success:
        encoded.tofile(output_path)
    else:
        raise IOError(f"无法保存血管增强图: {output_path}")