"""
[M2] 模型工厂
统一的模型构建入口，方便训练脚本中切换不同模型。
"""

import torch.nn as nn

from src.models.unet import UNetBaseline
from src.models.attention_unet import AttentionUNet
from src.models.unet_resnet import UNetResNet
from src.models.resunet_aspp import ResUNetASPP


# 模型注册表
_MODEL_REGISTRY = {
    'unet_baseline': UNetBaseline,
    'attention_unet': AttentionUNet,
    'unet_resnet': UNetResNet,
    'resunet_aspp': ResUNetASPP,
}


def build_model_from_config(model_config: dict) -> nn.Module:
    """根据配置构建模型，集中处理不同模型的专属参数。"""
    model_name = model_config["name"]
    kwargs = {
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
    }
    if model_name == "unet_resnet":
        kwargs.update(
            encoder_name=model_config.get("encoder_name", "resnet34"),
            pretrained=bool(model_config.get("pretrained", True)),
        )
    elif model_name == "resunet_aspp":
        kwargs.update(
            pretrained=bool(model_config.get("pretrained", True)),
            deep_supervision=bool(model_config.get("deep_supervision", True)),
        )
    return build_model(model_name, **kwargs)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    根据名称构建模型实例。

    Args:
        model_name: 模型名称，需在 _MODEL_REGISTRY 中注册
        **kwargs: 传递给模型构造函数的参数（如 in_channels, out_channels）

    Returns:
        nn.Module 实例

    Raises:
        ValueError: 未知的模型名称
    """
    if model_name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {available}"
        )

    model_cls = _MODEL_REGISTRY[model_name]
    return model_cls(**kwargs)
