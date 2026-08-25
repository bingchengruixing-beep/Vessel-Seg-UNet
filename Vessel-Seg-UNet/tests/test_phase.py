import numpy as np
import pytest
import torch

from src.dataset import VesselDataset, phase_id_from_filename
from src.models import build_model


def _build(name, **kwargs):
    torch.manual_seed(7)
    return build_model(name, in_channels=1, out_channels=1, **kwargs)


def test_film_identity_when_phase_none():
    torch.manual_seed(7)
    base = _build("unet_baseline")
    torch.manual_seed(7)
    conditional = _build("unet_baseline", phase_classes=4)
    x = torch.randn(2, 1, 64, 64)
    base_logits = base(x)
    cond_logits = conditional(x)[0]
    # 零初始化 FiLM: phase=None 时条件模型与无条件模型输出一致
    assert torch.allclose(base_logits, cond_logits, atol=1e-5)


def test_film_changes_output_with_phase():
    conditional = _build("unet_baseline", phase_classes=4)
    x = torch.randn(2, 1, 64, 64)
    p0 = torch.zeros(2, dtype=torch.long)
    p2 = torch.full((2,), 2, dtype=torch.long)
    # 扰动 FiLM 权重后, 不同相位条件产生不同输出
    with torch.no_grad():
        for layer in conditional.film.layers:
            layer.weight.uniform_(-0.5, 0.5)
            layer.bias.uniform_(-0.5, 0.5)
    out_a = conditional(x, p0)[0]
    out_b = conditional(x, p2)[0]
    assert not torch.allclose(out_a, out_b)


def test_phase_head_output_shape():
    conditional = _build("unet_baseline", phase_classes=4)
    x = torch.randn(2, 1, 64, 64)
    logits, phase_logits = conditional(x, torch.tensor([1, 2]))
    assert tuple(logits.shape) == (2, 1, 64, 64)
    assert tuple(phase_logits.shape) == (2, 4)


def test_phase_id_from_filename():
    assert phase_id_from_filename("d1_2-3s_5.png") == 0
    assert phase_id_from_filename("d1_4s_5.png") == 1
    assert phase_id_from_filename("d1_5-6s_5.png") == 2
    assert phase_id_from_filename("d2_5.png") == 3
    with pytest.raises(ValueError):
        phase_id_from_filename("other.png")


def test_dataset_returns_phase(tmp_path):
    import cv2
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 10:54] = 255
    ok, buf = cv2.imencode(".png", image)
    (img_dir / "d2_7.png").write_bytes(buf.tobytes())
    ok, buf = cv2.imencode(".png", mask)
    (mask_dir / "d2_7.png").write_bytes(buf.tobytes())

    dataset = VesselDataset(str(img_dir), str(mask_dir), return_phase=True)
    out_image, out_mask, phase = dataset[0]
    assert out_image.shape == out_mask.shape == (1, 64, 64)
    assert phase.item() == 3
