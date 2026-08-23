import pytest
import torch

from src.losses import BCEDiceLoss, DiceLoss


def test_dice_loss_averages_per_sample_across_batch():
    # 样本 1: 完美预测 (dice≈1 → loss≈0)；样本 2: 全背景预测 (dice≈0 → loss≈1)
    logits = torch.tensor([
        [[[20.0, 20.0, -20.0, -20.0]]],
        [[[-20.0, -20.0, -20.0, -20.0]]],
    ])
    targets = torch.tensor([
        [[[1.0, 1.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.0, 1.0]]],
    ])

    loss = DiceLoss()(logits, targets)

    assert loss.item() == pytest.approx(0.5, abs=1e-3)


def test_dice_loss_is_zero_for_perfect_and_one_for_disjoint():
    targets = torch.tensor([[[[1.0, 0.0]]]])
    perfect_logits = torch.tensor([[[[20.0, -20.0]]]])
    disjoint_logits = torch.tensor([[[[-20.0, 20.0]]]])

    assert DiceLoss()(perfect_logits, targets).item() == pytest.approx(0.0, abs=1e-3)
    assert DiceLoss()(disjoint_logits, targets).item() == pytest.approx(1.0, abs=1e-3)


def test_bce_dice_loss_composes_with_weights():
    logits = torch.tensor([[[[0.5, -0.5]]]])
    targets = torch.tensor([[[[1.0, 0.0]]]])

    bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
    dice = DiceLoss()(logits, targets)
    expected = 0.25 * bce + 0.75 * dice

    actual = BCEDiceLoss(bce_weight=0.25, dice_weight=0.75)(logits, targets)

    assert torch.isclose(actual, expected)
