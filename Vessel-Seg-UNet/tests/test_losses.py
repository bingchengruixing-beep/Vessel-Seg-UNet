import pytest
import torch

from src.losses import (
    BCEDiceLoss,
    CLDiceLoss,
    CombinedVesselLoss,
    DiceLoss,
    FocalTverskyLoss,
)


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


def test_focal_tversky_zero_for_perfect_and_one_for_disjoint():
    targets = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
    perfect = torch.tensor([[[[20.0, 20.0, -20.0, -20.0]]]])
    disjoint = torch.tensor([[[[-20.0, -20.0, 20.0, 20.0]]]])
    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

    assert loss_fn(perfect, targets).item() == pytest.approx(0.0, abs=1e-3)
    assert loss_fn(disjoint, targets).item() == pytest.approx(1.0, abs=1e-3)


def test_focal_tversky_penalizes_fn_more_than_fp():
    # 目标: 4 前景 + 4 背景。FN 情形漏 2 个前景, FP 情形多 2 个前景。
    # α(0.7) > β(0.3) 时, 同样 2 像素错误, 漏检损失应更大。
    targets = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]])
    fn_logits = torch.tensor([[[[20.0, 20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]]]])
    fp_logits = torch.tensor([[[[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, -20.0, -20.0]]]])
    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0)

    assert loss_fn(fn_logits, targets).item() > loss_fn(fp_logits, targets).item()


def test_cl_dice_loss_zero_for_thick_perfect_prediction():
    # 12x12 实心块足够厚: 软骨架 5 次迭代后仍存在且完全落在目标内
    skeleton = torch.zeros((1, 1, 20, 20))
    skeleton[:, :, 4:16, 4:16] = 1.0
    perfect = torch.where(skeleton > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    empty = torch.full_like(skeleton, -20.0)

    loss_fn = CLDiceLoss(iters=5)
    assert loss_fn(perfect, skeleton).item() == pytest.approx(0.0, abs=0.05)
    assert loss_fn(empty, skeleton).item() == pytest.approx(1.0, abs=0.05)


def test_combined_loss_adds_cldice_term():
    logits = torch.tensor([[[[0.5, -0.5, 0.5, -0.5]]]])
    targets = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    skeleton = targets.clone()
    main = BCEDiceLoss()
    combined = CombinedVesselLoss(main, CLDiceLoss(), cldice_weight=0.5)

    base = main(logits, targets)
    total = combined(logits, targets, skeleton)
    assert total.item() > base.item()
    # 无骨架时等于主损失
    assert torch.isclose(combined(logits, targets, None), base)
