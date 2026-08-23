import pytest
import torch

from src.metrics import (
    MetricAccumulator,
    calculate_dice,
    calculate_iou,
    calculate_precision,
    calculate_recall,
)


def _mask(values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def test_perfect_prediction_scores_one():
    pred = _mask([[1, 1], [1, 1]])
    gt = _mask([[1, 1], [1, 1]])

    assert calculate_dice(pred, gt) == pytest.approx(1.0)
    assert calculate_iou(pred, gt) == pytest.approx(1.0)
    assert calculate_precision(pred, gt) == pytest.approx(1.0)
    assert calculate_recall(pred, gt) == pytest.approx(1.0)


def test_empty_prediction_scores_zero_not_one_half():
    pred = _mask([[0, 0], [0, 0]])
    gt = _mask([[1, 1], [0, 0]])

    assert calculate_precision(pred, gt) == pytest.approx(0.0)
    assert calculate_recall(pred, gt) == pytest.approx(0.0)
    assert calculate_dice(pred, gt) == pytest.approx(0.0)


def test_both_empty_convention():
    empty = _mask([[0, 0], [0, 0]])
    # 约定：P、T 均为空时 Dice/IoU 记为 1.0；Precision/Recall 记为 0。
    assert calculate_dice(empty, empty) == pytest.approx(1.0)
    assert calculate_iou(empty, empty) == pytest.approx(1.0)
    assert calculate_precision(empty, empty) == pytest.approx(0.0)
    assert calculate_recall(empty, empty) == pytest.approx(0.0)


def test_partial_overlap_matches_closed_form():
    pred = _mask([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    gt = _mask([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # |P| = 2, |T| = 4, 交集 = 2
    assert calculate_dice(pred, gt) == pytest.approx(2 * 2 / (2 + 4), abs=1e-6)
    assert calculate_iou(pred, gt) == pytest.approx(2 / (2 + 4 - 2), abs=1e-6)
    assert calculate_precision(pred, gt) == pytest.approx(2 / 2, abs=1e-6)
    assert calculate_recall(pred, gt) == pytest.approx(2 / 4, abs=1e-6)


def test_accumulator_matches_manual_global_metric():
    acc = MetricAccumulator()
    acc.update(_mask([[1, 1], [0, 0]]), _mask([[1, 0], [1, 0]]))
    acc.update(_mask([[0, 0], [0, 1]]), _mask([[0, 1], [0, 1]]))
    # 全局: 交集 = 2, |P| = 3, |T| = 4
    assert acc.dice() == pytest.approx(2 * 2 / (3 + 4), abs=1e-6)
    assert acc.iou() == pytest.approx(2 / (3 + 4 - 2), abs=1e-6)
    assert acc.precision() == pytest.approx(2 / 3, abs=1e-6)
    assert acc.recall() == pytest.approx(2 / 4, abs=1e-6)
