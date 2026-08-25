import pytest

from src.dataset import (
    domain_balanced_weights,
    grouped_kfold_split,
    sample_group_key,
)


def test_grouped_kfold_keeps_three_phases_in_one_fold():
    filenames = [
        f"dataset1_{phase}_{index:03d}.png"
        for index in range(1, 10)
        for phase in ("2-3s", "4s", "5-6s")
    ] + [f"dias_train_{index:03d}.png" for index in range(1, 7)]

    training, validation = grouped_kfold_split(filenames, 3, 1, 42)
    training_groups = {sample_group_key(name) for name in training}
    validation_groups = {sample_group_key(name) for name in validation}

    assert not training_groups.intersection(validation_groups)
    assert sorted(training + validation) == sorted(filenames)
    assert sum(name.startswith("dias_train_") for name in validation) == 2


def test_domain_balanced_weights_match_requested_probability():
    filenames = [f"own_{index}.png" for index in range(7)] + [
        f"dias_train_{index}.png" for index in range(3)
    ]
    weights = domain_balanced_weights(filenames, ["dias_train_"], 0.4)

    target_total = sum(
        weight for name, weight in zip(filenames, weights) if name.startswith("dias_train_")
    )
    source_total = sum(weights) - target_total
    assert target_total == pytest.approx(0.4)
    assert source_total == pytest.approx(0.6)


def test_group_key_binds_dataset1_phases_only_by_sequence_number():
    assert sample_group_key("dataset1_2-3s_007.png") == "dataset1_007"
    assert sample_group_key("dataset1_4s_007.png") == "dataset1_007"
    assert sample_group_key("dataset1_5-6s_007.png") == "dataset1_007"
    assert sample_group_key("dataset2_4s_007.png") == "dataset2_4s_007"
