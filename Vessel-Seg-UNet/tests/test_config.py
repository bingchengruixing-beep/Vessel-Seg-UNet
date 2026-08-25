import pytest

from src.config import ConfigError, normalize_config, resolve_checkpoint_dir


def test_legacy_config_is_migrated_to_canonical_training_sections(tmp_path):
    config = normalize_config({
        "loss": {"bce_weight": 0.25, "dice_weight": 0.75},
        "checkpoint": {"save_dir": "artifacts", "save_best_only": False},
        "training": {"epochs": 3},
    })

    assert config["training"]["loss"]["bce_weight"] == 0.25
    assert config["training"]["checkpoint"]["save_dir"] == "artifacts"
    assert config["training"]["checkpoint"]["save_best_only"] is False
    assert resolve_checkpoint_dir(config, tmp_path) == tmp_path / "artifacts"


def test_checkpoint_directory_cannot_escape_project_root():
    with pytest.raises(ConfigError, match="project-relative"):
        normalize_config({"training": {"checkpoint": {"save_dir": "../outside"}}})


def test_temporal_input_requires_three_channels_and_temporal_mode():
    with pytest.raises(ConfigError, match="in_channels"):
        normalize_config({"dataset": {"temporal_2_5d": {"enabled": True}}})

    config = normalize_config({
        "dataset": {"temporal_2_5d": {"enabled": True}},
        "model": {"in_channels": 3, "input_mode": "temporal"},
    })
    assert config["model"]["in_channels"] == 3


def test_temporal_input_and_frangi_are_mutually_exclusive():
    with pytest.raises(ConfigError, match="cannot be enabled together"):
        normalize_config({
            "dataset": {
                "temporal_2_5d": {"enabled": True},
                "frangi": {"enabled": True},
            },
            "model": {"in_channels": 3, "input_mode": "temporal"},
        })


def test_loader_speed_options_are_validated():
    config = normalize_config({})
    assert config["dataset"]["loader"] == {
        "persistent_workers": True,
        "prefetch_factor": 2,
        "cache_size": 32,
    }

    with pytest.raises(ConfigError, match="prefetch_factor"):
        normalize_config({"dataset": {"loader": {"prefetch_factor": 0}}})
