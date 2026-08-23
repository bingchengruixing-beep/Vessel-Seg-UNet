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


@pytest.mark.parametrize("model_name", ["unet_baseline", "attention_unet", "unet_resnet"])
def test_valid_model_names_are_accepted(model_name):
    config = normalize_config({"model": {"name": model_name}})
    assert config["model"]["name"] == model_name


def test_unknown_model_name_raises_config_error():
    with pytest.raises(ConfigError, match="model.name must be"):
        normalize_config({"model": {"name": "invalid_model_arch"}})

