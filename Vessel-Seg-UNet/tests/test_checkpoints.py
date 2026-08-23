import torch

from src.checkpoints import checkpoint_model_config, load_checkpoint, load_model_state, save_checkpoint
from src.config import normalize_config


def test_versioned_checkpoint_round_trip(tmp_path):
    config = normalize_config({"model": {"name": "unet_baseline"}})
    source_model = torch.nn.Conv2d(1, 1, kernel_size=1)
    path = tmp_path / "model.pth"
    save_checkpoint(
        path,
        model=source_model,
        optimizer=None,
        scheduler=None,
        epoch=2,
        best_dice=0.8,
        metrics={"dice": 0.8},
        config=config,
    )

    checkpoint = load_checkpoint(path)
    target_model = torch.nn.Conv2d(1, 1, kernel_size=1)
    load_model_state(target_model, checkpoint)

    assert checkpoint["format_version"] == 2
    assert checkpoint["epoch"] == 2
    assert checkpoint_model_config(checkpoint, config)["name"] == "unet_baseline"
    assert all(torch.equal(a, b) for a, b in zip(source_model.parameters(), target_model.parameters()))
