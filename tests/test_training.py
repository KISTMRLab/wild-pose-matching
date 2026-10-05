import json

import numpy as np
import pytest
import torch
from wild_pose_matching.model import GestureCLR, ntxent
from wild_pose_matching.training import PRESETS, config_from, load_checkpoint, save_checkpoint, train_gestureclr
from wild_pose_matching.units import Normalization


def test_ntxent_matches_paper_formula_and_rewards_aligned_pairs():
    torch.manual_seed(0)
    z2 = torch.nn.functional.normalize(torch.randn(6, 10), dim=-1)
    z3 = torch.nn.functional.normalize(torch.randn(6, 10), dim=-1)
    t = .1
    sim = (z2 @ z3.T) / t
    row = -(torch.diag(sim) - torch.logsumexp(sim, 1)).mean()
    col = -(torch.diag(sim) - torch.logsumexp(sim, 0)).mean()
    assert float(ntxent(z2, z3, t)) == pytest.approx(float((row + col) / 2), rel=1e-5)
    # Identical (aligned) pairs score lower than mismatched ones.
    assert float(ntxent(z3, z3, t)) < float(ntxent(z3[torch.randperm(6)], z3, t))
    # Gradient descent on the 2D latents moves each one towards its own 3D partner.
    q = z2.clone().requires_grad_(True)
    loss = ntxent(torch.nn.functional.normalize(q, dim=-1), z3, t)
    loss.backward()
    with torch.no_grad():
        moved = torch.nn.functional.normalize(q - .05 * q.grad, dim=-1)
    assert float(torch.diag(moved @ z3.T).mean()) > float(torch.diag(z2 @ z3.T).mean())
    assert float(ntxent(moved, z3, t)) < float(loss.detach())


def test_masks_hide_padded_frames():
    torch.manual_seed(0)
    model = GestureCLR(4, 6).eval()
    x2, x3 = torch.randn(2, 45, 4), torch.randn(2, 45, 6)
    mask = torch.zeros(2, 45, dtype=torch.bool); mask[:, :30] = True
    a = model.motion3d(x3, mask)
    x3b = x3.clone(); x3b[:, 30:] = 99
    assert torch.allclose(a, model.motion3d(x3b, mask), atol=1e-5)


def test_config_presets_and_overrides(tmp_path):
    assert PRESETS["paper"].epochs == 1000 and PRESETS["paper"].batch_size == 512 and PRESETS["paper"].lr == 5e-4
    assert PRESETS["demo"].epochs >= 300
    (tmp_path / "c.json").write_text(json.dumps({"temperature": .2, "noise_variances": [.01]}))
    cfg = config_from("demo", tmp_path / "c.json", lr=1e-3)
    assert cfg.temperature == .2 and cfg.noise_variances == (.01,) and cfg.lr == 1e-3
    with pytest.raises(ValueError):
        config_from("huge")


def test_training_schedule_validation_best_checkpoint_and_reload(tmp_path):
    rng = np.random.default_rng(0)
    n, frames, joints = 24, 45, 11
    motion = rng.normal(size=(n, frames, joints, 3)).astype(np.float32).cumsum(1) * .2
    motion[:, :, 1] = 0
    mask = np.ones((n, frames), bool); mask[::2, 35:] = False
    motion[~mask] = 0
    norm = Normalization(1, None)
    x3 = norm.apply(motion.reshape(n, frames, -1), 3, mask)
    x2 = norm.apply(motion[..., :2].reshape(n, frames, -1), 2, mask)
    cfg = config_from("demo", epochs=30, batch_size=8, val_fraction=.25, lr=1e-3)
    model, history = train_gestureclr(x2, x3, mask, cfg, log=None)
    rows = history["epochs"]
    assert history["val_pairs"] == 6 and len(rows) == 30
    assert rows[-1]["lr"] < 1e-5 < rows[0]["lr"]  # cosine annealing reaches ~0
    assert all("val_loss" in r and 0 <= r["val_top1"] <= 1 for r in rows)
    assert history["best_epoch"] == 1 + int(np.argmin([r["val_loss"] for r in rows]))
    assert rows[-1]["loss"] < rows[0]["loss"]
    save_checkpoint(tmp_path / "m.pt", model, x2.shape[-1], x3.shape[-1], cfg, norm, history)
    loaded, loaded_norm = load_checkpoint(tmp_path / "m.pt")
    assert loaded_norm == norm
    with torch.no_grad():
        assert torch.allclose(loaded.motion3d(torch.from_numpy(x3[:4]), torch.from_numpy(mask[:4])),
                              model.motion3d(torch.from_numpy(x3[:4]), torch.from_numpy(mask[:4])))
