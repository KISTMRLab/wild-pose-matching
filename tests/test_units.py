import json

import numpy as np
import pytest
from wild_pose_matching.cli import main as cli_main
from wild_pose_matching.units import Normalization, extract_units, pack_units, suggest_variance_threshold, window_variances

BASE = np.array([[0, -50, 0], [0, 0, 0], [0, 15, 0], [-5, -2, 0], [-18, -3, 0], [-30, -25, 5], [-32, -48, 10],
                 [5, -2, 0], [18, -3, 0], [30, -25, 5], [32, -48, 10]], np.float32)  # 11-joint upper body, cm, neck at index 1


def make_motion(plan, rng=None, noise=0.05):
    """plan: list of ("still", frames) or ("gesture", frames); a gesture is a closed hand loop (start pose == end pose)."""
    rng = rng or np.random.default_rng(0)
    frames = []
    for kind, n in plan:
        clip = np.repeat(BASE[None], n, 0).copy()
        if kind == "gesture":
            phase = (1 - np.cos(np.linspace(0, 2 * np.pi, n))) / 2
            clip[:, [5, 6], 1] += 30 * phase[:, None]
            clip[:, [9, 10], 0] += 20 * np.sin(np.linspace(0, 2 * np.pi, n))[:, None]
        frames.append(clip)
    motion = np.concatenate(frames)
    return motion + rng.normal(0, noise, motion.shape).astype(np.float32)


def test_normalization_is_translation_and_scale_invariant_and_keeps_padding_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(45, 11, 3)).astype(np.float32) + BASE
    moved = 2.5 * x + np.array([100, -20, 7], np.float32)
    norm = Normalization()
    assert np.allclose(norm.apply(x, 3), norm.apply(moved, 3), atol=1e-4)
    out = norm.apply(BASE[None].repeat(45, 0), 3)
    assert np.allclose(np.linalg.norm(out[:, 4] - out[:, 8], axis=-1), 1)
    mask = np.zeros(45, bool); mask[:30] = True
    flat = norm.apply(x.reshape(1, 45, -1), 3, mask[None])
    assert np.all(flat[0, 30:] == 0) and flat.shape == (1, 45, 33)
    with pytest.raises(ValueError, match="outside"):
        Normalization(1, (4, 20)).apply(x, 3)


def test_algorithm3_extracts_2_to_3_second_closed_units_and_skips_still_motion():
    plan = [("still", 25), ("gesture", 36), ("still", 25), ("gesture", 42), ("still", 25), ("gesture", 32), ("still", 25)]
    motion = make_motion(plan)
    gesture_var = window_variances([make_motion([("gesture", 36)])], 3, 36)[0]
    spans = extract_units(motion, variance_threshold=gesture_var / 4)
    assert spans and all(30 <= e - s <= 45 for s, e in spans)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    starts = np.cumsum([0] + [n for _, n in plan])[:-1]
    peaks = [s + n // 2 for (kind, n), s in zip(plan, starts) if kind == "gesture"]
    assert all(any(s <= p < e for s, e in spans) for p in peaks)
    x = Normalization().apply(motion, 3).reshape(len(motion), -1)
    for s, e in spans:
        assert np.linalg.norm(x[s] - x[e - 1]) < 0.2  # minimal start-end distance in shoulder widths
    # A pure still take yields nothing once the variance threshold is above noise.
    assert extract_units(make_motion([("still", 200)]), variance_threshold=gesture_var / 4) == []


def test_variance_threshold_helpers():
    v = np.concatenate([np.full(80, .01), np.linspace(.02, 1, 20)])
    assert suggest_variance_threshold(v, "percentile", 50) == pytest.approx(.01)
    assert .01 <= suggest_variance_threshold(v, "elbow") <= .1
    with pytest.raises(ValueError):
        suggest_variance_threshold(v, "kmeans")


def test_pack_units_masks_lengths_and_unique_ids():
    motion = make_motion([("gesture", 45)] * 2)
    packed = pack_units({"take_a": motion, "take_b": motion}, {"take_a": [(0, 30), (40, 85)], "take_b": [(5, 40)]}, 45)
    assert packed["motion3d"].shape == (3, 45, 33) and packed["pose2d"].shape == (3, 45, 22)
    assert packed["lengths"].tolist() == [30, 45, 35] and packed["mask"].sum(1).tolist() == [30, 45, 35]
    assert packed["ids"].tolist() == ["take_a:0-30", "take_a:40-85", "take_b:5-40"]
    assert np.all(packed["motion3d"][0, 30:] == 0)


def test_cli_extract_units_from_motion_npz(tmp_path, capsys):
    plan = [("still", 25), ("gesture", 36), ("still", 25), ("gesture", 40), ("still", 25)]
    for take in ("t1", "t2"):
        np.savez(tmp_path / f"{take}.npz", motion=make_motion(plan, np.random.default_rng(len(take))), take=np.asarray(take))
    cli_main(["extract-units", "--motion", str(tmp_path / "t1.npz"), str(tmp_path / "t2.npz"), "--variance-percentile", "50", "--output", str(tmp_path / "units.npz")])
    summary = json.loads(capsys.readouterr().out)
    data = np.load(tmp_path / "units.npz")
    assert summary["units"] == len(data["ids"]) >= 2 and set(data["takes"].tolist()) == {"t1", "t2"}
    assert data["mask"].shape == data["motion3d"].shape[:2] and int(data["dim2"]) == 22
    report = json.loads((tmp_path / "units.npz.report.json").read_text())
    assert report["variance_rule"] == "percentile 50" and report["per_take"]["t1"] >= 1
