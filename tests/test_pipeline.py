import numpy as np
import pytest
from wild_pose_matching.pipeline import add_noise, augment_projected, build_rules, cluster_latents, retrieve, sixgrams, temporal_shift


def test_utilities():
    assert sixgrams("one two three four five six seven")==["one two three four five six","seven"]
    labels,centers=cluster_latents(np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]),2); assert len(set(labels))==2 and centers.shape==(2,2)


def test_paired_latent_matching_and_cluster_sampling():
    z=np.eye(2,dtype=np.float32)
    rules=build_rules(z,["open hand","point right"],z,z,["open","point"],np.array([0,1]))
    assert [r["gesture_id"] for r in rules]==["open","point"]
    chosen=retrieve("point right",rules,lambda _:z[1:2],{0:["open"],1:["point"]},seed=3)
    assert chosen[0]["gesture_id"]=="point" and chosen[0]["cluster_id"]==1
    assert build_rules(z,["a","b"],z*.5,z,["open","point"],np.array([0,1]),min_pose_match=.9)==[]


def test_noise_standard_deviation_is_sqrt_of_paper_variance():
    rng = np.random.default_rng(0)
    for variance in (0.001, 0.01, 0.1):
        noisy = add_noise(np.zeros((200, 45, 22), np.float32), variance, rng)
        assert noisy.std() == pytest.approx(np.sqrt(variance), rel=0.02)
    mask = np.zeros(45, bool); mask[:30] = True
    assert np.all(add_noise(np.zeros((45, 4)), .1, rng, mask)[30:] == 0)


def test_temporal_shift_places_30_frame_crop_at_offset_1_to_15_in_45_frame_buffer():
    rng = np.random.default_rng(1)
    x = (np.arange(45.0)[:, None] ** 2 + np.arange(4)).astype(np.float32)  # no row equals the mean pose
    for fill in ("mean", "zero"):
        for _ in range(50):
            out = temporal_shift(x, rng, fill=fill)
            assert out.shape == (45, 4)
            body = np.flatnonzero(~np.all(out == (x.mean(0) if fill == "mean" else 0), axis=1))
            offset, start = body[0], int(round(np.sqrt(out[body[0], 0])))
            assert 1 <= offset <= 15 and len(body) == 30
            assert np.array_equal(out[offset:offset + 30], x[start:start + 30])
    # Only the real frames of a padded unit are cropped; the fill is that unit's mean pose.
    mask = np.zeros(45, bool); mask[:33] = True
    out = temporal_shift(x, rng, mask, fill="mean", offset=15, start=3)
    assert np.array_equal(out[15:45], x[3:33]) and np.allclose(out[:15], x[:33].mean(0))
    view, m = augment_projected(x, rng, mask, noise_variances=(0.0,), shift_prob=1.0)
    assert m.all() and view.shape == x.shape


def test_retrieve_timing_idle_and_unit_lengths():
    z = np.eye(2, dtype=np.float32)
    rules = build_rules(z, ["open hand", "point right"], z, z, ["open", "point"], np.array([0, 1]))
    text = "one two three four five six seven eight nine"
    encode = lambda chunks: np.stack([z[1] if i == 0 else np.array([.6, -.8], np.float32) for i in range(len(chunks))])
    out = retrieve(text, rules, encode, {0: ["open"], 1: ["point"]}, audio_seconds=3.0, min_similarity=.7, unit_seconds={"point": 2.4})
    assert out[0]["gesture_id"] == "point" and out[0]["duration_seconds"] == pytest.approx(2.0) and out[0]["playback_rate"] == pytest.approx(1.2)
    assert out[1]["gesture_id"] == "idle" and out[1]["map"] == "idle" and out[1]["start_seconds"] == pytest.approx(2.0)
