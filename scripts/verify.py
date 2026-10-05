"""Exercise the full CLI route on procedural motion: extract-units -> train -> cluster -> mine -> retrieve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize

# 11-joint upper body in centimetres (Hips, Neck, Head, L/R Shoulder, Arm, ForeArm, Hand), neck at index 1.
BASE = np.array([[0, -50, 0], [0, 0, 0], [0, 15, 0], [-5, -2, 0], [-18, -3, 0], [-30, -25, 5], [-32, -48, 10],
                 [5, -2, 0], [18, -3, 0], [30, -25, 5], [32, -48, 10]], np.float32)
KINDS = {"raise_left": ([5, 6], 1, 30), "raise_right": ([9, 10], 1, 30), "spread": ([6, 10], 0, 15), "push": ([6, 10], 2, 25)}


def make_take(rng: np.random.Generator, events: int = 12) -> tuple[np.ndarray, list[str]]:
    clips, kinds = [], []
    for _ in range(events):
        clips.append(np.repeat(BASE[None], int(rng.integers(15, 25)), 0))
        kind = str(rng.choice(list(KINDS))); joints, axis, amp = KINDS[kind]; n = int(rng.integers(32, 44))
        clip = np.repeat(BASE[None], n, 0).copy()
        clip[:, joints, axis] += amp * ((1 - np.cos(np.linspace(0, 2 * np.pi, n))) / 2)[:, None]
        clips.append(clip); kinds.append(kind)
    motion = np.concatenate(clips)
    return motion + rng.normal(0, .3, motion.shape).astype(np.float32), kinds


def make_text_encoder(path: Path, texts: list[str]) -> None:
    """Local 384-D SentenceTransformer fixture standing in for the downloadable Sentence-BERT weights."""
    torch.manual_seed(7)
    vocab = sorted({token for text in texts for token in text.lower().split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 384), Normalize()]).save_pretrained(str(path))


def run_cli(*args: object) -> str:
    return subprocess.run([sys.executable, "-m", "wild_pose_matching.cli", *map(str, args)], check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    parser.add_argument("--max-steps", type=int, default=40, help="training budget for this plumbing check")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    for take in ("take_a", "take_b"):
        motion, _ = make_take(rng)
        np.savez(out / f"{take}.npz", motion=motion, take=np.asarray(take))
    # Held-out "wild" take: projected, noisy 3 s windows with text.
    wild_motion, kinds = make_take(rng, 8)
    words = {"raise_left": "raise your left hand high", "raise_right": "lift the right side up", "spread": "open wide to everyone", "push": "push it away from us"}
    windows = [wild_motion[s:s + 45, :, :2].reshape(45, -1) + rng.normal(0, .5, (45, 22)) for s in range(0, len(wild_motion) - 44, 45)]
    texts = [words[kinds[i % len(kinds)]] for i in range(len(windows))]
    np.savez(out / "wild.npz", pose2d=np.stack(windows).astype(np.float32), texts=np.asarray(texts))

    report = json.loads(run_cli("extract-units", "--motion", out / "take_a.npz", out / "take_b.npz", "--variance-percentile", "40", "--output", out / "units.npz"))
    train = json.loads(run_cli("train", "--pairs", out / "units.npz", "--output", out / "gestureclr.pt", "--max-steps", args.max_steps, "--batch-size", 8, "--seed", 7))
    run_cli("cluster", "--units", out / "units.npz", "--checkpoint", out / "gestureclr.pt", "--clusters", 4, "--output", out / "clusters.npz")
    query = "raise your left hand high and then push it away from us"
    make_text_encoder(out / "tiny-sbert", texts + [query])
    mined = json.loads(run_cli("mine", "--wild", out / "wild.npz", "--units", out / "units.npz", "--checkpoint", out / "gestureclr.pt",
                               "--clusters", out / "clusters.npz", "--sbert", out / "tiny-sbert", "--output", out / "rules.jsonl"))
    run_cli("retrieve", "--rules", out / "rules.jsonl", "--clusters", out / "clusters.npz", "--sbert", out / "tiny-sbert", "--text", query,
            "--audio-seconds", 4.0, "--min-similarity", 0.2, "--seed", 7, "--output", out / "sequence.json")
    sequence = json.loads((out / "sequence.json").read_text(encoding="utf-8"))
    units = np.load(out / "units.npz")
    if not report["units"] or not mined["rules"] or not sequence:
        raise RuntimeError("CLI route produced empty output")
    if not all(30 <= n <= 45 for n in units["lengths"]) or abs(sum(s["duration_seconds"] for s in sequence) - 4.0) > 1e-6:
        raise RuntimeError("unit lengths or slot timing violate the contract")
    print(json.dumps({"units": report["units"], "variance_threshold": report["variance_threshold"], "train": train,
                      "rules": mined["rules"], "slots": [[s["text"], s["gesture_id"], round(s["duration_seconds"], 2)] for s in sequence], "output": str(out)}))


if __name__ == "__main__":
    main()
