"""Train GestureCLR briefly and exercise clustering, mining, and retrieval."""
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

from wild_pose_matching.model import GestureCLR, ntxent
from wild_pose_matching.pipeline import build_rules, cluster_latents, retrieve, write_jsonl


def make_text_encoder(path: Path, texts: list[str]) -> None:
    torch.manual_seed(7)
    vocab = sorted({token for text in texts for token in text.lower().split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 384), Normalize()]).save_pretrained(str(path))


def run_cli(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "wild_pose_matching.cli", *map(str, args)], check=True)


def text_embedding(texts: list[str], width: int = 384) -> np.ndarray:
    """Deterministic offline stand-in at the documented Sentence-BERT boundary."""
    rows = []
    for text in texts:
        row = np.zeros(width, np.float32)
        for token in text.lower().split():
            row[sum(token.encode("utf-8")) % width] += 1
        row /= max(np.linalg.norm(row), 1e-8)
        rows.append(row)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)

    pose2d = rng.normal(size=(6, 45, 8)).astype("float32")
    motion3d = np.concatenate((pose2d, pose2d[..., :4] * 0.5), axis=-1)
    np.savez(out / "pairs.npz", pose2d=pose2d, motion3d=motion3d)
    model = GestureCLR(8, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    z2, z3 = model(torch.from_numpy(pose2d), torch.from_numpy(motion3d))
    loss = ntxent(z2, z3)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), out / "gestureclr.pt")

    model = GestureCLR(8, 12)
    model.load_state_dict(torch.load(out / "gestureclr.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        unit_latents = model.motion3d(torch.from_numpy(motion3d)).numpy()
        wild_latents = model.pose2d(torch.from_numpy(pose2d[:3])).numpy()
    ids = [f"unit_{i}" for i in range(len(motion3d))]
    labels, centers = cluster_latents(unit_latents, n_clusters=3, seed=7)
    np.savez(out / "clusters.npz", ids=np.asarray(ids), labels=labels, centroids=centers)
    texts = ["open both hands", "point to the chart", "explain the next idea"]
    embeddings = text_embedding(texts)
    rules = build_rules(embeddings, texts, wild_latents, unit_latents, ids, labels)
    write_jsonl(out / "rules.jsonl", rules)
    clusters = {int(k): [ids[i] for i in np.flatnonzero(labels == k)] for k in np.unique(labels)}
    sequence = retrieve("open both hands and explain the next idea", rules, text_embedding, clusters, seed=7)
    (out / "sequence.json").write_text(json.dumps(sequence, indent=2), encoding="utf-8")

    # Verify the installed CLI against the same public-data contracts.
    np.savez(out / "units.npz", motion3d=motion3d, ids=np.asarray(ids), dim2=np.asarray(8))
    np.savez(out / "wild.npz", pose2d=pose2d[:3], texts=np.asarray(texts))
    make_text_encoder(out / "tiny-sbert", texts + ["open both hands and explain the next idea"])
    run_cli("train", "--pairs", out / "pairs.npz", "--output", out / "cli-gestureclr.pt", "--epochs", 1, "--batch-size", 6, "--seed", 7)
    run_cli("cluster", "--units", out / "units.npz", "--checkpoint", out / "cli-gestureclr.pt", "--clusters", 3, "--output", out / "cli-clusters.npz")
    run_cli("mine", "--wild", out / "wild.npz", "--units", out / "units.npz", "--checkpoint", out / "cli-gestureclr.pt", "--clusters", out / "cli-clusters.npz", "--sbert", out / "tiny-sbert", "--output", out / "cli-rules.jsonl")
    run_cli("retrieve", "--rules", out / "cli-rules.jsonl", "--clusters", out / "cli-clusters.npz", "--sbert", out / "tiny-sbert", "--text", "open both hands and explain the next idea", "--seed", 7, "--output", out / "cli-sequence.json")
    cli_sequence = json.loads((out / "cli-sequence.json").read_text(encoding="utf-8"))
    if not cli_sequence: raise RuntimeError("installed CLI produced no gestures")
    print(json.dumps({"loss": float(loss.detach()), "rules": len(rules), "gestures": len(sequence), "output": str(out)}))


if __name__ == "__main__":
    main()
