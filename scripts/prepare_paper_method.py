"""Prepare the Wild Pose Matching paper method on public BEAT for the browser demo.

Launcher hook (see scripts/beat_demo/BEAT_INGEST.md): progress goes to stderr
and the last stdout line is ``{"ready": ..., "server_args": [...]}``.

Disjoint BEAT speakers take the paper's three roles:

* ``library`` speakers: continuous 3D motion -> Algorithm 3 gesture units
  (``gestureclr extract-units``), clustered with Bisecting K-Means;
* ``train`` speakers: paired 3 s windows, clean 3D plus a random-yaw 2D
  projection -> GestureCLR training (``gestureclr train``);
* ``wild`` speakers: held-out 3 s windows projected through an explicit camera
  (yaw 20, pitch 5) and corrupted like OpenPose tracks, with their transcripts
  -> latent nearest-unit rules (``gestureclr mine``).

Results are cached under ``outputs/paper-method/<settings hash>/``. Projected
BEAT motion stands in for the paper's wild video; it is not the paper data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_method_common as pm  # noqa: E402

PACKAGE = "wild_pose_matching"
DEFAULT_SPEAKERS = "1,2,3,4,5,6"
DEFAULT_TAKES = 2


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pm.add_source_args(p, DEFAULT_SPEAKERS, DEFAULT_TAKES, "library, train, wild")
    p.add_argument("--sbert", help=f"Sentence-BERT directory (default models/{pm.SBERT_NAME}; env {pm.ENV_SBERT})")
    p.add_argument("--preset", choices=("demo", "paper"), default="demo",
                   help="demo: 300 epochs at batch 64, ~N/4 clusters; paper: 1000 epochs at batch 512, 100 clusters")
    p.add_argument("--epochs", type=int); p.add_argument("--max-steps", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--clusters", type=int, help="Bisecting K-Means clusters (default: preset)")
    p.add_argument("--variance-percentile", type=float, default=25.0, help="Algorithm 3 variance cut-off percentile")
    p.add_argument("--yaw", type=float, default=20.0); p.add_argument("--pitch", type=float, default=5.0)
    p.add_argument("--noise", type=float, default=0.02); p.add_argument("--jitter", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--min-similarity", type=float, default=0.2, help="demo text-similarity floor; below it a chunk idles")
    return p.parse_args(argv)


def _settings(args, source, kind, descriptors, sbert):
    return {"repo": PACKAGE, "source": str(source), "kind": kind, "selection": pm.selection(args),
            "takes": [d["take"] for d in descriptors], "role": args.role, "role_unit": args.role_unit,
            "seed": args.seed, "max_frames": args.max_frames, "preset": args.preset, "epochs": args.epochs,
            "max_steps": args.max_steps, "batch_size": args.batch_size, "clusters": args.clusters,
            "variance_percentile": args.variance_percentile, "sbert": sbert,
            "camera": [args.yaw, args.pitch], "corruption": [args.noise, args.jitter, args.dropout],
            "min_similarity": args.min_similarity}


def heldout_cross_view(checkpoint, wild_npz, wild_windows):
    """Wild 2D window -> its own 3D motion among all held-out wild windows (top-1 and chance)."""
    from wild_pose_matching.training import encode, load_checkpoint, top1
    wild = np.load(wild_npz)
    by_id = {w["id"]: w for w in wild_windows}
    keep = [i for i, gid in enumerate(wild["ids"]) if str(gid) in by_id]
    if len(keep) < 2:
        return {}
    motion = np.stack([by_id[str(wild["ids"][i])]["positions"].reshape(len(wild["pose2d"][i]), -1) for i in keep])
    model, norm = load_checkpoint(checkpoint)
    z2 = encode(model.pose2d, norm.apply(wild["pose2d"][keep].astype("float32"), 2))
    z3 = encode(model.motion3d, norm.apply(motion.astype("float32"), 3))
    return {"heldout_cross_view_top1": round(top1(z2, z3), 4), "heldout_chance": round(1 / len(keep), 4),
            "heldout_windows": len(keep),
            "heldout": "held-out wild speakers: corrupted yaw-20 2D window -> its own 3D window (never trained on)"}


def suggestions(rules, unit_texts, limit=6):
    """Rule phrases from distinct clusters (they route through learned pose rules) plus held-out probes."""
    chosen, clusters = [], set()
    for rule in sorted(rules, key=lambda r: -r["pose_match"]):
        words = rule["text"].split()
        if rule["cluster_id"] in clusters or not 4 <= len(words) <= 12:
            continue
        chosen.append(" ".join(words[:6]))
        clusters.add(rule["cluster_id"])
        if len(chosen) == limit:
            break
    probes = [t for t in unit_texts if len(t.split()) >= 4][:2]
    return chosen, [" ".join(t.split()[:6]) for t in probes]


def prepare(args):
    source, kind = pm.find_source(args.processed, args.beat_root)
    if source is None:
        return pm.not_ready("no local BEAT source found", pm.SOURCE_STEPS)
    sbert, why = pm.find_sbert(args.sbert)
    if sbert is None:
        return pm.not_ready(why, [pm.SBERT_STEP])
    beat = pm.ingest()
    select = pm.selection(args)
    descriptors = beat.list_takes(source, kind=kind, **select)
    if not descriptors:
        return pm.not_ready(f"no BEAT takes in {source} match the selection", pm.SOURCE_STEPS)
    code = [pm.ROOT / "src" / PACKAGE, Path(__file__), Path(pm.__file__), pm.SCRIPTS / "beat_demo" / "beat_ingest.py"]
    folder = Path(args.output_root) / pm.cache_key(_settings(args, source, kind, descriptors, sbert), code)
    if not args.force and pm.cached(folder):
        pm.progress(f"cached result {pm.portable(folder)}")
        return pm.ready(folder, pm.cached(folder), cached_result=True)
    if folder.exists():
        shutil.rmtree(folder)
    data = folder / "data"
    timings, started = {}, time.perf_counter()
    pm.use_repository_package(PACKAGE)
    from wild_pose_matching import cli

    pm.progress(f"exporting {len(descriptors)} takes from {source} ({kind}) with disjoint speaker roles")
    t0 = time.perf_counter()
    summary = beat.export_wild(source, data, fps=pm.FPS, unit_seconds=pm.UNIT_SECONDS, units="cm", seed=args.seed,
                               camera={"yaw": args.yaw, "pitch": args.pitch, "focal": 1.0, "distance": 3.0},
                               corruption={"noise": args.noise, "jitter": args.jitter, "dropout": args.dropout},
                               kind=kind, role_spec=pm.role_spec(args), role_unit=args.role_unit,
                               max_frames=args.max_frames, **select)
    assignment = json.loads((data / "joint_order.json").read_text(encoding="utf-8"))["roles"]
    roles = pm.descriptor_roles(assignment)
    library = pm.load_takes([d for d in descriptors if roles.get(d["take"]) == "library"], args.max_frames)
    wild_records = pm.load_takes([d for d in descriptors if roles.get(d["take"]) == "wild"], args.max_frames)
    takes_dir = data / "library-takes"; takes_dir.mkdir(parents=True, exist_ok=True)
    paths, words, info = [], {}, {}
    for record in library:
        path = takes_dir / f"{record['take']}.npz"
        np.savez(path, motion=record["positions"], take=np.asarray(record["take"]), fps=np.asarray(pm.FPS))
        paths.append(path); words[record["take"]] = record["words"]; info[record["take"]] = record
    timings["export_seconds"] = round(time.perf_counter() - t0, 2)

    units = data / "units.npz"
    extract, timings["extract_units_seconds"] = pm.run_cli(cli.main, ["extract-units", "--motion", *paths, "--output", units,
                                                                      "--fps", pm.FPS, "--variance-percentile", args.variance_percentile,
                                                                      "--seed", args.seed], "Algorithm 3 units")
    u = np.load(units)
    unit_rows = []
    for gid, take, s, e in zip(u["ids"], u["takes"], u["starts"], u["ends"]):
        record = info[str(take)]
        unit_rows.append({"id": str(gid), "text": pm.span_text(words[str(take)], int(s), int(e)),
                          "speaker": record["speaker"], "take": str(take), "start_frame": int(s), "end_frame": int(e),
                          "role": "library", **pm.source_info(record["source"])})
    (folder / "units.json").write_text(json.dumps(unit_rows, ensure_ascii=False), encoding="utf-8")

    checkpoint = folder / "gestureclr.pt"
    train = ["train", "--pairs", data / "pairs.npz", "--output", checkpoint, "--preset", args.preset, "--seed", args.seed]
    for flag, value in (("--epochs", args.epochs), ("--max-steps", args.max_steps), ("--batch-size", args.batch_size)):
        if value is not None:
            train += [flag, value]
    trained, timings["train_seconds"] = pm.run_cli(cli.main, train, "GestureCLR training")
    history = json.loads(Path(f"{checkpoint}.history.json").read_text(encoding="utf-8"))

    count = args.clusters or (100 if args.preset == "paper" else max(4, round(len(unit_rows) / 4)))
    clusters = folder / "clusters.npz"
    clustered, timings["cluster_seconds"] = pm.run_cli(cli.main, ["cluster", "--units", units, "--checkpoint", checkpoint,
                                                                  "--output", clusters, "--clusters", count,
                                                                  "--seed", args.seed], "Bisecting K-Means")
    rules_path = folder / "rules.jsonl"
    mined, timings["mine_seconds"] = pm.run_cli(cli.main, ["mine", "--wild", data / "wild.npz", "--units", units,
                                                           "--checkpoint", checkpoint, "--clusters", clusters,
                                                           "--output", rules_path, "--sbert", sbert, "--sbert-local-only"],
                                                "wild rule mining")
    from wild_pose_matching.pipeline import read_jsonl
    rules = read_jsonl(rules_path)
    heldout = heldout_cross_view(checkpoint, data / "wild.npz", pm.take_windows(wild_records))
    labels = np.load(clusters)["labels"]
    used = {r["gesture_id"] for r in rules}
    suggested, probes = suggestions(rules, [r["text"] for r in unit_rows])
    timings["total_seconds"] = round(time.perf_counter() - started, 2)
    metrics = {"library_units": len(unit_rows), "train_pairs": int(history.get("train_pairs", 0)),
               "val_pairs": int(history.get("val_pairs", 0)), "wild_windows": summary["wild_windows"],
               "rules": len(rules), "clusters": int(labels.max() + 1),
               "cluster_sizes": np.bincount(labels).tolist(), "rule_units_used": len(used),
               "rule_unit_share": round(len(used) / max(1, len(unit_rows)), 4),
               "mean_pose_match": round(float(np.mean([r["pose_match"] for r in rules])), 4) if rules else None,
               "train_top1_eval": history.get("train_top1_eval"), "val_top1_best": history.get("val_top1_best"),
               "best_epoch": history.get("best_epoch"), **heldout}
    manifest = {
        "schema": "paperreach.paper-method.v1", "repo": "wild-pose-matching", "mode": "wild",
        "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"path": str(source), "kind": kind, "takes": len(descriptors)},
        "roles": assignment, "preset": args.preset, "sbert": sbert, "seed": args.seed,
        "min_similarity": args.min_similarity, "fps": pm.FPS, "units": "cm (neck-centred)",
        "files": {"units": pm.portable(units), "unit_info": "units.json", "rules": "rules.jsonl",
                  "clusters": "clusters.npz", "checkpoint": "gestureclr.pt", "pairs": pm.portable(data / "pairs.npz"),
                  "wild": pm.portable(data / "wild.npz")},
        "export": summary, "extract": extract, "train": trained, "cluster": clustered, "mine": mined,
        "metrics": metrics, "timings": timings, "suggested_queries": suggested, "heldout_probes": probes,
        "summary": {"rules": len(rules), "units": len(unit_rows), "clusters": metrics["clusters"],
                    "heldout_cross_view_top1": heldout.get("heldout_cross_view_top1"),
                    "heldout_chance": heldout.get("heldout_chance"),
                    "data": f"BEAT {kind} speakers {', '.join(sorted({d['speaker'] for d in descriptors}, key=lambda s: (len(s), s)))} "
                            f"via beat_ingest export-wild ({args.preset} preset)",
                    "seconds": timings["total_seconds"]},
    }
    pm.write_manifest(folder, manifest)
    pm.progress(f"done in {timings['total_seconds']} s: {len(rules)} rules over {len(unit_rows)} units; "
                f"held-out cross-view top-1 {heldout.get('heldout_cross_view_top1')} (chance {heldout.get('heldout_chance')})")
    return pm.ready(folder, manifest)


def main(argv=None):
    args = parse(argv)
    try:
        return prepare(args)
    except (Exception, SystemExit) as error:  # the launcher falls back to the default demo
        if isinstance(error, SystemExit) and error.code in (0, None):
            raise
        pm.progress(f"failed: {error.__class__.__name__}: {error}")
        return pm.emit({"ready": False, "reason": f"{error.__class__.__name__}: {error}"}, 1)


if __name__ == "__main__":
    raise SystemExit(main())
