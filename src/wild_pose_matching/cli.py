from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

DEFAULT_SBERT = "all-MiniLM-L6-v2"


def _floats(text):
    return tuple(float(v) for v in text.split(",") if v.strip())


def _norm_from_args(a):
    from .units import Normalization
    if a.no_normalize:
        return Normalization(enabled=False)
    if a.scale_joints == "none":
        return Normalization(a.neck_joint, None)
    left, right = (int(v) for v in a.scale_joints.split(","))
    return Normalization(a.neck_joint, (left, right))


def _add_norm_args(p):
    p.add_argument("--neck-joint", type=int, default=1, help="joint index used as the origin (default 1 = Neck)")
    p.add_argument("--scale-joints", default="4,8", help="'L,R' shoulder joint indices for scale (default 4,8 = LeftArm,RightArm), or 'none' for RMS scale")
    p.add_argument("--no-normalize", action="store_true", help="feed arrays unchanged (they must already be normalised)")


def _paths(groups, manifest):
    paths = [Path(p) for group in (groups or []) for p in group]
    if manifest:
        m = Path(manifest)
        entries = json.loads(m.read_text(encoding="utf-8")) if m.suffix.lower() == ".json" else [l.strip() for l in m.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        paths += [p if p.is_absolute() else m.parent / p for p in map(Path, entries)]
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(prog="gestureclr"); s=p.add_subparsers(dest="cmd",required=True)

    e=s.add_parser("extract-units", help="Algorithm 3 gesture units (2-3 s, minimal start-end distance, variance threshold) from 3D motion")
    e.add_argument("--motion", action="append", nargs="+", help="NPZ with motion[F,J,3] (and optional take, fps); repeatable")
    e.add_argument("--manifest", help="text or JSON list of motion NPZ paths")
    e.add_argument("--output", required=True)
    e.add_argument("--fps", type=float, default=15); e.add_argument("--min-seconds", type=float, default=2.0); e.add_argument("--max-seconds", type=float, default=3.0)
    e.add_argument("--variance-threshold", type=float, help="explicit scale-normalised variance cut-off")
    e.add_argument("--variance-percentile", type=float, default=25.0, help="otherwise use this percentile of window variances (default 25)")
    e.add_argument("--variance-elbow", action="store_true", help="otherwise use the elbow of the sorted window-variance curve")
    e.add_argument("--max-closure", type=float, help="optional cap on scale-normalised start-end distance")
    e.add_argument("--mode", choices=("algorithm3", "windows"), default="algorithm3", help="'windows' writes overlapping 2-3 s training sequences instead")
    e.add_argument("--stride", type=int, default=5, help="window stride for --mode windows")
    e.add_argument("--seed", type=int, default=0)
    _add_norm_args(e)

    t=s.add_parser("train", help="train GestureCLR (paper augmentation, cosine annealing, validation, best checkpoint)")
    t.add_argument("--pairs",required=True, help="NPZ with pose2d[N,F,D2], motion3d[N,F,D3], optional mask[N,F]"); t.add_argument("--output",required=True)
    t.add_argument("--preset", choices=("demo", "paper"), default="demo", help="demo: 300 epochs, batch 64; paper: 1000 epochs, batch 512")
    t.add_argument("--config", help="JSON file overriding preset fields")
    t.add_argument("--epochs",type=int); t.add_argument("--max-steps", type=int, help="step budget instead of epochs"); t.add_argument("--batch-size",type=int)
    t.add_argument("--lr", type=float); t.add_argument("--weight-decay", type=float); t.add_argument("--temperature", type=float); t.add_argument("--latent-dim", type=int)
    t.add_argument("--noise-variances", type=_floats, help="comma list, paper: 0.001,0.01,0.1"); t.add_argument("--shift-prob", type=float, help="probability of the 30-in-45 temporal shift")
    t.add_argument("--crop-frames", type=int); t.add_argument("--max-offset", type=int); t.add_argument("--fills", type=lambda v: tuple(v.split(",")), help="mean,zero")
    t.add_argument("--val-fraction", type=float); t.add_argument("--seed",type=int)
    _add_norm_args(t)

    c=s.add_parser("cluster"); c.add_argument("--units",required=True); c.add_argument("--checkpoint",required=True); c.add_argument("--output",required=True); c.add_argument("--clusters",type=int,default=100); c.add_argument("--seed", type=int, default=0)
    m=s.add_parser("mine"); m.add_argument("--wild",required=True); m.add_argument("--units",required=True); m.add_argument("--checkpoint",required=True); m.add_argument("--clusters",required=True); m.add_argument("--output",required=True)
    m.add_argument("--sbert",default=DEFAULT_SBERT, help="Sentence-BERT model name or local directory"); m.add_argument("--sbert-local-only", action="store_true", help="never download; fail if not cached")
    m.add_argument("--min-pose-match", type=float, help="drop wild records whose best unit cosine is below this")
    r=s.add_parser("retrieve"); r.add_argument("--rules",required=True); r.add_argument("--clusters",required=True); r.add_argument("--text",required=True); r.add_argument("--output",required=True); r.add_argument("--seed",type=int,default=0)
    r.add_argument("--sbert",default=DEFAULT_SBERT); r.add_argument("--sbert-local-only", action="store_true")
    r.add_argument("--audio-seconds", type=float, help="speech length; slots are proportional to chunk word counts"); r.add_argument("--chunk-words", type=int, default=6)
    r.add_argument("--min-similarity", type=float, help="text similarity floor; below it the slot is idle"); r.add_argument("--idle-id", default="idle")
    s.add_parser("presets", help="print the training presets")
    a=p.parse_args(argv)
    {"extract-units": extract_cmd, "train": train, "cluster": cluster, "mine": mine, "retrieve": retrieve_cmd, "presets": presets_cmd}[a.cmd](a)


def presets_cmd(a):
    from dataclasses import asdict
    from .training import PRESETS
    print(json.dumps({k: asdict(v) for k, v in PRESETS.items()}, indent=2))


def extract_cmd(a):
    from .units import dense_windows, extract_units, pack_units, suggest_variance_threshold, window_variances
    paths = _paths(a.motion, a.manifest)
    if not paths: raise SystemExit("give --motion NPZ files or --manifest")
    norm = _norm_from_args(a); motions = {}
    for path in paths:
        d = np.load(path, allow_pickle=False)
        x = d["motion"] if "motion" in d.files else d["motion3d"]
        x = x.reshape(len(x), -1, 3)
        take = str(d["take"]) if "take" in d.files else path.stem
        if take in motions: raise SystemExit(f"duplicate take id {take!r}")
        motions[take] = x
    lo, hi = round(a.min_seconds * a.fps), round(a.max_seconds * a.fps)
    report = {"mode": a.mode, "takes": len(motions), "fps": a.fps, "frames": [lo, hi], "normalization": norm.to_dict()}
    if a.mode == "windows":
        spans = {take: dense_windows(len(x), a.fps, a.min_seconds, a.max_seconds, a.stride, a.seed + i) for i, (take, x) in enumerate(motions.items())}
    else:
        if a.variance_threshold is not None:
            vthr, how = a.variance_threshold, "explicit"
        else:
            v = window_variances(motions.values(), 3, (lo + hi) // 2, norm)
            vthr = suggest_variance_threshold(v, "elbow" if a.variance_elbow else "percentile", a.variance_percentile)
            how = "elbow" if a.variance_elbow else f"percentile {a.variance_percentile:g}"
            report["window_variance_percentiles"] = {f"p{q}": float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)}
        report.update(variance_threshold=float(vthr), variance_rule=how)
        spans = {take: extract_units(x, a.fps, a.min_seconds, a.max_seconds, vthr, a.max_closure, 3, norm) for take, x in motions.items()}
    packed = pack_units(motions, spans, hi)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **packed, fps=np.asarray(a.fps), normalization_json=np.asarray(json.dumps(norm.to_dict())))
    lengths = packed["lengths"]
    covered = 0
    for take, take_spans in spans.items():
        used = np.zeros(len(motions[take]), bool)
        for s0, e0 in take_spans: used[s0:e0] = True
        covered += int(used.sum())
    report.update(units=int(len(lengths)), per_take={k: len(v) for k, v in spans.items()},
                  coverage=covered / sum(len(x) for x in motions.values()),
                  length_histogram={str(k): int(v) for k, v in sorted(Counter(lengths.tolist()).items())}, output=str(out))
    Path(f"{out}.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("units", "takes", "coverage") if k in report} | ({"variance_threshold": report["variance_threshold"]} if "variance_threshold" in report else {})))


def _mask(d, key="mask"):
    return d[key].astype(bool) if key in d.files else None


def train(a):
    from .training import config_from, save_checkpoint, train_gestureclr
    cfg=config_from(a.preset, a.config, epochs=a.epochs, max_steps=a.max_steps, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
                    temperature=a.temperature, latent_dim=a.latent_dim, noise_variances=a.noise_variances, shift_prob=a.shift_prob,
                    crop_frames=a.crop_frames, max_offset=a.max_offset, fills=a.fills, val_fraction=a.val_fraction, seed=a.seed)
    norm=_norm_from_args(a)
    data=np.load(a.pairs); mask=_mask(data)
    x2=norm.apply(data["pose2d"],2,mask); x3=norm.apply(data["motion3d"],3,mask)
    model,history=train_gestureclr(x2,x3,mask,cfg,log=lambda line: print(line, file=sys.stderr))
    save_checkpoint(a.output,model,x2.shape[-1],x3.shape[-1],cfg,norm,history)
    Path(f"{a.output}.history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    summary={k:history.get(k) for k in ("best_epoch","best_monitor","monitor","train_top1_eval","val_top1_best","train_pairs","val_pairs")}
    print(json.dumps(summary))


def _units_latents(model, norm, u):
    from .training import encode
    mask=_mask(u); x=norm.apply(u["motion3d"].astype("float32"),3,mask)
    return encode(model.motion3d, x, mask)


def cluster(a):
    from .pipeline import cluster_latents
    from .training import load_checkpoint
    d=np.load(a.units); x=d["motion3d"]; ids=[str(v) for v in d["ids"]]
    model,norm=load_checkpoint(a.checkpoint,int(d["dim2"]),x.shape[-1])
    z=_units_latents(model,norm,d)
    labels,centers=cluster_latents(z,a.clusters,a.seed); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    lengths=d["lengths"] if "lengths" in d.files else np.full(len(ids),x.shape[1])
    fps=d["fps"] if "fps" in d.files else np.asarray(15.0)
    np.savez(output,ids=np.asarray(ids),labels=labels,centroids=centers,latents=z,lengths=lengths,fps=fps)
    print(json.dumps({"units":len(ids),"clusters":int(labels.max()+1),"sizes":np.bincount(labels).tolist()}))


def load_sentence_encoder(name, local_only=False):
    """Sentence-BERT by hub name (downloaded once into the Hugging Face cache) or by local directory."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(str(name), local_files_only=local_only)
    except Exception as exc:  # network or cache miss: explain the acquisition step
        raise SystemExit(f"could not load Sentence-BERT {name!r} ({exc.__class__.__name__}: {exc}). Download it once with "
                         f"`python -c \"from sentence_transformers import SentenceTransformer as S; S('{DEFAULT_SBERT}').save('models/{DEFAULT_SBERT}')\"` "
                         f"and pass --sbert models/{DEFAULT_SBERT}") from exc


def mine(a):
    from .pipeline import build_rules,write_jsonl
    from .training import encode, load_checkpoint
    w=np.load(a.wild); u=np.load(a.units); cl=np.load(a.clusters)
    model,norm=load_checkpoint(a.checkpoint,w["pose2d"].shape[-1],u["motion3d"].shape[-1])
    wm=_mask(w); wz=encode(model.pose2d, norm.apply(w["pose2d"].astype("float32"),2,wm), wm)
    uz=_units_latents(model,norm,u)
    texts=[str(x) for x in w["texts"]]; emb=load_sentence_encoder(a.sbert,a.sbert_local_only).encode(texts,normalize_embeddings=True)
    rules=build_rules(emb,texts,wz,uz,[str(x) for x in u["ids"]],cl["labels"],a.min_pose_match)
    write_jsonl(a.output,rules)
    print(json.dumps({"wild_records":len(texts),"rules":len(rules),"distinct_units":len({r["gesture_id"] for r in rules})}))


def retrieve_cmd(a):
    from .pipeline import read_jsonl,retrieve
    rules=read_jsonl(a.rules); d=np.load(a.clusters); clusters={int(k):[str(x) for x in d["ids"][d["labels"]==k]] for k in np.unique(d["labels"])}
    unit_seconds=None
    if "lengths" in d.files:
        fps=float(d["fps"]) if "fps" in d.files else 15.0
        unit_seconds={str(i):float(n)/fps for i,n in zip(d["ids"],d["lengths"])}
    model=load_sentence_encoder(a.sbert,a.sbert_local_only)
    result=retrieve(a.text,rules,lambda x:model.encode(x,normalize_embeddings=True),clusters,a.seed,chunk_words=a.chunk_words,
                    min_similarity=a.min_similarity,idle_id=a.idle_id,audio_seconds=a.audio_seconds,unit_seconds=unit_seconds)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8")

if __name__=="__main__": main()
