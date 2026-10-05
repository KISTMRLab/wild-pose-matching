"""Prepared demo mode: serve GestureCLR rules mined from public BEAT to the browser viewer.

``scripts/demo_server.py --prepared outputs/paper-method/<key>`` loads the
artifacts written by ``prepare_paper_method.py`` and answers the viewer's
``/api/beat-library``, ``/api/beat-query`` and ``/api/query`` requests with
the repository's own six-word Sentence-BERT retrieval and cluster sampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import paper_method_common as pm

ALGORITHM = ("Wild Pose Matching: Algorithm 3 units from library speakers, GestureCLR 2D/3D contrastive matching, "
             "Bisecting K-Means clusters, rules mined from held-out projected+corrupted wild speakers, "
             "six-word Sentence-BERT retrieval with cluster sampling")
DATA_LABEL = "Public BEAT, disjoint speakers; projected BEAT motion stands in for wild video (local demo-scale fit)"


class PreparedDemo:
    def __init__(self, folder, args=None, encoder=None):
        pm.use_repository_package("wild_pose_matching")
        sbert = getattr(args, "sbert", None)
        from wild_pose_matching.pipeline import read_jsonl
        self.folder, self.manifest = pm.load_manifest(folder)
        files = self.manifest["files"]
        units = np.load(pm.resolve(files["units"]))
        self.motion = {str(i): m for i, m in zip(units["ids"], units["motion3d"])}
        self.lengths = {str(i): int(n) for i, n in zip(units["ids"], units["lengths"])}
        self.info = {u["id"]: u for u in json.loads((self.folder / files["unit_info"]).read_text(encoding="utf-8"))}
        self.rules = read_jsonl(self.folder / files["rules"])
        clusters = np.load(self.folder / files["clusters"])
        self.groups = {int(k): [str(x) for x in clusters["ids"][clusters["labels"] == k]] for k in np.unique(clusters["labels"])}
        firsts = np.stack([m.reshape(len(m), -1, 3)[0] for m in self.motion.values()])
        self.rest = np.median(firsts, axis=0)
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(str(sbert or self.manifest["sbert"]))
            encoder = lambda texts: model.encode(list(texts), normalize_embeddings=True)
        self.encode = encoder

    def library(self):
        clips = [{"id": gid, "text": row.get("text", ""), "duration": self.lengths[gid] / pm.FPS,
                  "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "role", "kind",
                                                     "motion_url", "alignment_url")}}
                 for gid, row in self.info.items()]
        suggested = list(self.manifest["suggested_queries"]) + list(self.manifest["heldout_probes"])
        if len(suggested) >= 3:
            suggested.append(". ".join(suggested[:3]))
        return {"ready": True, "prepared": True, "mode": "wild", "clips": clips, "suggested_queries": suggested,
                "heldout_probes": self.manifest["heldout_probes"], "metrics": self.manifest["metrics"],
                "roles": self.manifest["roles"]["roles"], "algorithm": ALGORITHM, "data_label": DATA_LABEL}

    def query(self, text, params):
        from wild_pose_matching.pipeline import retrieve
        seed = pm.param(params, "seed", self.manifest["seed"], int)
        floor = pm.param(params, "min_similarity", self.manifest["min_similarity"], float)
        sequence = retrieve(text, self.rules, self.encode, self.groups, seed, chunk_words=6, min_similarity=floor,
                            idle_id="idle")
        if not sequence:
            raise ValueError("Query text has no words")
        slots = []
        for entry in sequence:
            if entry["map"] == "idle":
                slots.append(pm.idle_slot(entry["text"], self.rest, "below similarity floor", floor=floor,
                                          similarity=round(entry["similarity"], 5), rule_text=entry.get("rule_text")))
                continue
            gid = entry["gesture_id"]
            row = self.info.get(gid, {})
            slots.append({"gesture_id": gid, "text": entry["text"],
                          "frames": pm.frames_m(self.motion[gid], self.lengths[gid]),
                          "route": "learned_pose_rule", "confidence": round(entry["similarity"], 5),
                          "similarity": round(entry["similarity"], 5), "cluster_id": entry["cluster_id"],
                          "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "kind",
                                                             "motion_url", "alignment_url")},
                          "rule_source": {"rule_text": entry.get("rule_text"), "cluster_id": entry["cluster_id"]},
                          "blend_frames": 5})
        metrics = {k: self.manifest["metrics"].get(k) for k in ("rules", "library_units", "clusters",
                                                                 "heldout_cross_view_top1", "heldout_chance")}
        metrics.update(rule_count=len(self.rules), min_similarity=floor,
                       learned_rule_usage=sum(s["route"] == "learned_pose_rule" for s in slots))
        return pm.query_result(slots, algorithm=ALGORITHM, data_label=DATA_LABEL, metrics=metrics,
                               trace={"input": text, "retrieval_text": text, "seed": seed},
                               joints=self.manifest.get("joints", pm.ingest().UPPER_BODY),
                               extra={"rule_count": len(self.rules), "cluster_count": len(self.groups), "seed": seed})
