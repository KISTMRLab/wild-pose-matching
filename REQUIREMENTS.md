# Requirements derived from the paper

## Paper facts

- GestureCLR learns correspondence between a clean 3D motion unit and a 2D projection of that same unit, augmented with Gaussian noise and temporal shifts to resemble wild OpenPose tracks.
- Units cover upper-body motion lasting 2–3 seconds. Wild pose/text records are three-second chunks.
- The model is SimCLR-inspired and uses contrastive learning. The later detailed paper describes a three-layer Transformer, five heads, 512 feed-forward width, and a 10D latent.
- Gesture units are clustered with Bisecting K-Means. At runtime, text is split into six-word chunks, embedded with Sentence-BERT, matched against the rule map, then a gesture is sampled from the matched cluster.

## Reimplementation decisions

- Both modalities use independent input projections and Transformer encoders, then masked temporal mean pooling and L2-normalized 10D output.
- Symmetric cross-entropy over the 2D-by-3D similarity matrix implements NT-Xent.
- All arrays use padded `[N,F,D]` tensors plus optional `[N,F]` masks. IDs and timed text use JSONL. The original counts are documentary facts, not shipped assets or promised outputs.
- The full CLI accepts prepared arrays in the documented common joint order. The small browser demo additionally converts one public BVH/TextGrid take into local paired windows.

## Acceptance criteria

Training must update a genuine contrastive model; mining must assign wild records through latent nearest-neighbor matching; clustering must be fitted on 3D-unit latents; retrieval must use six-word Sentence-BERT chunks and cluster sampling; tests cover augmentation, loss direction, and deterministic sampling utilities.


## Interactive data handoff

`scripts/start_demo.py` downloads one official BEAT BVH and aligned TextGrid, builds an ignored nine-clip bank with separate paired windows, and locally fits a compact dual-Transformer matcher and four clusters. The browser uses a TF-IDF text surrogate, while the full CLI above uses Sentence-BERT. Its trace shows chosen clips and retrieval route. This small same-speaker simulation is not evidence of general wild-pose matching quality or paper-scale coverage. The older `--example` server path remains an explicitly authored offline fixture. Speech is optional; public recordings and fitted weights are not bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
