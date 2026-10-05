# Requirements derived from the paper

## Paper facts

- GestureCLR learns correspondence between a clean 3D motion unit and a 2D projection of that same unit, augmented with Gaussian noise and temporal shifts to resemble wild OpenPose tracks.
- Units cover upper-body motion lasting 2–3 seconds. Wild pose/text records are three-second chunks.
- The model is SimCLR-inspired and uses contrastive learning. The later detailed paper describes a three-layer Transformer, five heads, 512 feed-forward width, and a 10D latent.
- Gesture units are clustered with Bisecting K-Means. At runtime, text is split into six-word chunks, embedded with Sentence-BERT, matched against the rule map, then a gesture is sampled from the matched cluster.

## Reimplementation decisions

- **Gesture units (Algorithm 3, thesis section 5.2.1).** Units are 30–45 frames, with minimal start–end distance and a variance threshold. They are implemented as a global greedy search over the still-unused motion; the paper's "remove the longest clip" step is read as removing each extracted clip. The expert-dependent variance is chosen by an explicit value, a percentile or an elbow helper, and is recorded in a report. Distances and variances use neck-centred, shoulder-width-scaled poses.
- **Normalisation.** Training, clustering and mining apply the same neck-centring and shoulder-width scaling, stored in the checkpoint.
- **Noise.** Gaussian noise uses the paper's variances 0.001, 0.01 and 0.1, with standard deviation √variance on the scale-normalised poses.
- **Temporal shift.** This follows the thesis: a random 30-frame crop at offset 1–15 in a 45-frame buffer filled with the mean pose or zeros. It is applied with a configurable probability (default 0.5).
- **Encoders.** Both modalities use independent input projections and Transformer encoders, then masked temporal mean pooling and L2-normalized 10D output. Padding masks pass through training, clustering and mining.
- **Loss.** Symmetric cross-entropy over the 2D-by-3D similarity matrix implements NT-Xent.
- **Training.** AdamW with per-step cosine annealing, a validation split, best checkpoint by validation loss, and per-epoch validation top-1. The paper preset is 1000 epochs at batch 512. The demo preset (300 epochs, batch 64) is sized to converge on a few hundred sequences. The paper does not state the temperature; 0.07 is a declared default with a flag.
- **Arrays.** All arrays use padded `[N,F,D]` tensors plus optional `[N,F]` masks and `lengths`. IDs and timed text use JSONL. The original counts are documentary facts, not shipped assets or promised outputs.
- **Retrieval.** It reports per-slot timing from the speech length, and plays an optional idle ID below a similarity floor.
- **Data.** The full CLI accepts prepared arrays in the documented common joint order. The small browser demo additionally converts one public BVH/TextGrid take into local paired windows.

## Acceptance criteria

- Unit extraction must follow Algorithm 3.
- Training must update a genuine contrastive model with the paper's augmentation.
- Mining must assign wild records through latent nearest-neighbor matching.
- Clustering must be fitted on 3D-unit latents.
- Retrieval must use six-word Sentence-BERT chunks and cluster sampling.
- Tests cover:
  - unit extraction and normalisation;
  - noise variance and the temporal shift;
  - the NT-Xent value against the formula, and its direction: aligned pairs score lower, and a gradient step moves latents toward their partners;
  - masks;
  - schedule, validation and checkpoint reload;
  - retrieval timing and idle handling.


## Interactive data handoff

`scripts/start_demo.py` downloads one official BEAT BVH and aligned TextGrid, builds an ignored nine-clip bank with separate paired windows, and locally fits a compact dual-Transformer matcher and four clusters. The browser uses a TF-IDF text surrogate, while the full CLI above uses Sentence-BERT. Its trace shows chosen clips and retrieval route. This small same-speaker simulation is not evidence of general wild-pose matching quality or paper-scale coverage. The older `--example` server path remains an explicitly authored offline fixture. Speech is optional; public recordings and fitted weights are not bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; mouth shapes follow a rule-based text-to-phoneme-to-viseme track timed to speech playback, an approximation rather than forced phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
