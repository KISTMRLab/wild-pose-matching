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
- Skeleton projection and cleanup stay outside this repository; the README defines the required common joint order.

## Acceptance criteria

Training must update a genuine contrastive model; mining must assign wild records through latent nearest-neighbor matching; clustering must be fitted on 3D-unit latents; retrieval must use six-word Sentence-BERT chunks and cluster sampling; tests cover augmentation, loss direction, and deterministic sampling utilities.

