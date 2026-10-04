# Improving Co-speech gesture rule-map generation via wild pose matching with gesture units.

**Ghazanfar Ali, Jae-In Hwang**

**SIGGRAPH Asia Posters · 2022** · Published

[Paper / publisher](https://doi.org/10.1145/3550082.3564185) · [Project page](https://ghazanfarali.com/research/wild-pose-matching/) · [Video presentation](https://www.youtube.com/watch?v=QBtGdGE1Wgk) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> Contrastive matching turns noisy video poses into a richer gesture rule map.

![Method diagram from Figure 1 of the wild-pose-matching paper](paper-assets/method.png)

*Original method figure from the paper: Figure 1, PDF page 1. Extracted for this research introduction; the diagram describes the original system, not verification of this reimplementation.*

## Why this research

Poses estimated from everyday video are noisy and differ from clean 3D motion capture. A robust matching model can turn those videos into a richer source of gesture rules.

The poster aligns 2D poses from public monologue videos with gesture units extracted from 3D motion capture. GestureCLR learns robust matching; K-Means clusters the units to support variety when retrieving gestures at runtime.

## Method at a glance

**Text + noisy video pose** → **GestureCLR matching** → **Clustered gesture rules**

| | Research system |
|---|---|
| Input | Video-derived text and 2D pose; captured 3D motion |
| Method | Contrastive pose-to-unit matching and K-Means clustering |
| Output | Text-to-gesture rules and clustered gesture units |

## Evidence and scope

Poster demonstrates the expanded gesture library and mapping pipeline

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** 2,035 gesture units and 210,000 rules.

**Limitations:** This is a two-page poster; the later multilingual paper contains a separate user study and should be cited for that evidence.

## Explore the implementation

Trainable GestureCLR with noise/shift augmentation, clustered motion units, rule mining and semantic text retrieval. The public-data contract replaces the private unit library.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1145/3550082.3564185). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Watch the [existing YouTube presentation](https://www.youtube.com/watch?v=QBtGdGE1Wgk).

Please cite the research paper when using its ideas; [download the BibTeX citation](citation.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

Clean-room educational implementation of *Improving Co-speech gesture rule-map generation via wild pose matching with gesture units* (Ali and Hwang, SIGGRAPH Asia Posters 2022, DOI: [10.1145/3550082.3564185](https://doi.org/10.1145/3550082.3564185)). It implements the poster's learned noisy-2D/clean-3D matching, balanced gesture clustering, rule mining, and six-gram retrieval. It is independent of the institute implementation.

### Install and data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the `source` line.

Verify the complete local path with generated arrays:

```bash
python scripts/smoke.py
```

The command generates every documented NPZ contract and a local 384-D SentenceTransformer fixture, then invokes the installed `train`, `cluster`, `mine`, and `retrieve` CLI paths into `outputs/smoke/`. The local encoder replaces only the downloadable Sentence-BERT weights. Real runs swap that boundary for `all-MiniLM-L6-v2`; pose/motion shapes, checkpoints, clustering, and retrieval are identical.

Users prepare all data. [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M) is the public training source cited by the paper; follow its access terms and derive synchronized 2–3 second paired units. For wild records, use videos you may process and produce timestamped text plus 2D pose; the [TED Gesture Dataset](https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation) is a practical public replacement. No dataset, videos, motion, weights, or claimed 2,035-unit/210k-rule artifact is included.

`pairs.npz`: `pose2d[N,F,D2]`, `motion3d[N,F,D3]`. `units.npz`: `motion3d`, string `ids`, and scalar `dim2`. `wild.npz`: `pose2d`, string `texts`. Keep the same upper-body joint order, 15 FPS, root/neck centering, coordinate scale, padding, and masks across files. The compact CLI assumes fixed-length padded batches; remove invalid frames before packaging.

```bash
gestureclr train --pairs data/pairs.npz --output checkpoints/gestureclr.pt
gestureclr cluster --units data/units.npz --checkpoint checkpoints/gestureclr.pt --clusters 100 --output outputs/clusters.npz
gestureclr mine --wild data/wild.npz --units data/units.npz --checkpoint checkpoints/gestureclr.pt --clusters outputs/clusters.npz --output outputs/rules.jsonl
gestureclr retrieve --rules outputs/rules.jsonl --clusters outputs/clusters.npz --text "A sentence to animate in several chunks" --output outputs/sequence.json
python -m pytest
```

Training uses Gaussian noise levels sampled from `.001/.01/.1`, temporal displacement up to 15 frames, a three-layer Transformer pair, normalized 10D latents, symmetric NT-Xent, AdamW, and the paper learning rate/weight decay. Clustering uses Bisecting K-Means and retrieval randomly samples within the semantically matched cluster.

### Limits and licensing

This package begins with prepared pose arrays and does not perform face tracking, transcription, OpenPose inference, BVH conversion, retargeting, rendering, or temporal blending. Results depend heavily on projection and skeleton consistency. The two-page poster leaves architectural details underspecified; details documented in `REQUIREMENTS.md` are drawn from the later expanded methodology or declared implementation choices. Code is MIT licensed; datasets, Sentence-BERT weights, and source motion retain their own licenses.

### Citation

Machine-readable metadata is in [citation.bib](citation.bib).

```bibtex
@inproceedings{ali2022wild, title={Improving Co-speech gesture rule-map generation via wild pose matching with gesture units}, author={Ali, Ghazanfar and Hwang, Jae-In}, booktitle={SIGGRAPH Asia 2022 Posters}, year={2022}, doi={10.1145/3550082.3564185}}
```
