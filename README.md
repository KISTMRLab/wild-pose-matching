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

<!-- demo-preview:start -->
## Demo preview

![Wild Pose Matching runnable demo](demo-assets/preview.png)

*Local demo with small starter examples; the capture illustrates the interface, not a reproduced paper benchmark.*

From the repository root, using the Python environment described below:

```sh
python -m pip install -e .
python scripts/start_demo.py
```

Open **http://127.0.0.1:8080/**. A starter query and motion clip load automatically. Click **Play**, or change the text and click **Retrieve motion**. The launcher selects the bundled inputs automatically; it also builds the small authored index for RAG demos. Avatar demos prepare their pinned Three.js modules on first launch, so that step needs internet access. Model weights and public datasets are optional for the starter workflow and are prepared separately for real-data use.

The 3D presentation uses shared Three.js avatar components and bundled fictional CC0 characters. The paper-specific algorithms and data adapters live in this repository.

<!-- demo-preview:end -->

## Implementation and usage

<!-- implementation-guide -->

Clean-room educational implementation of *Improving Co-speech gesture rule-map generation via wild pose matching with gesture units* (Ali and Hwang, SIGGRAPH Asia Posters 2022, DOI: [10.1145/3550082.3564185](https://doi.org/10.1145/3550082.3564185)). It implements the poster's learned noisy-2D/clean-3D matching, balanced gesture clustering, rule mining, and six-gram retrieval. It is independent of the institute implementation.

For an immediate browser example after installation, run `python scripts/prepare_viewer.py --out static/vendor` and `python scripts/demo_server.py --example`, then open the printed URL. Author-created motion and transparent illustrative pose/text vectors exercise the actual cluster and retrieval functions. The UI labels them as examples; no GestureCLR model is claimed to have been trained. The prepared-data commands below use an actual trained checkpoint.

```bash
python -m pip install -e .
python scripts/prepare_viewer.py --out static/vendor
python scripts/demo_server.py --example
```

### Install and data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the `source` line.

Verify the complete local path with generated arrays:

```bash
python scripts/verify.py
```

The command generates every documented NPZ contract and a local 384-D SentenceTransformer fixture, then invokes the installed `train`, `cluster`, `mine`, and `retrieve` CLI paths into `outputs/verification/`. The local encoder replaces only the downloadable Sentence-BERT weights. Real runs swap that boundary for `all-MiniLM-L6-v2`; pose/motion shapes, checkpoints, clustering, and retrieval are identical.

Users prepare all data. [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M) is the public training source cited by the paper; follow its access terms and derive synchronized 2–3 second paired units. For wild records, use videos you may process and produce timestamped text plus 2D pose; the [TED Gesture Dataset](https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation) is a practical public replacement. No dataset, videos, motion, weights, or claimed 2,035-unit/210k-rule artifact is included.

### Prepare data and launch the motion demo

`scripts/prepare_public_data.py` accepts a licensed BVH and word-aligned JSONL transcript. Supply either one record with `words` or one word per line, using `word`, `start_seconds`, and `end_seconds`. It applies BVH hierarchy rotations, resamples to 15 FPS, centers on the neck, and creates paired 2D/3D units plus a projected `wild.npz` proxy. Use a recording of at least six seconds for two training pairs. The proxy demonstrates the interface; replace `wild.npz` with actual aligned video-estimated 2D poses and text for wild-pose mining. Retarget other BVH skeletons to the joint names in the script.

```bash
python scripts/prepare_public_data.py --bvh data/licensed_motion.bvh --transcript data/words.jsonl --output-dir data/prepared
gestureclr train --pairs data/prepared/pairs.npz --epochs 20 --output checkpoints/gestureclr.pt
gestureclr cluster --units data/prepared/units.npz --checkpoint checkpoints/gestureclr.pt --clusters 10 --output outputs/clusters.npz
gestureclr mine --wild data/prepared/wild.npz --units data/prepared/units.npz --checkpoint checkpoints/gestureclr.pt --clusters outputs/clusters.npz --output outputs/rules.jsonl
python scripts/prepare_viewer.py --out static/vendor
python scripts/demo_server.py --data-dir data/prepared --rules outputs/rules.jsonl --clusters outputs/clusters.npz
```

Open the printed local URL. Queries use six-word chunks, Sentence-BERT rule similarity and seeded sampling from the learned cluster. The trace shows the chosen cluster, score and actual unit frames. `gestureclr retrieve` remains available for batch results; `scripts/export_playback.py --sequence outputs/sequence.json --motion data/prepared/units.npz --output outputs/playback.json` joins those IDs to frames. The checkpoint is fitted only to the supplied pairs; a few demo epochs or projected proxy data do not establish useful wild-video accuracy. `scripts/verify.py` exercises plumbing with random arrays and a local text-encoder fixture, never a trained public model.

The [automatic rule-mining precursor](https://github.com/ghazanPK/automatic-text-to-gesture) motivates harvesting mappings from video; [multilingual gesture retrieval](https://github.com/ghazanPK/multilingual-gesture) and [RIDGE](https://github.com/ghazanPK/ridge) develop the GestureCLR lineage. These are research references, not package dependencies.

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

### Optional local speech adapters

The viewer can speak its query or transcribe user-selected audio. Browser voice and typed text work without model weights. Install `python -m pip install -e ".[speech]"` for local adapters. Obtain Kokoro files from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) yourself: `config.json`, `kokoro-v1_0.pth` and `voices/af_heart.pt`. Set `KOKORO_MODEL_DIR` to their parent folder before launching the server. Follow [Kokoro's English phonemizer setup](https://github.com/hexgrad/kokoro), including espeak-ng where required, then choose Local Kokoro. For ASR, set `WHISPER_MODEL_DIR` to a user-downloaded [faster-whisper](https://github.com/SYSTRAN/faster-whisper) small model directory containing `model.bin` and its tokenizer/configuration files. ASR runs on CPU with INT8, requests word timestamps and VAD, and disables implicit model downloads. No speech model files or audio recordings are included in this repo.

<!-- avatar-recorded-motion:start -->
## Bundled characters and recorded public motion

The browser demos include Rowan and Mira, two new fictional GLB characters built with MPFB and MakeHuman community assets under CC0 1.0. See [avatar licensing and provenance](static/avatars/LICENSE.md). Use the character selector in the stage. The shared renderer supports body bones, ARKit facial channels, and approximate speaking motion.

Recorded motion is adapted to the characters' proportions. Palm landmarks set hand orientation; finger curl uses bounded hinge bends and preserves the character's finger spacing. Thumb-base opposition stays in the authored pose, with conservative recorded curl at the remaining joints. Distal bends are estimated from the preceding joint when fingertip landmarks are absent. Use the companion's hand close-up views to inspect the result.

The [avatar motion companion](static/recorded-motion.html) opens at `/recorded-motion.html` while the demo server is running. A small authored motion and face sample loads automatically; click **Play** without uploading files. It also plays locally selected BEAT motion, face, and WAV files on the bundled characters. These are presentation and data-inspection tools, separate from the paper implementation. No BEAT recording, dataset archive, or trained model is bundled. For recorded public motion, install the one preparation dependency and fetch a small official sample into ignored `outputs/beat-demo/`:

```sh
python -m pip install numpy
python scripts/beat_demo/fetch_modalities.py --speaker 1 --sequence 1_wayne_0_1_1 --include-bvh --max-bytes 25000000 --output-dir outputs/beat-demo/source
python scripts/beat_demo/prepare_bvh.py --bvh outputs/beat-demo/source/1_wayne_0_1_1.bvh --output outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json --frames 120
python scripts/beat_demo/prepare_modalities.py --sequence 1_wayne_0_1_1 --source outputs/beat-demo/source --output outputs/beat-demo/sample --frames 120
```

Open the companion and select `outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json`, `1_wayne_0_1_1-face.json`, and `1_wayne_0_1_1.wav`. The downloader caps each original file at 25 MB; the prepared clip contains up to 120 frames. The viewer uses local files and does not upload them. For other BEAT takes, substitute a matching official speaker and sequence ID.

If you already have OmniMo's processed 52-joint Unity humanoid data, use that normalized motion instead:

```sh
python scripts/beat_demo/prepare.py --dataset /path/to/processed/beat --speaker 1 --take 1_wayne_0_1_1 --output outputs/beat-demo/sample/1_wayne_0_1_1-motion.json --max-frames 120
```

Select the resulting `*-motion.json` in the companion. Its metadata carries the humanoid joint mapping and source-to-avatar coordinate conversion. The viewer fits source FK directions from the avatar's bind pose, following the spine explicitly at branching joints. This avoids applying incompatible source bone twist to the MPFB skin; it does not reproduce exact performer twist. The adapter supports Unity proximal/intermediate/distal finger names. Raw BVH remains a public-data alternative; do not mix the two skeleton conventions.
<!-- avatar-recorded-motion:end -->
