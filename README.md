# SOFA, fine-tuned for pansori

This is a trimmed checkout of [qiuqiao/SOFA](https://github.com/qiuqiao/SOFA)
(Singing-Oriented Forced Aligner, commit `9549c6a`). It holds the Korean pansori
fine-tuning of SOFA used by the DiffSinger pansori project
(`~/Desktop/DiffSinger`, see `pansori/README.md` §8 and `pansori/docs/LYRICS_AB.md` §7 there).

The checkpoint currently in use:

```
~/Desktop/SOFA/ckpt/korean_pansori_hyb/best/6000-7.1941.ckpt
```

It produced `phones_hyb2.jsonl`, the labels behind the hyb2 model that the demo and the
paper use. This was established by re-alignment: it reproduces `aligned_hyb2/` byte for
byte on 20 of 20 test utterances.

---

## Contents

1. [Directory layout](#1-directory-layout)
2. [Setup](#2-setup)
3. [How the training works](#3-how-the-training-works)
4. [The floor-rate criterion (MFA labels)](#4-the-floor-rate-criterion-mfa-labels)
5. [Running it, step by step](#5-running-it-step-by-step)
6. [What was actually run](#6-what-was-actually-run)
7. [Gotchas](#7-gotchas)

---

## 1. Directory layout

```
SOFA/
├── pansori/                    ← everything specific to this project
│   ├── run.sh                    one entry point: mfa | data | binarize | train | align
│   ├── make_sofa_data.py         alignment → full/weak/no_label training data (§4)
│   └── train.py                  SOFA's train.py + best-checkpoint keeper + wandb + overrides
├── configs/
│   ├── binarize_config.yaml      upstream defaults (44.1 kHz, 128 mels, hop 512)
│   └── train_config.yaml         upstream defaults except model_name and dataloader_workers: 8
├── dictionary/korean-coda.txt  every Hangul syllable → Coda-34 phonemes (copied from DiffSinger)
├── data/                       round-2 training data (hybrid lyrics)
├── data_whisper/               round-1 training data (Whisper lyrics)
├── ckpt/
│   ├── pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt   starting point
│   ├── korean_pansori/best/5000-5.9491.ckpt                               round 1 best
│   └── korean_pansori_hyb/best/6000-7.1941.ckpt                           round 2 — in use
├── wandb/                      logs of the five training runs (2026-09-06, 2026-09-13)
├── binarize.py · train.py · infer.py · dataset.py · modules/    upstream SOFA, unmodified
└── LICENSE                     upstream MIT licence
```

The SOFA source code itself has not been edited. Removed from upstream: ONNX export and
inference, the evaluation script and its metrics, the Mandarin dictionary, the example
image, and the Chinese docs.

Both `binary/` folders are missing `train.h5py`; it was cleaned up to save disk space.
Re-run `binarize` before training on either folder again.

## 2. Setup

```sh
conda create -y -n sofa python=3.10
mkdir -p ~/miniconda3/envs/sofa/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > ~/miniconda3/envs/sofa/etc/conda/activate.d/no_user_site.sh
conda activate sofa
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt "setuptools<81"
```

- **`PYTHONNOUSERSITE=1` is mandatory.** Old packages in `~/.local` otherwise shadow the
  env's own packages.
- **`setuptools<81` is required.** `librosa<0.10` still imports `pkg_resources`.

The pretrained model is `pretrained_multilingual_singing` (v1.0.1) from the
[SOFA releases](https://github.com/qiuqiao/SOFA/releases). Unpack it into
`ckpt/pretrained_multilingual_singing/`; it is already in place here.

Steps 1 and 5 (the MFA and SOFA alignment runs) go through DiffSinger's stage 04. They
need DiffSinger's `aligner` env (MFA 3.4.1 + `korean_mfa`) and its data under
`~/Desktop/DiffSinger/pansori/data/full/`. Set `DIFFSINGER=...` if the repo lives
elsewhere.

## 3. How the training works

SOFA is a forced aligner:

- **Input:** an utterance's audio, plus the sequence of phonemes sung in it.
- **Output:** a start and end time for each phoneme.

It already knows *what* was sung. It only has to find *when*.

### Inputs

- **Audio.** 44.1 kHz audio becomes a 128-bin mel spectrogram, one frame per 11.6 ms. Each
  utterance is normalised to zero mean and unit variance. The model upsamples ×4, so it
  makes one decision every **~2.9 ms**.
- **Pitch-shifted copies.** Binarize stores two extra copies of every utterance, shifted
  by ±1–6 semitones.
- **Phoneme sequence.** The lyric text is mapped through `dictionary/korean-coda.txt` to
  35 symbols (34 Coda phonemes + `SP`). There is no IPA step, unlike MFA.
- **Labels.** These are borrowed from an earlier alignment and graded by how far they can
  be trusted (§4).

### Model

```
mel ──► U-Net backbone ──► linear head ──► per 2.9 ms frame:
        (from the           (new: our        ① which phoneme (35 classes)
         multilingual        vocab differs)   ② is this a phoneme boundary
         checkpoint)                          ③ CTC logits (ordering)
```

`-p <pretrained>` loads the backbone. Because the vocabulary size differs, SOFA
re-initialises only the head (`modules/task/forced_alignment.py`, `load_pretrained`).
In short, the "ear" is inherited and only the Korean phoneme labels are learned.

### One training step

A batch holds about 100 s of audio, with the three label grades sampled equally. Each
utterance enters twice, as two randomly chosen versions out of {original, shift A,
shift B}. Losses depend on the grade:

| Loss | Applied to | Teaches | Weight |
|---|---|---|---|
| Frame classification | full_label | "this 2.9 ms is `a`" | **10** |
| Boundary (BCE, EMD, diff) | full_label | where a phoneme changes; near misses cost less | 0.1 / 0.01 / 0.1 |
| CTC | full + weak | the phonemes appear in this order, timing unknown | 1 |
| Consistency | **all** (incl. no_label) | same output on the two pitch versions | 1 |
| Pseudo-label | — | disabled in SOFA's code (always 0) | (5) |

- **Ramp-up.** The CTC, consistency and boundary-diff losses start weak and ramp up.
- **Phoneme mask.** Only phonemes present in the utterance are allowed as frame classes.
- **Settings.** lr 5e-4 (backbone and head), weight decay 0.1, bf16-mixed, gradient
  clip 1.0.

### Validation and checkpoint selection

- **Validation set.** `binarize` holds out **15 labelled utterances** at random.
- **Schedule.** Every 500 steps the frame + boundary + CTC losses are measured on those 15.
- **Checkpoint keeper.** `pansori/train.py` keeps the 5 lowest `valid/total_loss`
  checkpoints in `ckpt/<model_name>/best/`. Upstream keeps only the last.

Two caveats when reading these numbers:

- 15 utterances is a small sample.
- The validation labels are themselves borrowed: MFA in round 1, round-1 SOFA in round 2.
  So "lowest loss" means "closest to the borrowed labels on those 15", not "most accurate".

Each run's validation set is different, so losses from different runs are **not
comparable**.

### Alignment

At alignment time the model outputs a phoneme probability and a boundary probability per
frame. A dynamic program then places the given phoneme sequence, in order and with
nothing skipped, at the boundaries that maximise the joint score. There is no minimum
duration, unlike MFA's 30 ms (§4).

## 4. The floor-rate criterion (MFA labels)

Nobody hand-labelled this corpus, so SOFA's targets come from an existing alignment —
in round 1, from **MFA**. MFA is a speech aligner and struggles with pansori. This
section explains how its timings are screened before SOFA sees them.

### MFA has a 30 ms floor

MFA works in 10 ms frames and models each phoneme with 3 HMM states, each needing at
least one frame. **No phoneme can be shorter than 0.03 s.**

When a singer holds one vowel for a second, MFA tends to hand all that time to one
phoneme and crush its neighbours onto the 0.03 s floor. A phone of exactly 0.03 s is
therefore **MFA giving up, not a measurement**. In this corpus 34.03 % of MFA's phonemes
sit on that floor.

### Two real utterances

**Trustworthy** — `c3c28013_0059` "초반을 마친 후", 3.31 s:

```
ch 0.110  o 0.290  b 0.050  a 0.200  n 0.060  eu 0.020  L 0.360
m 0.070   a 0.280  ch 0.090 i 0.150  N 0.180  h 0.050   u 1.397
```

1 of 14 phones is at or under the floor → floor rate **0.07**.

**Untrustworthy** — `037e72a6_0048` "고맙습니다", 3.02 s:

```
sil 1.210 | g 0.030  o 0.030  m 0.080  a 0.190  P 0.030  s 0.030
            eu 0.020  M 1.040  n 0.050  i 0.040  d 0.030  a 0.030 | sil 0.209
```

1.04 s went to `M`, and `g o P s eu d a` were crushed. 7 of 12 phones are at the floor →
floor rate **0.58**. The *order* g→o→m→a… is still right; the *timings* are not.

Durations under 0.03 s, like `eu 0.020`, appear where DiffSinger splits one IPA span into
two Coda phonemes and halves its length. They count as floor too.

### Definition

```
floor_rate = (# phonemes with duration ≤ 0.03 s) / (# phonemes, silence excluded)
```

### Distribution (Whisper lyrics + MFA, 26,955 aligned utterances)

| p10 | p25 | **median** | p75 | p90 |
|---|---|---|---|---|
| 0.10 | 0.20 | **0.35** | 0.55 | 0.75 |

### Choosing the threshold (`--max_floor_rate`)

How many utterances count as full_label at each threshold:

| threshold | 0.00 | 0.05 | 0.10 | **0.15** | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|---|
| full_label | 1,697 | 1,796 | 2,964 | **4,501** | 6,856 | 11,165 | 19,553 |

0.15 (at most about 1 phone in 7 on the floor) is the default. It was **not tuned**.
Stricter means cleaner timing targets, but fewer of them.

### The three grades — demotion, not discarding

| Grade | Condition | What SOFA gets | Losses it feeds |
|---|---|---|---|
| **full_label** | floor rate ≤ 0.15 | phoneme sequence **+ durations** | all |
| **weak_label** | floor rate > 0.15 | phoneme sequence only | CTC, consistency |
| **no_label** | no alignment, an OOV phone, or < 2 Hangul syllables | audio only | consistency |

SOFA learns *timing* only where MFA was trustworthy. It learns *order* and *sound*
everywhere else.

Applied to SOFA's own output (round 2), the criterion means less, because SOFA has no
floor. It then just screens out utterances with many very short phones, and many more
utterances pass (11,498 at 0.15).

## 5. Running it, step by step

All commands run from this directory. `run.sh` calls `conda run`, so no env needs to be
active. Its default data set is `SET=full`.

**Prerequisite:** DiffSinger stages 01–03 have produced, under
`~/Desktop/DiffSinger/pansori/data/full/`, the files `segments.jsonl`, `lyrics.jsonl`
(and `lyrics_hybrid.jsonl` for round 2), `spk_map.json` and `wavs/`. If `wavs/` is
missing, restore it in DiffSinger with
`python pansori/preprocess/01b_recut_wavs.py --set full`. Never re-run `01_segment.py`.

### Round 1 — bootstrap from MFA

```sh
# 1. MFA alignment of the Whisper lyrics → DiffSinger .../data/full/phones.jsonl
#    (hours; already done, skip if phones.jsonl exists)
bash pansori/run.sh mfa lyrics.jsonl

# 2. Grade into full/weak/no_label (prints the floor-rate table of §4)
bash pansori/run.sh data phones.jsonl lyrics.jsonl data_whisper --overwrite

# 3. Binarize (mel + 2 pitch-shifted copies, holds out 15 validation utterances)
bash pansori/run.sh binarize data_whisper

# 4. Fine-tune. Round 1 ran the config's 100,000 steps; validation bottomed at 5,000,
#    so 6,000 is plenty.
bash pansori/run.sh train -d data_whisper --model-name korean_pansori --max-steps 6000 \
    -p ckpt/pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt
```

Step 2 reproduces the existing `data_whisper/` exactly: 4,501 / 22,454 / 4,574, with
identical `transcriptions.csv`.

### Round 2 — bootstrap from round-1 SOFA, on hybrid lyrics

```sh
# 5. Align the hybrid lyrics with the round-1 SOFA → .../data/full/phones_hyb.jsonl
bash pansori/run.sh align ckpt/korean_pansori/best/5000-5.9491.ckpt lyrics_hybrid.jsonl _hyb

# 6–8. Same as above, from the SOFA labels
bash pansori/run.sh data phones_hyb.jsonl lyrics_hybrid.jsonl data --overwrite
bash pansori/run.sh binarize data
bash pansori/run.sh train -d data --model-name korean_pansori_hyb --max-steps 6000 \
    -p ckpt/pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt \
    --wandb-group lyrics-sofa-ab
```

Round 2 starts again from the **multilingual** checkpoint, not from round 1. Round 1
contributes only the labels.

### Using a trained checkpoint

1. **Pick one.** Look at the scores in `ckpt/<model_name>/best/{step}-{valid_loss}.ckpt`
   and at TensorBoard (`tensorboard --logdir ckpt/<model_name>/lightning_logs`).
2. **Align the corpus with it.** This writes `phones<SUFFIX>.jsonl` plus a `.meta.json`
   that records the exact checkpoint file:

   ```sh
   bash pansori/run.sh align ckpt/korean_pansori_hyb/best/6000-7.1941.ckpt lyrics_hybrid.jsonl _hyb2
   ```

3. **Make it the default (optional).** Point `sofa.ckpt` in DiffSinger's
   `pansori/preprocess/align.yaml` at it. Passing `--sofa-ckpt` per run, as `run.sh align`
   does, is preferred, because the alignment then records which weights made it.

From there, DiffSinger stages 06–08 take `phones<SUFFIX>.jsonl`; see DiffSinger
`pansori/docs/LYRICS_AB.md` §7.

## 6. What was actually run

| | Round 1 `korean_pansori` | Round 2 `korean_pansori_hyb` |
|---|---|---|
| Date | 2026-09-06 | 2026-09-13 |
| Lyrics | Whisper (`lyrics.jsonl`) | hybrid: Whisper + changbon syllables, Whisper's syllable count kept |
| Labels from | MFA (`phones.jsonl`) | round-1 SOFA (`phones_hyb.jsonl`) |
| Data folder | `data_whisper/` | `data/` |
| full / weak / no_label | 4,501 / 22,454 / 4,574 | 11,494 / 16,068 / 3,967 |
| Steps | 100,000 (two identical runs in parallel) | 6,000 (third attempt) |
| Best `valid/total_loss` | **5.9491 @ 5,000**, rising to 7.52 by 100k | **7.1941 @ 6,000**, still falling |
| Kept checkpoint | `ckpt/korean_pansori/best/5000-5.9491.ckpt` | `ckpt/korean_pansori_hyb/best/6000-7.1941.ckpt` |
| Produced | `phones_sofa.jsonl` → DiffSinger `sofa`/`sofar` | `phones_hyb2.jsonl` → DiffSinger `hyb2`/`hyb2r` |

Round 1 took about 2 h on one RTX 4090; round 2's successful run took about 5 min.

Round 2's first attempts:

- **v0** died at once with an h5py `KeyError: 'ph_seq'`.
- **v1** went NaN from step 1,500.
- **v2** ran cleanly after `data/binary` was rebuilt at 13:45, so stale binaries are the
  likely cause of both failures.

Round 2 was exported by a version of the data script that no longer exists. The current
`make_sofa_data.py` gives 11,498 / 16,222 / 3,809 from the same inputs. Where both
exports have a full_label utterance, the labels match exactly. The original additionally
routed 158 Whisper repetition hallucinations (36–440 repeated syllables, e.g. "으 으 으 …")
to no_label, and linked labelled wavs from DiffSinger's exported
`data/pansori/<speaker>/wavs/`. The `data/` folder here is the original export.

The 20 other checkpoints from these runs were deleted on 2026-11-22.

## 7. Gotchas

- **Zero-shot alignment is impossible.** The vocabulary is baked into the checkpoint, and
  the multilingual one has no Coda phonemes, so `infer.py` dies with a `KeyError`. Train
  first.
- **The latest checkpoint is not the best.** Round 1's validation minimum was at 5% of the
  run. Always pick from `best/`.
- **`binarize.py` writes `global_config.yaml` to `data/binary/` whatever `data_folder`
  says** (`binarize.py:396`). `run.sh binarize` moves it to the right folder and restores
  `data/`'s copy. If you call `binarize.py` by hand on another folder, do the same.
- **Re-binarize after re-exporting.** Training against a `binary/` built from older data
  is the likely cause of round 2's `KeyError` and NaN runs.
- **`make_sofa_data.py` refuses a non-empty `--out`** without `--overwrite`. Symlinks left
  over from an earlier export would otherwise be silently mixed in.
- **`pansori/train.py` rejects unknown flags.** Its predecessor silently ignored them; a
  mistyped `--max-steps` then meant 100,000 steps into the wrong `model_name` folder.
- **Small datasets.** `binning_length: 1000` (seconds) in `train_config.yaml` yields
  "No training batches" on a dataset much shorter than that. Lower it for smoke tests.
