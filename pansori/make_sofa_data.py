#!/usr/bin/env python3
"""Turn an existing alignment (MFA, or an earlier SOFA) into SOFA training data.

Standalone port of DiffSinger's pansori/preprocess/04b_sofa_data.py -- no
DiffSinger imports, so it runs from this directory under any env with Python 3.

Why the three-way split (see README.md, "The floor-rate criterion"):
MFA cannot make a phone shorter than 0.03 s (10 ms frames x 3 HMM states). When
it loses track of a sustained pansori vowel it crushes the neighbouring phones
onto that floor, so a phone of exactly 0.03 s is MFA giving up, not a
measurement. Handing every MFA timing to SOFA would teach SOFA that habit. So
each utterance is graded by its floor rate -- the share of non-silence phones
<= 0.03 s:

    floor_rate <= --max_floor_rate  ->  full_label   phone sequence + durations
    above it                        ->  weak_label   phone sequence only
    no usable alignment             ->  no_label     audio only

Input  (--src, a DiffSinger set directory such as .../pansori/data/full):
    segments.jsonl   every utterance (item_name)
    <phones>         the alignment: {"item_name", "phones": [[ph, start, end], ...]}
    <lyrics>         {"item_name", "text"}  (only used to count Hangul)
    spk_map.json     {"item2spk": {item_name: speaker}}
    wavs/*.wav       the cut utterances

Output (--out, default ./data):
    full_label/{speaker}/wavs/*.wav + transcriptions.csv   name,ph_seq,ph_dur
    weak_label/{speaker}/wavs/*.wav + transcriptions.csv   name,ph_seq
    no_label/*.wav

wavs are symlinks; SOFA's binarizer only reads them.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SRC = Path.home() / "Desktop/DiffSinger/pansori/data/full"

# MFA's minimum phone duration. Same constant as DiffSinger's
# pansori/eval/label_report.py -- keep the two in step or the numbers drift.
MFA_FLOOR = 0.030
EPS = 1e-6
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def hangul_only(text: str) -> str:
    return "".join(ch for ch in text if "가" <= ch <= "힣")


def phone_rows(record):
    """phones row -> [(ph, start, end)], silence renamed SP (SOFA's convention).

    None if any phone is OOV: SOFA's binarizer dies on a phone missing from the
    dictionary, so such an utterance can only be used as audio.
    """
    out = []
    for ph, start, end in record["phones"]:
        if ph.startswith("<OOV"):
            return None
        out.append(("SP" if ph.lower() in SILENCE_LABELS else ph, start, end))
    return out or None


def floor_rate(phones) -> float:
    """Share of non-SP phones at or under MFA's 0.03 s floor."""
    spoken = [(ph, s, e) for ph, s, e in phones if ph != "SP"]
    if not spoken:
        return 1.0
    return sum(1 for _, s, e in spoken if e - s <= MFA_FLOOR + EPS) / len(spoken)


def link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def write_csv(path: Path, rows, with_dur: bool) -> None:
    fields = ["name", "ph_seq"] + (["ph_dur"] if with_dur else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"DiffSinger set directory (default {DEFAULT_SRC})")
    p.add_argument("--phones", default="phones.jsonl",
                   help="alignment file inside --src (phones.jsonl = MFA on Whisper lyrics)")
    p.add_argument("--lyrics", default="lyrics.jsonl", help="lyrics file inside --src")
    p.add_argument("--out", type=Path, default=Path("data"), help="SOFA data folder")
    p.add_argument("--max_floor_rate", type=float, default=0.15,
                   help="above this floor rate an utterance is demoted to weak_label")
    p.add_argument("--min_hangul", type=int, default=2)
    p.add_argument("--no_label_limit", type=int, default=0, help="0 = all")
    p.add_argument("--overwrite", action="store_true",
                   help="delete existing full_label/weak_label/no_label/binary under --out first")
    args = p.parse_args()

    src, out = args.src.expanduser(), args.out.expanduser()
    wav_dir = src / "wavs"
    if not wav_dir.is_dir():
        sys.exit(f"{wav_dir} does not exist. Restore it in DiffSinger with "
                 f"`python pansori/preprocess/01b_recut_wavs.py --set {src.name}`.")

    existing = [out / d for d in ("full_label", "weak_label", "no_label", "binary")
                if (out / d).exists() and any((out / d).iterdir())]
    if existing:
        if not args.overwrite:
            sys.exit(f"{out} already holds data ({', '.join(d.name for d in existing)}). "
                     "Stale utterances from an earlier export would be mixed in. "
                     "Pass --overwrite, or choose another --out.")
        for d in existing:
            shutil.rmtree(d)

    with open(src / "spk_map.json", encoding="utf-8") as f:
        item2spk = json.load(f)["item2spk"]
    lyrics = {r["item_name"]: r["text"] for r in read_jsonl(src / args.lyrics)}
    aligned = {r["item_name"]: r for r in read_jsonl(src / args.phones)}

    buckets = {"full_label": {}, "weak_label": {}}
    no_label, skipped, rates = [], Counter(), []

    for record in read_jsonl(src / "segments.jsonl"):
        item = record["item_name"]
        wav = wav_dir / f"{item}.wav"
        if not wav.exists():
            skipped["no wav"] += 1
            continue
        speaker = item2spk.get(item)
        if speaker is None:
            skipped["speaker unknown"] += 1
            continue

        phones = phone_rows(aligned[item]) if item in aligned else None
        if phones is None:
            if len(hangul_only(lyrics.get(item, ""))) < args.min_hangul:
                skipped["too little text (kept as no_label)"] += 1
            no_label.append((item, wav))
            continue

        rate = floor_rate(phones)
        rates.append(rate)
        bucket = "full_label" if rate <= args.max_floor_rate else "weak_label"
        row = {"name": item,
               "ph_seq": " ".join(ph for ph, _, _ in phones),
               "ph_dur": " ".join(f"{e - s:.6f}" for _, s, e in phones)}
        buckets[bucket].setdefault(speaker, []).append((row, wav))

    if rates:
        rates.sort()
        n = len(rates)
        print(f"floor_rate over {n:,} aligned utterances:")
        print("  " + "  ".join(f"p{q}={rates[min(n - 1, q * n // 100)]:.2f}"
                               for q in (10, 25, 50, 75, 90)))
        print("  full_label by threshold: " + "  ".join(
            f"{t:.2f}->{sum(r <= t for r in rates):,}" for t in (0.05, 0.10, 0.15, 0.20, 0.30)))
        print(f"  chosen: {args.max_floor_rate:.2f}\n")

    for bucket, by_speaker in buckets.items():
        total = 0
        for speaker, entries in sorted(by_speaker.items()):
            base = out / bucket / speaker
            for row, wav in entries:
                link(wav, base / "wavs" / f"{row['name']}.wav")
            write_csv(base / "transcriptions.csv", [r for r, _ in entries],
                      with_dur=(bucket == "full_label"))
            total += len(entries)
        print(f"{bucket:<11} {total:>7,} utterances  {len(by_speaker)} speakers")

    if args.no_label_limit:
        no_label = no_label[:args.no_label_limit]
    for item, wav in no_label:
        link(wav, out / "no_label" / f"{item}.wav")
    print(f"{'no_label':<11} {len(no_label):>7,} utterances")
    if skipped:
        print("notes: " + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common()))
    print(f"\n-> {out}   next: bash pansori/run.sh binarize {out}")


if __name__ == "__main__":
    main()
