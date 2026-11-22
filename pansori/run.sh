#!/usr/bin/env bash
# One entry point for the pansori SOFA workflow, run from the SOFA root.
# See README.md for what each step does and why.
#
#   bash pansori/run.sh mfa      [LYRICS] [SUFFIX]          # 1. MFA alignment (in DiffSinger)
#   bash pansori/run.sh data     PHONES LYRICS OUT [args]   # 2. grade into full/weak/no_label
#   bash pansori/run.sh binarize DATA                       # 3. SOFA binarize
#   bash pansori/run.sh train    [pansori/train.py args]    # 4. fine-tune
#   bash pansori/run.sh align    CKPT LYRICS SUFFIX         # 5. align the corpus with SOFA
#
# Environment overrides: DIFFSINGER (repo path), SET (DiffSinger data set, default full),
# SOFA_ENV (default sofa), ALIGNER_ENV (default aligner).
set -euo pipefail

SOFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSINGER="${DIFFSINGER:-$HOME/Desktop/DiffSinger}"
SET="${SET:-full}"
SOFA_ENV="${SOFA_ENV:-sofa}"
ALIGNER_ENV="${ALIGNER_ENV:-aligner}"
export PYTHONNOUSERSITE=1          # stale ~/.local packages break both envs
cd "$SOFA_ROOT"

in_env() { local env="$1"; shift; conda run --no-capture-output -n "$env" "$@"; }

cmd="${1:-help}"; shift || true
case "$cmd" in
  mfa)
    # DiffSinger stage 04 with MFA: writes $DIFFSINGER/pansori/data/$SET/phones$SUFFIX.jsonl.
    # Kept in DiffSinger because it owns the IPA -> Coda phoneme rules (pansori_lib.convert_phone).
    lyrics="${1:-lyrics.jsonl}"; suffix="${2:-}"
    (cd "$DIFFSINGER/pansori/preprocess" &&
     in_env "$ALIGNER_ENV" python 04_align.py --set "$SET" --aligner mfa \
        --lyrics "$lyrics" ${suffix:+--suffix "$suffix"})
    ;;
  data)
    [ $# -ge 3 ] || { echo "usage: data PHONES LYRICS OUT [--max_floor_rate X] [--overwrite]"; exit 2; }
    phones="$1"; lyrics="$2"; out="$3"; shift 3
    python3 pansori/make_sofa_data.py --src "$DIFFSINGER/pansori/data/$SET" \
        --phones "$phones" --lyrics "$lyrics" --out "$out" "$@"
    ;;
  binarize)
    data="${1:?usage: binarize DATA}"
    mkdir -p "$data/binary" data/binary
    # The binarize config records which folder it built, next to the data.
    in_env "$SOFA_ENV" python - "$data" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/binarize_config.yaml"))
cfg["data_folder"] = sys.argv[1].rstrip("/") + "/"
yaml.safe_dump(cfg, open(f"{sys.argv[1]}/binarize_config.yaml", "w"), sort_keys=False)
EOF
    # SOFA's binarize.py writes global_config.yaml to data/binary/ whatever data_folder
    # says (binarize.py:396). Preserve data/'s copy and move the new one where it belongs.
    keep=""; [ -f data/binary/global_config.yaml ] && keep=$(mktemp) && cp data/binary/global_config.yaml "$keep"
    in_env "$SOFA_ENV" python binarize.py -c "$data/binarize_config.yaml"
    if [ "$(realpath "$data")" != "$(realpath data)" ]; then
      mv data/binary/global_config.yaml "$data/binary/global_config.yaml"
      [ -n "$keep" ] && mv "$keep" data/binary/global_config.yaml
    fi
    ls -la "$data/binary"
    ;;
  train)
    in_env "$SOFA_ENV" python pansori/train.py "$@"
    ;;
  align)
    [ $# -eq 3 ] || { echo "usage: align CKPT LYRICS SUFFIX   (e.g. align ckpt/korean_pansori/best/5000-5.9491.ckpt lyrics_hybrid.jsonl _hyb)"; exit 2; }
    ckpt="$(realpath "$1")"; lyrics="$2"; suffix="$3"
    # DiffSinger stage 04 with SOFA: builds the .lab corpus, runs infer.py here, and
    # writes phones$SUFFIX.jsonl + a .meta.json naming this exact checkpoint.
    (cd "$DIFFSINGER/pansori/preprocess" &&
     in_env "$ALIGNER_ENV" python 04_align.py --set "$SET" --aligner sofa \
        --sofa-ckpt "$ckpt" --lyrics "$lyrics" --suffix "$suffix")
    ;;
  *)
    sed -n '2,13p' "$0"; exit 2
    ;;
esac
