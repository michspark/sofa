#!/usr/bin/env python3
"""SOFA's train.py, plus a best-checkpoint keeper, wandb, and run-level overrides.

Not one line of SOFA itself is edited; pl.Trainer is wrapped instead.

What it adds over `python train.py`:
  1. ModelCheckpoint on valid/total_loss (top-k) into ckpt/<model_name>/best/.
     SOFA keeps only the last checkpoint, and on this corpus validation loss
     bottoms out early (round 1: step 5,000 of 100,000) and then climbs.
  2. --model-name / --max-steps override the YAML without editing it. The
     effective config is written to ckpt/<model_name>/train_config.yaml so a
     run records what it actually used.
  3. wandb (TensorBoard scalars mirrored via sync_tensorboard). --no-wandb to skip.

Unknown flags are an error: an earlier version accepted and silently ignored
them, which would train the wrong model_name for the wrong number of steps.

    python pansori/train.py -p ckpt/pretrained_multilingual_singing/v1.0.0_multilingual_singing.ckpt \
        --model-name korean_pansori_hyb --max-steps 6000 -d data
"""

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Fine-tune SOFA (see README.md)")
parser.add_argument("--pretrained_model_path", "-p", default=None)
parser.add_argument("--config_path", "-c", default="configs/train_config.yaml")
parser.add_argument("--data_folder", "-d", default="data")
parser.add_argument("--resume", "-r", action="store_true")
parser.add_argument("--model-name", default=None, help="overrides model_name in the config")
parser.add_argument("--max-steps", type=int, default=None,
                    help="overrides optimizer_config.total_steps")
parser.add_argument("--save-top-k", type=int, default=5)
parser.add_argument("--wandb-project", default="pansori")
parser.add_argument("--wandb-group", default="aligner-ab")
parser.add_argument("--wandb-name", default=None, help="defaults to sofa_<model_name>")
parser.add_argument("--no-wandb", action="store_true")
args = parser.parse_args()

os.chdir(ROOT)                      # SOFA imports modules.* relative to its root
sys.path.insert(0, str(ROOT))

import lightning as pl              # noqa: E402
import yaml                         # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

config = yaml.safe_load(open(args.config_path, encoding="utf-8"))
if args.model_name:
    config["model_name"] = args.model_name
if args.max_steps:
    config["optimizer_config"]["total_steps"] = args.max_steps

run_dir = pathlib.Path("ckpt") / config["model_name"]
run_dir.mkdir(parents=True, exist_ok=True)
effective_config = run_dir / "train_config.yaml"
with open(effective_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
print(f"[pansori/train] model_name={config['model_name']} "
      f"total_steps={config['optimizer_config']['total_steps']} data={args.data_folder}")

if not args.no_wandb:
    import wandb
    wandb.init(
        project=args.wandb_project, group=args.wandb_group,
        name=args.wandb_name or f"sofa_{config['model_name']}",
        job_type="aligner-train", sync_tensorboard=True,
        config={"stage": "sofa-align", "model_name": config["model_name"],
                "pretrained": args.pretrained_model_path, "data_folder": args.data_folder,
                "total_steps": config["optimizer_config"]["total_steps"]},
    )

_Trainer = pl.Trainer


class TrainerWithBestCkpt(_Trainer):
    def __init__(self, *a, **kw):
        best_dir = run_dir / "best"
        kw["callbacks"] = list(kw.get("callbacks") or []) + [ModelCheckpoint(
            dirpath=str(best_dir), monitor="valid/total_loss", mode="min",
            save_top_k=args.save_top_k, save_last=True,
            filename="{step}-{valid/total_loss:.4f}", auto_insert_metric_name=False,
        )]
        print(f"[pansori/train] best checkpoints -> {best_dir} (top {args.save_top_k})")
        super().__init__(*a, **kw)


pl.Trainer = TrainerWithBestCkpt

import train as sofa_train           # noqa: E402  (must follow the patch)

sofa_train.main.callback(
    config_path=str(effective_config), data_folder=args.data_folder,
    pretrained_model_path=args.pretrained_model_path, resume=args.resume,
)
