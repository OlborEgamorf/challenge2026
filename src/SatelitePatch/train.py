"""
Entraînement GLC25 — modèle image ResNet18
==========================================

Usage (serveur NVIDIA)
-----
python models/train.py                         # auto-sélection GPU le plus libre
python models/train.py --gpu 2                 # forcer GPU 2
python models/train.py --fast-dev              # 2 epochs, 2% des données

Contraintes serveur
-------------------
1 GPU RTX 3090 (24 576 MiB) dédié au groupe.
ResNet18 + batch_size=128 + fp16 → ~5-6 GB VRAM.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

try:
    import tensorboard as _tb
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        import tensorboardX as _tb
        _TENSORBOARD_AVAILABLE = True
    except ImportError:
        _TENSORBOARD_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.SatelitePatch.dataset import build_datasets
from src.SatelitePatch.model import GLC25ImageSystem


# ─────────────────────────────────────────────────────────────────────────────
# Auto-sélection GPU
# ─────────────────────────────────────────────────────────────────────────────

def select_free_gpu() -> int:
    """Retourne l'index du GPU NVIDIA avec le plus de VRAM libre."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mem = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        best = int(np.argmax(free_mem))
        print(f"[GPU] GPU {best} sélectionné ({free_mem[best]} MiB libres)")
        return best
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "data_root":    ROOT,
    "val_fraction": 0.2,
    "num_workers":  8,
    "seed":         42,
    "batch_size":   128,
    "max_epochs":   100,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "dropout":      0.3,
    "threshold":    0.3,
    "pretrained":   True,
    "pos_weight":   10.0,   # poids des positifs dans BCE (espèces rares)
    "ckpt_dir":     os.path.join(ROOT, "checkpoints"),
    "log_dir":      os.path.join(ROOT, "logs"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class GLC25DataModule(pl.LightningDataModule):

    def __init__(self, cfg: dict, max_surveys: int = None):
        super().__init__()
        self.cfg = cfg
        self.max_surveys = max_surveys

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.meta = build_datasets(
            data_root=self.cfg["data_root"],
            val_fraction=self.cfg["val_fraction"],
            seed=self.cfg["seed"],
            max_surveys=self.max_surveys,
        )

    def train_dataloader(self):
        pin = self.cfg.get("pin_memory", True)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
            persistent_workers=self.cfg["num_workers"] > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        pin = self.cfg.get("pin_memory", True)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
            persistent_workers=self.cfg["num_workers"] > 0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class EpochSummary(pl.Callback):
    """Affiche un tableau récapitulatif à chaque fin d'epoch."""
    def on_validation_epoch_end(self, trainer, pl_module):
        # Lit les accumulateurs du modèle avant qu'ils soient vidés
        # (les callbacks s'exécutent avant on_validation_epoch_end du LightningModule)
        if not pl_module._val_f1_scores:
            return
        import torch as _torch
        mean_f1   = _torch.stack(pl_module._val_f1_scores).mean().item()
        mean_loss = _torch.stack(pl_module._val_losses).mean().item()
        epoch = trainer.current_epoch
        lr    = trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"\n  Epoch {epoch:>3} │ "
            f"val/loss={mean_loss:.5f} │ "
            f"val/F1={mean_f1:.5f} │ "
            f"lr={lr:.2e}"
        )


def build_callbacks(cfg: dict):
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return [
        ModelCheckpoint(
            dirpath=cfg["ckpt_dir"],
            filename="glc25-{epoch:02d}-{val_F1:.4f}",
            monitor="val/F1",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/F1", mode="max", patience=25),
        LearningRateMonitor(logging_interval="step"),
        EpochSummary(),
    ]


def build_loggers(cfg: dict, name: str):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    loggers = [CSVLogger(cfg["log_dir"], name=name)]
    if _TENSORBOARD_AVAILABLE:
        loggers.append(TensorBoardLogger(cfg["log_dir"], name=name))
    return loggers


def build_trainer(cfg: dict, accelerator: str, fast_dev: bool = False) -> pl.Trainer:
    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32",
        max_epochs=2 if fast_dev else cfg["max_epochs"],
        limit_train_batches=0.02 if fast_dev else 1.0,
        limit_val_batches=0.1  if fast_dev else 1.0,
        callbacks=build_callbacks(cfg),
        logger=build_loggers(cfg, "image_only"),
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, accelerator: str = "gpu", fast_dev: bool = False):
    pl.seed_everything(cfg["seed"], workers=True)

    print("\n══════════════════════════════════════")
    print("   GLC25 — ResNet18 4 canaux (image)")
    print(f"   Accélérateur : {accelerator}")
    print("══════════════════════════════════════\n")

    dm = GLC25DataModule(cfg)
    dm.setup()
    meta = dm.meta

    model = GLC25ImageSystem(
        num_classes=meta["num_classes"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        dropout=cfg["dropout"],
        threshold=cfg["threshold"],
        pretrained=cfg["pretrained"],
        pos_weight=cfg["pos_weight"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Espèces      : {meta['num_classes']}")
    print(f"Paramètres   : {n_params / 1e6:.1f}M")

    trainer = build_trainer(cfg, accelerator=accelerator, fast_dev=fast_dev)
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("resume"))

    best = trainer.checkpoint_callback.best_model_path
    print(f"\nMeilleur checkpoint : {best}")
    return model, best


# ─────────────────────────────────────────────────────────────────────────────
# CLI (serveur NVIDIA)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement GLC25 — serveur NVIDIA")
    p.add_argument("--fast-dev",    action="store_true")
    p.add_argument("--batch-size",  type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",      type=int,   default=DEFAULTS["max_epochs"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    p.add_argument("--workers",     type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--gpu",         type=int,   default=None,
                   help="Index GPU (0-3). Absent = auto-sélection du GPU le plus libre.")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--resume",      type=str, default=None,
                   help="Chemin vers un checkpoint .ckpt pour reprendre l'entraînement.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gpu_idx = args.gpu if args.gpu is not None else select_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    print(f"[GPU] CUDA_VISIBLE_DEVICES={gpu_idx}")

    cfg = {**DEFAULTS,
           "batch_size":  args.batch_size,
           "max_epochs":  args.epochs,
           "lr":          args.lr,
           "num_workers": args.workers,
           "pretrained":  not args.no_pretrain,
           "resume":      args.resume}

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    train(cfg, accelerator=accelerator, fast_dev=args.fast_dev)
