"""
Entraînement du modèle de distribution d'espèces GLC25
=======================================================

Usage
-----
# Modèle de fusion (image + tabulaire) — recommandé
python models/train.py

# Baseline tabulaire seulement (plus rapide)
python models/train.py --model tabular

# Vérification rapide (2 epochs, subset)
python models/train.py --fast-dev

# Forcer un GPU spécifique (sinon auto-sélection du GPU le plus libre)
python models/train.py --gpu 2

Contraintes serveur
-------------------
4 GPUs × 11 264 MiB disponibles → on utilise 1 GPU (auto-sélectionné).
Avec ResNet18 + batch_size=64 + fp16 → ~3-4 GB VRAM.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# ── Racine du projet ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-sélection GPU
# ─────────────────────────────────────────────────────────────────────────────

def select_free_gpu() -> int:
    """
    Interroge nvidia-smi et retourne l'index du GPU avec le plus de VRAM libre.
    Fallback sur GPU 0 si nvidia-smi n'est pas disponible.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mem = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        best = int(np.argmax(free_mem))
        print(f"[GPU] Auto-sélection → GPU {best} "
              f"({free_mem[best]} MiB libres | "
              f"autres : {[f'GPU{i}={m}MiB' for i, m in enumerate(free_mem) if i != best]})")
        return best
    except Exception as e:
        print(f"[GPU] nvidia-smi indisponible ({e}), fallback GPU 0")
        return 0

from models.dataset import build_datasets
from models.model import GLC25FusionSystem, TabularOnlySystem


# ─────────────────────────────────────────────────────────────────────────────
# Configuration par défaut
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Données
    "data_root":       ROOT,
    "val_fraction":    0.1,
    "num_workers":     8,       # threads DataLoader
    "seed":            42,

    # Entraînement
    "batch_size":      64,      # OK pour ResNet50 + fp16 sur 11 GB
    "max_epochs":      30,
    "lr":              1e-4,
    "weight_decay":    1e-4,
    "dropout":         0.4,

    # Modèle
    "tab_hidden":      256,
    "head_hidden":     512,
    "pretrained":      True,
    "pos_weight":      None,    # float ou None

    # Checkpoints
    "ckpt_dir":        os.path.join(ROOT, "checkpoints"),
    "log_dir":         os.path.join(ROOT, "logs"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class GLC25DataModule(pl.LightningDataModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.meta = build_datasets(
            data_root=self.cfg["data_root"],
            val_fraction=self.cfg["val_fraction"],
            seed=self.cfg["seed"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"] * 2,   # pas de gradient → double batch
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(cfg: dict, monitor: str = "val/F1"):
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return [
        ModelCheckpoint(
            dirpath=cfg["ckpt_dir"],
            filename="glc25-{epoch:02d}-{val/mAP:.4f}",
            monitor=monitor,
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=6,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def build_loggers(cfg: dict, name: str):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    return [
        TensorBoardLogger(cfg["log_dir"], name=name),
        CSVLogger(cfg["log_dir"], name=name),
    ]


def build_trainer(cfg: dict, name: str, fast_dev: bool = False) -> pl.Trainer:
    return pl.Trainer(
        # ── GPU : 1 seul (1/4 du serveur) ────────────────────────────────
        accelerator="gpu",
        devices=1,
        # ── Précision mixte fp16 (économise ~40% VRAM) ───────────────────
        precision="16-mixed",
        # ── Durée ────────────────────────────────────────────────────────
        max_epochs=2 if fast_dev else cfg["max_epochs"],
        limit_train_batches=0.02 if fast_dev else 1.0,
        limit_val_batches=0.1  if fast_dev else 1.0,
        # ── Callbacks & logs ─────────────────────────────────────────────
        callbacks=build_callbacks(cfg),
        logger=build_loggers(cfg, name),
        # ── Optimisations ────────────────────────────────────────────────
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        # ── Reproductibilité ─────────────────────────────────────────────
        deterministic=False,    # True ralentit ~30%
        enable_progress_bar=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement fusion (image + tabulaire)
# ─────────────────────────────────────────────────────────────────────────────

def train_fusion(cfg: dict, fast_dev: bool = False):
    pl.seed_everything(cfg["seed"], workers=True)

    print("\n══════════════════════════════════════════")
    print("   GLC25 — Modèle de fusion (ResNet18 + MLP)")
    print("══════════════════════════════════════════\n")

    # ── DataModule ────────────────────────────────────────────────────────
    dm = GLC25DataModule(cfg)
    dm.setup()
    meta = dm.meta

    print(f"Espèces        : {meta['num_classes']}")
    print(f"Features tab   : {meta['n_tab_features']}")
    print(f"GPU utilisé    : {os.environ.get('CUDA_VISIBLE_DEVICES', '0')} (1/4 du serveur)\n")

    # ── Modèle ───────────────────────────────────────────────────────────
    model = GLC25FusionSystem(
        num_classes=meta["num_classes"],
        n_tab_features=meta["n_tab_features"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        tab_hidden=cfg["tab_hidden"],
        head_hidden=cfg["head_hidden"],
        dropout=cfg["dropout"],
        pretrained_image=cfg["pretrained"],
        pos_weight=cfg["pos_weight"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres     : {n_params / 1e6:.1f}M")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = build_trainer(cfg, name="fusion", fast_dev=fast_dev)
    trainer.fit(model, datamodule=dm)

    best = trainer.checkpoint_callback.best_model_path
    print(f"\nMeilleur checkpoint : {best}")
    return model, dm, best


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement baseline tabulaire seulement
# ─────────────────────────────────────────────────────────────────────────────

def train_tabular_only(cfg: dict, fast_dev: bool = False):
    pl.seed_everything(cfg["seed"], workers=True)

    print("\n══════════════════════════════════════════")
    print("   GLC25 — Baseline tabulaire (MLP seul)")
    print("══════════════════════════════════════════\n")

    dm = GLC25DataModule(cfg)
    dm.setup()
    meta = dm.meta

    model = TabularOnlySystem(
        num_classes=meta["num_classes"],
        n_tab_features=meta["n_tab_features"],
        lr=3e-4,
    )

    trainer = build_trainer(cfg, name="tabular_only", fast_dev=fast_dev)
    trainer.fit(model, datamodule=dm)
    return model, dm


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement GLC25")
    p.add_argument("--model",       choices=["fusion", "tabular"], default="fusion")
    p.add_argument("--fast-dev",    action="store_true", help="2 epochs, subset de données")
    p.add_argument("--batch-size",  type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",      type=int,   default=DEFAULTS["max_epochs"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    p.add_argument("--workers",     type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--gpu",         type=int,   default=None,
                   help="Index du GPU à utiliser (0-3). Si absent : auto-sélection du GPU le plus libre.")
    p.add_argument("--no-pretrain", action="store_true", help="Désactiver ImageNet weights")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Sélection GPU ─────────────────────────────────────────────────────────
    gpu_idx = args.gpu if args.gpu is not None else select_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    cfg = {**DEFAULTS}
    cfg["batch_size"]  = args.batch_size
    cfg["max_epochs"]  = args.epochs
    cfg["lr"]          = args.lr
    cfg["num_workers"] = args.workers
    cfg["pretrained"]  = not args.no_pretrain

    if not torch.cuda.is_available():
        print("[AVERTISSEMENT] Aucun GPU détecté — passage en mode CPU (lent)")

    if args.model == "fusion":
        train_fusion(cfg, fast_dev=args.fast_dev)
    else:
        train_tabular_only(cfg, fast_dev=args.fast_dev)
