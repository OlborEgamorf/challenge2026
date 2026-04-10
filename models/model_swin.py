"""
Modèle image GLC25 — Swin Transformer Tiny 4 canaux
====================================================

Architecture :
    Image (4, 224, 224) ─► SwinTransformer-Tiny ─► 768-d ─► Linear(num_classes) ─► logits

Adaptation 4 canaux :
    Le patch embedding original est Conv2d(3, 96, 4, 4).
    On étend à 4 canaux en initialisant le 4e canal NIR
    avec la moyenne des 3 canaux RGB → préserve le transfer learning.

Normalisation :
    Z-score par bande par image : (x - µ) / σ  (fait dans dataset.py)

Métrique d'évaluation : F1 samples-averaged (identique à model.py)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

try:
    import timm
except ImportError:
    raise ImportError("timm requis : pip install timm")

from models.model import f1_samples


# ─────────────────────────────────────────────────────────────────────────────
# Backbone Swin-Tiny 4 canaux
# ─────────────────────────────────────────────────────────────────────────────

class SwinTiny4ch(nn.Module):
    """
    Swin Transformer Tiny pré-entraîné (ImageNet-1k) adapté pour 4 bandes.

    Modification du patch embedding :
        Conv2d(3→4, 96, kernel=4, stride=4)
        Le 4e canal NIR est initialisé à mean(R, G, B) → transfer learning préservé.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,      # supprime la tête de classification
            in_chans=3,         # charge d'abord avec 3 canaux
        )

        # Adapter le patch embedding : 3 → 4 canaux
        old_proj = self.backbone.patch_embed.proj   # Conv2d(3, 96, 4, 4)
        new_proj = nn.Conv2d(
            in_channels=4,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            new_proj.weight[:, 3:] = old_proj.weight.mean(dim=1, keepdim=True)
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.backbone.patch_embed.proj = new_proj

        self.embed_dim = self.backbone.num_features  # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, 224, 224) → (B, 768)"""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# Module PyTorch Lightning
# ─────────────────────────────────────────────────────────────────────────────

class GLC25SwinSystem(pl.LightningModule):
    """
    Swin-Tiny 4ch → classifieur multi-label (BCEWithLogitsLoss).

    Hyperparamètres
    ---------------
    num_classes  : nombre d'espèces (5016)
    lr           : learning rate AdamW
    weight_decay : régularisation L2
    dropout      : dropout avant la tête linéaire
    threshold    : seuil sigmoid pour F1 (défaut 0.3)
    pretrained   : utiliser les poids ImageNet
    pos_weight   : poids des positifs dans BCE (défaut 10.0)
    """

    def __init__(
        self,
        num_classes: int,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        threshold: float = 0.3,
        pretrained: bool = True,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = SwinTiny4ch(pretrained=pretrained)
        self.head     = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.embed_dim, num_classes),
        )

        pw = torch.full((num_classes,), pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self._val_f1_scores = []
        self._val_losses    = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))

    # ── Steps ────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        logits = self(batch["image"])
        loss   = self.criterion(logits, batch["label"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"])
        loss   = self.criterion(logits, batch["label"])
        probs  = torch.sigmoid(logits)
        f1     = f1_samples(probs, batch["label"], self.hparams.threshold)
        self._val_f1_scores.append(f1)
        self._val_losses.append(loss)

    def on_validation_epoch_end(self):
        mean_f1   = torch.stack(self._val_f1_scores).mean()
        mean_loss = torch.stack(self._val_losses).mean()
        self.log("val_F1",   mean_f1,   prog_bar=True)
        self.log("val_loss", mean_loss, prog_bar=True)
        self._val_f1_scores.clear()
        self._val_losses.clear()

    # ── Optimiseur ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            anneal_strategy="cos",
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
