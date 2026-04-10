"""
Modèle image GLC25 — ResNet18 4 canaux
=======================================

Architecture :
    Image (4, 64, 64) ─► ResNet18 (4ch) ─► 512-d ─► Linear(num_classes) ─► logits

Métrique d'évaluation (challenge officiel) :
    F1 "samples-averaged" :
      Pour chaque survey i :
        F1_i = TP_i / (TP_i + (FP_i + FN_i) / 2)
      Score final = moyenne sur tous les surveys
    → implémenté manuellement (torchmetrics average="samples" non compatible)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import pytorch_lightning as pl


# ─────────────────────────────────────────────────────────────────────────────
# Métrique F1 samples-averaged (challenge officiel)
# ─────────────────────────────────────────────────────────────────────────────

def f1_samples(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    F1 samples-averaged : calcule un F1 par survey, puis fait la moyenne.

    Pour chaque survey i :
        pred_i = probs_i > threshold
        TP_i   = (pred_i & labels_i).sum()
        FP_i   = (pred_i & ~labels_i).sum()
        FN_i   = (~pred_i & labels_i).sum()
        F1_i   = TP_i / (TP_i + (FP_i + FN_i) / 2)   [0 si dénominateur = 0]

    Retourne la moyenne des F1_i sur le batch.

    Paramètres
    ----------
    probs  : (B, num_classes) float  — probabilités sigmoid
    labels : (B, num_classes) int/bool — labels ground truth
    """
    pred   = (probs >= threshold)
    labels = labels.bool()

    tp = (pred & labels).sum(dim=1).float()   # (B,)
    fp = (pred & ~labels).sum(dim=1).float()  # (B,)
    fn = (~pred & labels).sum(dim=1).float()  # (B,)

    denom = tp + (fp + fn) / 2.0
    f1    = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return f1.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Backbone ResNet18 4 canaux
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18_4ch(nn.Module):
    """
    ResNet18 pré-entraîné (ImageNet) adapté pour 4 bandes d'entrée (R, G, B, NIR).

    Initialisation du 4e canal : w_nir = mean(w_R, w_G, w_B)
    → gradient fluide dès la 1re epoch, sans casser les poids pré-entraînés.

    Le maxpool initial est remplacé par Identity() pour éviter de trop réduire
    les images 64×64 (sans maxpool : 64→32 au lieu de 64→32→16).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)

        # Adapter conv1 : 3 → 4 canaux
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Supprimer le maxpool : conserve mieux l'info spatiale sur 64×64
        backbone.maxpool = nn.Identity()

        self.embed_dim = backbone.fc.in_features   # 512
        backbone.fc    = nn.Identity()
        self.backbone  = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, 64, 64) → (B, 512)"""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# Module PyTorch Lightning
# ─────────────────────────────────────────────────────────────────────────────

class GLC25ImageSystem(pl.LightningModule):
    """
    ResNet18 4ch → classifieur multi-label (BCEWithLogitsLoss).

    Métrique de validation : F1 samples-averaged (métrique officielle du challenge).

    Hyperparamètres
    ---------------
    num_classes  : nombre d'espèces (5016)
    lr           : learning rate AdamW
    weight_decay : régularisation L2
    dropout      : dropout avant la tête linéaire
    threshold    : seuil sigmoid pour F1 (défaut 0.3)
    pretrained   : utiliser les poids ImageNet
    pos_weight   : poids des positifs dans BCE (défaut 10.0)
                   Pénalise davantage les faux négatifs (espèces présentes manquées).
                   Utile car 56% des espèces apparaissent dans < 20 surveys.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        threshold: float = 0.3,
        pretrained: bool = True,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone  = ResNet18_4ch(pretrained=pretrained)
        self.head      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.embed_dim, num_classes),
        )
        # pos_weight : donne plus d'importance aux espèces présentes (labels=1)
        # → réduit les faux négatifs sur les espèces rares
        pw = torch.full((num_classes,), pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Accumulateurs pour la validation
        self._val_f1_scores  = []
        self._val_losses     = []

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

        # F1 samples-averaged sur ce batch (accumulé pour la moyenne finale)
        f1 = f1_samples(probs, batch["label"], self.hparams.threshold)
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
