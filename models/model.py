"""
Modèles pour GLC25 – distribution d'espèces
============================================

ImageEncoder   : ResNet18 modifié pour 4 bandes (R, G, B, NIR)
TabularEncoder : MLP avec BatchNorm et résiduel (64 features → 256)
FusionModel    : concatène les deux encodeurs → classifieur multi-label

Architecture :
    Image (4, 64, 64) ─► ResNet18 (4ch) ─►  512-d
                                                    ├─► Concat (512+256) ─► head ─► logits (5016)
    Tabular (64,) ──────► MLP ──────────►  256-d

Note ResNet18 4 canaux :
    Le poids conv1 officiel est (64, 3, 7, 7).
    On étend à (64, 4, 7, 7) en initialisant le 4e canal NIR
    avec la moyenne des 3 canaux originaux → préserve le transfer learning.
    ResNet18 produit des embeddings 512-d (vs 2048-d pour ResNet50),
    ce qui réduit la mémoire et accélère l'entraînement.

Métrique d'évaluation (challenge officiel) :
    F1-score micro, moyenné par sample.
    Pour chaque survey : F1 = 2|pred ∩ gt| / (|pred| + |gt|)
    → on seuille les probabilités à 0.5 pour décider des espèces prédites.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import pytorch_lightning as pl
from torchmetrics import F1Score, Precision, Recall, AUROC


# ─────────────────────────────────────────────────────────────────────────────
# 1. Encodeur image : ResNet18 4 canaux
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18_4ch(nn.Module):
    """
    ResNet18 pré-entraîné (ImageNet) adapté pour 4 bandes d'entrée.

    Stratégie d'initialisation du 4e canal (NIR) :
        w_nir = mean(w_R, w_G, w_B)  → gradient fluide dès la 1re epoch.
    La couche de classification finale est remplacée par Identity()
    pour exposer les embeddings 512-d.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)

        # ── Adapter conv1 pour 4 canaux ───────────────────────────────────
        old_conv = backbone.conv1             # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight          # R, G, B → identiques
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)  # NIR ← moy(RGB)
        backbone.conv1 = new_conv

        # ── Supprimer la tête de classification ───────────────────────────
        self.embed_dim = backbone.fc.in_features   # 512
        backbone.fc    = nn.Identity()

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, 64, 64) → (B, 512)"""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Encodeur tabulaire : MLP avec connexion résiduelle
# ─────────────────────────────────────────────────────────────────────────────

class TabularMLP(nn.Module):
    """
    MLP robuste pour les features environnementales.

    Architecture :
        Input (64) → [Linear(256) → BN → GELU → Dropout]×2
                   → [Linear(256) → BN → GELU] + résidu
                   → Output (256)

    Choix de GELU : meilleures performances empiriques sur features tabulaires
    continues vs ReLU.
    """

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = hidden

        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        # Résidu : ajouter l'entrée de block3 à sa sortie
        # (pas de projection nécessaire car hidden→hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_features) → (B, 256)"""
        x = self.block1(x)
        x = self.block2(x)
        residual = x
        x = self.block3(x) + residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modèle de fusion
# ─────────────────────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Fusion de l'encodeur image et tabulaire pour la classification multi-label.

    Architecture de la tête :
        Concat(512 + 256) → Linear(512) → BN → GELU → Dropout(0.4)
                          → Linear(num_classes)

    Sortie : logits non-activés (utiliser BCEWithLogitsLoss).
    """

    def __init__(
        self,
        num_classes: int,
        n_tab_features: int,
        tab_hidden: int = 256,
        head_hidden: int = 512,
        dropout: float = 0.4,
        pretrained_image: bool = True,
    ):
        super().__init__()
        self.image_encoder   = ResNet18_4ch(pretrained=pretrained_image)
        self.tabular_encoder = TabularMLP(n_tab_features, hidden=tab_hidden)

        fusion_in = self.image_encoder.embed_dim + self.tabular_encoder.embed_dim  # 512 + 256

        self.head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        image   : (B, 4, 64, 64)
        tabular : (B, n_tab_features)
        → logits : (B, num_classes)
        """
        img_emb = self.image_encoder(image)      # (B, 512)
        tab_emb = self.tabular_encoder(tabular)  # (B, 256)
        fused   = torch.cat([img_emb, tab_emb], dim=1)  # (B, 768)
        return self.head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Module PyTorch Lightning (compatible Malpolon / standalone)
# ─────────────────────────────────────────────────────────────────────────────

class GLC25FusionSystem(pl.LightningModule):
    """
    LightningModule wrappant FusionModel.

    Compatible avec l'interface Malpolon (ClassificationSystem).
    Peut être utilisé directement avec pl.Trainer sans Malpolon.

    Hyperparamètres
    ---------------
    lr            : learning rate initial (AdamW)
    weight_decay  : régularisation L2
    warmup_steps  : nombre de steps de warmup cosine
    pos_weight    : poids pour les positifs dans BCEWithLogitsLoss
                    (None = pas de pondération)
    threshold     : seuil sigmoid pour décider des espèces prédites (défaut 0.3)
                    utilisé pour le F1-score (métrique officielle du challenge)
    """

    def __init__(
        self,
        num_classes: int,
        n_tab_features: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        pos_weight: float = None,
        tab_hidden: int = 256,
        head_hidden: int = 512,
        dropout: float = 0.4,
        pretrained_image: bool = True,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FusionModel(
            num_classes=num_classes,
            n_tab_features=n_tab_features,
            tab_hidden=tab_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
            pretrained_image=pretrained_image,
        )

        # Perte : BCEWithLogitsLoss est plus stable numériquement que BCE + Sigmoid
        if pos_weight is not None:
            pw = torch.full((num_classes,), pos_weight)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # ── Métriques de validation ──────────────────────────────────────────
        # F1 samples : métrique officielle du challenge
        self.val_f1 = F1Score(
            task="multilabel", num_labels=num_classes,
            average="samples", threshold=threshold,
        )
        # Precision & Recall macro (moyenne par espèce)
        self.val_precision = Precision(
            task="multilabel", num_labels=num_classes,
            average="macro", threshold=threshold,
        )
        self.val_recall = Recall(
            task="multilabel", num_labels=num_classes,
            average="macro", threshold=threshold,
        )
        # AUC-ROC macro (pas de seuil, calculé sur la courbe entière)
        self.val_auroc = AUROC(
            task="multilabel", num_labels=num_classes,
            average="macro",
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, image, tabular):
        return self.model(image, tabular)

    # ── Steps ────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        image   = batch["image"]
        tabular = batch["tabular"]
        label   = batch["label"]

        logits = self(image, tabular)
        loss   = self.criterion(logits, label)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
        return loss, logits, label

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, label = self._shared_step(batch, "val")
        probs = torch.sigmoid(logits)
        labels_int = label.int()
        self.val_f1.update(probs, labels_int)
        self.val_precision.update(probs, labels_int)
        self.val_recall.update(probs, labels_int)
        self.val_auroc.update(probs, labels_int)

    def on_validation_epoch_end(self):
        self.log("val/F1",        self.val_f1.compute(),        prog_bar=True)
        self.log("val/Precision", self.val_precision.compute(), prog_bar=False)
        self.log("val/Recall",    self.val_recall.compute(),    prog_bar=False)
        self.log("val/AUC-ROC",   self.val_auroc.compute(),     prog_bar=False)
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()

    # ── Optimiseur ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Scheduler cosine avec warmup linéaire
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,          # ~5% warmup
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline tabulaire : MLP seul (pour ablation)
# ─────────────────────────────────────────────────────────────────────────────

class TabularOnlySystem(pl.LightningModule):
    """
    Baseline MLP sur features tabulaires seulement (sans image).
    Utile pour mesurer la contribution de chaque modalité.
    """

    def __init__(
        self,
        num_classes: int,
        n_tab_features: int,
        lr: float = 3e-4,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TabularMLP(n_tab_features, hidden=512)
        self.head    = nn.Linear(512, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_f1 = F1Score(
            task="multilabel", num_labels=num_classes,
            average="samples", threshold=threshold,
        )
        self.val_precision = Precision(
            task="multilabel", num_labels=num_classes,
            average="macro", threshold=threshold,
        )
        self.val_recall = Recall(
            task="multilabel", num_labels=num_classes,
            average="macro", threshold=threshold,
        )
        self.val_auroc = AUROC(
            task="multilabel", num_labels=num_classes,
            average="macro",
        )

    def forward(self, tabular):
        return self.head(self.encoder(tabular))

    def _shared_step(self, batch, stage):
        logits = self(batch["tabular"])
        loss   = self.criterion(logits, batch["label"])
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        return loss, logits, batch["label"]

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, logits, label = self._shared_step(batch, "val")
        probs = torch.sigmoid(logits)
        labels_int = label.int()
        self.val_f1.update(probs, labels_int)
        self.val_precision.update(probs, labels_int)
        self.val_recall.update(probs, labels_int)
        self.val_auroc.update(probs, labels_int)

    def on_validation_epoch_end(self):
        self.log("val/F1",        self.val_f1.compute(),        prog_bar=True)
        self.log("val/Precision", self.val_precision.compute(), prog_bar=False)
        self.log("val/Recall",    self.val_recall.compute(),    prog_bar=False)
        self.log("val/AUC-ROC",   self.val_auroc.compute(),     prog_bar=False)
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
