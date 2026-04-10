import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split


# =============================================================================
# LABELS
# =============================================================================

def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)

    unique_species = sorted(df["speciesId"].dropna().astype(int).unique())
    species2idx    = {sp: i for i, sp in enumerate(unique_species)}
    idx2species    = {i: sp for sp, i in species2idx.items()}
    num_classes    = len(unique_species)

    print(f"Espèces uniques   : {num_classes:,}")
    print(f"Observations CSV  : {len(df):,}")
    print(f"Sites uniques     : {df['surveyId'].nunique():,}")

    labels = {}
    for survey_id, group in df.groupby("surveyId"):
        vec = torch.zeros(num_classes, dtype=torch.float32)
        for sp in group["speciesId"]:
            vec[species2idx[sp]] = 1.0
        labels[int(survey_id)] = vec

    # Statistiques de fréquence par espèce (utile pour debug / ASL)
    all_vecs  = torch.stack(list(labels.values()))
    n_sites   = all_vecs.shape[0]
    n_pos     = all_vecs.sum(dim=0).clamp(min=1)
    freq      = n_pos / n_sites          # fréquence de présence par espèce
    print(f"Fréquence espèce  : min={freq.min():.5f} | "
          f"median={freq.median():.4f} | max={freq.max():.4f}")

    return labels, species2idx, idx2species, num_classes


# =============================================================================
# ASYMMETRIC LOSS  (Ridnik et al., 2021)
# Meilleure que BCE+pos_weight pour le multi-label déséquilibré :
#   - down-weighting agressif des faux-négatifs faciles  (gamma_neg > gamma_pos)
#   - margin shift m pour éliminer les négatifs très faciles
# Ceci évite que les espèces rares soient noyées par le signal des négatifs.
# =============================================================================

class AsymmetricLoss(nn.Module):
    """
    Args:
        gamma_neg : focusing pour les négatifs (défaut 4)
        gamma_pos : focusing pour les positifs (défaut 1)
        clip      : margin shift m ∈ [0, 0.05] sur les négatifs
        eps       : stabilité numérique
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float      = 0.05,
        eps: float       = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Margin shift sur les négatifs
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Focusing asymétrique
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt  = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w     = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


# =============================================================================
# DATASET
# =============================================================================

class LandsatCubeDataset(Dataset):
    def __init__(self, root_dir: str, labels: Optional[dict] = None, transform=None):
        """
        labels=None pour les données de test (inférence seule).
        """
        all_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if labels is not None:
            self.files = [f for f in all_files if self._sid(f) in labels]
            if not self.files:
                raise ValueError("Aucun cube ne correspond aux surveyId du CSV.")
        else:
            self.files = all_files          # mode test : on prend tout

        self.labels    = labels
        self.transform = transform
        print(f"Cubes disponibles : {len(all_files):,}")
        print(f"Cubes utilisés    : {len(self.files):,}")

    @staticmethod
    def _sid(path: str) -> int:
        return int(os.path.basename(path).split("_")[-2])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cube = torch.load(self.files[idx], map_location="cpu").float()
        cube = torch.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

        # Padding temporel à 21 si nécessaire
        if cube.shape[2] < 21:
            pad  = torch.zeros(6, 4, 21 - cube.shape[2])
            cube = torch.cat([cube, pad], dim=2)

        # Normalisation min-max par bande (préserve les contrastes spectraux)
        c_min = cube.flatten(1).min(dim=1).values[:, None, None]
        c_max = cube.flatten(1).max(dim=1).values[:, None, None]
        cube  = (cube - c_min) / (c_max - c_min + 1e-6)

        if self.transform:
            cube = self.transform(cube)

        if self.labels is not None:
            return cube, self.labels[self._sid(self.files[idx])]
        return cube,                        # tuple à 1 élément en mode test


# =============================================================================
# SPLIT TRAIN / VALIDATION
# =============================================================================

def make_dataloaders(dataset, val_ratio=0.15, batch_size=64, num_workers=4, seed=42):
    n_val   = math.floor(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"\nSplit  →  train : {n_train:,}  |  val : {n_val:,}  ({val_ratio:.0%})")
    return train_loader, val_loader, val_ds


# =============================================================================
# ARCHITECTURE : SE-ResNet  (Squeeze-and-Excitation + 4 blocs)
#
# L'attention par canal (SE) permet au réseau d'apprendre quelles bandes
# spectrales (NDVI, NIR, SWIR…) sont discriminantes pour chaque espèce,
# ce qui améliore la détection des espèces rares dont le signal est fin.
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation : recalibration des canaux."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SEResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se   = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.se(self.net(x)))


class SEResNet(nn.Module):
    """
    Input  : (B, 6, 4, 21)   — 6 bandes, 4 pixels, 21 pas de temps
    Output : (B, num_classes) — logits bruts

    Architecture :
        stem       : Conv 6→64
        stage1 ×2  : SEResBlock 64  (spatial 4×21)
        downsample : Conv 64→128, stride 2
        stage2 ×2  : SEResBlock 128 (spatial 2×10)
        GAP + head : Dropout + Linear
    """
    def __init__(self, num_classes: int, base_ch: int = 64, dropout: float = 0.3):
        super().__init__()
        ch2 = base_ch * 2

        self.stem = nn.Sequential(
            nn.Conv2d(6, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            SEResBlock(base_ch, dropout=dropout * 0.5),
            SEResBlock(base_ch, dropout=dropout * 0.5),
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_ch, ch2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            SEResBlock(ch2, dropout=dropout),
            SEResBlock(ch2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        return self.head(self.pool(x))


# =============================================================================
# SCHEDULER : warmup linéaire + cosine decay
# Le warmup évite les grands gradients en début d'entraînement, ce qui
# stabilise les BatchNorm et réduit l'overfitting précoce.
# =============================================================================

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs           # warmup linéaire
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# MICRO F1
# =============================================================================

def micro_f1_from_counts(tp, fp, fn) -> float:
    sum_tp = tp.sum().item()
    sum_fp = fp.sum().item()
    sum_fn = fn.sum().item()
    precision = sum_tp / (sum_tp + sum_fp + 1e-8)
    recall    = sum_tp / (sum_tp + sum_fn + 1e-8)
    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# BOUCLES TRAIN / VALIDATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(cubes), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * cubes.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    num_classes = model.head[-1].out_features

    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)

    preds_list, labels_list = [], []

    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        logits = model(cubes)
        total_loss += criterion(logits, labels).item() * cubes.size(0)

        preds = torch.sigmoid(logits) >= threshold
        tp += (preds  &  labels.bool()).float().sum(dim=0)
        fp += (preds  & ~labels.bool()).float().sum(dim=0)
        fn += (~preds &  labels.bool()).float().sum(dim=0)

        preds_list.append(preds.cpu())
        labels_list.append(labels.cpu())

    return (
        total_loss / len(loader.dataset),
        micro_f1_from_counts(tp, fp, fn),
        torch.cat(preds_list,  dim=0),
        torch.cat(labels_list, dim=0),
    )


# =============================================================================
# EXPORT CSV VALIDATION
# =============================================================================

def export_predictions(all_preds, all_labels, val_ds, full_dataset, idx2species,
                        out_path="predictions.csv"):
    val_indices = val_ds.indices
    survey_ids  = [full_dataset._sid(full_dataset.files[i]) for i in val_indices]
    rows = []
    for site_idx, survey_id in enumerate(survey_ids):
        true_vec = all_labels[site_idx]
        pred_vec = all_preds[site_idx]
        active   = (true_vec.bool() | pred_vec).nonzero(as_tuple=True)[0]
        for k in active.tolist():
            rows.append({
                "surveyId"      : survey_id,
                "speciesId"     : idx2species[k],
                "presence_true" : int(true_vec[k].item()),
                "presence_pred" : int(pred_vec[k].item()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nExport val → {out_path}  ({len(df):,} lignes, {df['surveyId'].nunique():,} sites)")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    CSV_PATH      = "/data/challenge2026MIASHS/GLC25_PA_metadata_train.csv"
    DATA_DIR      = "/data/challenge2026MIASHS/SateliteTimeSeries-Landsat/cubes/PA-train"
    OUT_DIR       = "res_landsat"
    os.makedirs(OUT_DIR, exist_ok=True)

    EPOCHS        = 60
    WARMUP_EPOCHS = 5
    BATCH         = 32
    LR            = 3e-4
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {DEVICE}\n")

    # 1. Labels
    labels, species2idx, idx2species, num_classes = load_labels(CSV_PATH)

    # 2. Dataset + split
    dataset = LandsatCubeDataset(root_dir=DATA_DIR, labels=labels)
    train_loader, val_loader, val_ds = make_dataloaders(
        dataset, val_ratio=0.15, batch_size=BATCH
    )

    # 3. Modèle
    model = SEResNet(num_classes=num_classes, base_ch=64, dropout=0.3).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres        : {n_p:,}\n")

    # 4. Loss + optimiseur + scheduler
    #    ASL gamma_neg=4, gamma_pos=1, clip=0.05 : valeurs par défaut recommandées
    #    par les auteurs pour les benchmarks multi-label faune/flore.
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    # 5. Entraînement
    best_f1     = 0.0
    best_preds  = None
    best_labels = None

    print(f"\n{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>8}  {'micro_F1':>8}  {'lr':>8}")
    print("-" * 58)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, micro_f1, all_preds, all_labels = evaluate(
            model, val_loader, criterion, DEVICE, threshold=0.5
        )
        scheduler.step()

        flag = ""
        if micro_f1 > best_f1:
            best_f1     = micro_f1
            best_preds  = all_preds
            best_labels = all_labels
            torch.save(model.state_dict(), f"{OUT_DIR}/best_seresnet.pt")
            flag = "  ← best"

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:6d}  {train_loss:10.4f}  {val_loss:8.4f}  "
              f"{micro_f1:8.4f}  {lr_now:8.1e}{flag}")

    print(f"\nMeilleur micro F1 : {best_f1:.4f}  →  {OUT_DIR}/best_seresnet.pt")

    # 6. Export CSV validation
    export_predictions(
        all_preds    = best_preds,
        all_labels   = best_labels,
        val_ds       = val_ds,
        full_dataset = dataset,
        idx2species  = idx2species,
        out_path     = f"{OUT_DIR}/predictions_val.csv",
    )