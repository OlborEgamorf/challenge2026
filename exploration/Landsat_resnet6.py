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
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

def load_labels(csv_path: str):
    """
    Lit le CSV de métadonnées PA et construit les labels multi-hot.

    Colonnes utilisées : surveyId, speciesId
    Une ligne = une observation → plusieurs lignes par site.

    Returns:
        labels      : dict {surveyId (int) -> Tensor float32 (num_classes,)}
        species2idx : dict {speciesId (int) -> class_index (int)}
        num_classes : int
        pos_weight  : Tensor float32 (num_classes,)
    """
    df = pd.read_csv(csv_path)

    # 1. Encodage des espèces en indices contigus 0..C-1
    unique_species = sorted(df["speciesId"].unique())
    species2idx    = {sp: i for i, sp in enumerate(unique_species)}
    num_classes    = len(unique_species)

    print(f"Espèces uniques   : {num_classes:,}")
    print(f"Observations CSV  : {len(df):,}")
    print(f"Sites uniques     : {df['surveyId'].nunique():,}")

    # 2. Construction des vecteurs multi-hot par site
    labels = {}
    for survey_id, group in df.groupby("surveyId"):
        vec = torch.zeros(num_classes, dtype=torch.float32)
        for sp in group["speciesId"]:
            vec[species2idx[sp]] = 1.0
        labels[int(survey_id)] = vec

    # 3. pos_weight = n_négatifs / n_positifs par espèce
    #    Rééquilibre la BCE sur les espèces rares (présentes sur peu de sites)
    all_vecs   = torch.stack(list(labels.values()))       # (N_sites, C)
    n_sites    = all_vecs.shape[0]
    n_pos      = all_vecs.sum(dim=0).clamp(min=1)         # évite division par 0
    n_neg      = n_sites - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=100.0)         # cap à 100 pour stabilité numérique

    print(f"pos_weight        : min={pos_weight.min():.1f} | "
          f"median={pos_weight.median():.1f} | max={pos_weight.max():.1f}")

    return labels, species2idx, num_classes, pos_weight


# =============================================================================
# DATASET
# =============================================================================

class LandsatCubeDataset(Dataset):
    """
    Charge les cubes .pt Landsat (6, 4, ≤21) et renvoie (cube, label_multihot).
    Filtre automatiquement les cubes sans label dans le CSV.
    """
    def __init__(self, root_dir: str, labels: dict, transform=None):
        all_files  = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.files = [f for f in all_files if self._sid(f) in labels]

        if not self.files:
            raise ValueError("Aucun cube ne correspond aux surveyId du CSV.")

        self.labels    = labels
        self.transform = transform
        print(f"Cubes disponibles : {len(all_files):,}")
        print(f"Cubes avec labels : {len(self.files):,}")

    @staticmethod
    def _sid(path: str) -> int:
        # Pattern filename : ..._<surveyId>_cube.pt
        return int(os.path.basename(path).split("_")[-2])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cube = torch.load(self.files[idx], map_location="cpu").float()  # (6, 4, ≤21)

        # Padding de la dimension année à 21 si la série est incomplète
        if cube.shape[2] < 21:
            pad  = torch.zeros(6, 4, 21 - cube.shape[2])
            cube = torch.cat([cube, pad], dim=2)

        if self.transform:
            cube = self.transform(cube)

        return cube, self.labels[self._sid(self.files[idx])]


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
    return train_loader, val_loader


# =============================================================================
# MODÈLE ResNet-6 2D
# =============================================================================

class ResBlock2D(nn.Module):
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class ResNet6_2D(nn.Module):
    """
    Input  : (B, 6, 4, 21)  — bandes × (trimestres × années)
    Output : (B, num_classes) — logits bruts (sigmoid appliqué à l'inférence)
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock2D(64, kernel_size=3, dropout=dropout)
        self.block2 = ResBlock2D(64, kernel_size=3, dropout=dropout)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.block2(self.block1(self.stem(x)))))


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(cubes), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * cubes.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        total += criterion(model(cubes), labels).item() * cubes.size(0)
    return total / len(loader.dataset)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    CSV_PATH = "/data/challenge2026MIASHS/GLC25_PA_metadata_train.csv"
    DATA_DIR = "/data/challenge2026MIASHS/SateliteTimeSeries-Landsat/cubes/PA-train"
    EPOCHS   = 30
    BATCH    = 64
    LR       = 3e-4
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {DEVICE}\n")

    # ── 1. Labels depuis le CSV réel ──────────────────────────────────────────
    labels, species2idx, num_classes, pos_weight = load_labels(CSV_PATH)

    # ── 2. Dataset + split ────────────────────────────────────────────────────
    dataset = LandsatCubeDataset(root_dir=DATA_DIR, labels=labels)
    train_loader, val_loader = make_dataloaders(dataset, val_ratio=0.15, batch_size=BATCH)

    # ── 3. Modèle ─────────────────────────────────────────────────────────────
    model = ResNet6_2D(num_classes=num_classes, dropout=0.2).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres        : {n_p:,}\n")

    # ── 4. Loss BCE avec pos_weight calculé depuis les vrais labels ───────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    # ── 5. Optimiseur + scheduler cosine ──────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 6. Entraînement ───────────────────────────────────────────────────────
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_resnet6_2d.pt")
            flag = "  ← best"

        print(f"Epoch {epoch:3d}/{EPOCHS}  |  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}{flag}")

    print(f"\nMeilleure val_loss : {best_val:.4f}  →  best_resnet6_2d.pt")

    # ── Inférence ─────────────────────────────────────────────────────────────
    # model.load_state_dict(torch.load("best_resnet6_2d.pt"))
    # probs   = torch.sigmoid(model(cube.unsqueeze(0).to(DEVICE)))  # (1, num_classes)
    # present = (probs > 0.5).nonzero(as_tuple=True)[1]             # indices espèces présentes
    # species = [idx2species[i.item()] for i in present]            # retour aux vrais IDs