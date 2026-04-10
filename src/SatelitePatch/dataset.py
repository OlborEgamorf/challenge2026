"""
GLC25ImageDataset
=================
Dataset image-only pour GeoLifeCLEF 2025 (données PA).

Chaque sample :
  - image  : TIFF 4 bandes (R, G, B, NIR) → tensor (4, 64, 64) float32 normalisé
  - label  : vecteur multi-hot float32 (NUM_CLASSES,)
  - survey_id : int

Utilisation :
    from models.dataset import build_datasets
    train_ds, val_ds, meta = build_datasets(data_root)
"""

import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# Utilitaires chemins TIFF
# ─────────────────────────────────────────────

def survey_to_tiff_path(survey_id: int, base_dir: str) -> str:
    """
    Règle officielle du challenge :
      …/{CD}/{AB}/{surveyId}.tiff
    AB = chiffres 3-4 depuis la droite, CD = 2 derniers chiffres.
    Ex : 3018575 → ./75/85/3018575.tiff
    """
    s = str(survey_id)
    cd = s[-2:] if len(s) >= 2 else s
    ab = s[-4:-2] if len(s) >= 4 else (s[:-2] if len(s) > 2 else (s[0] if len(s) == 3 else ""))
    return os.path.join(base_dir, cd, ab, f"{survey_id}.tiff")


def load_tiff(path: str) -> np.ndarray:
    """Charge un TIFF 4 bandes, remplace nodata (-1) par 0, retourne float32 (4,64,64)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr[arr == -1.0] = 0.0
    return arr


# ─────────────────────────────────────────────
# Construction des labels multi-hot
# ─────────────────────────────────────────────

def build_labels(meta_df: pd.DataFrame, species_list: List[int]) -> Dict[int, np.ndarray]:
    """Retourne dict surveyId → vecteur multi-hot float32 (NUM_CLASSES,)."""
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    n_classes = len(species_list)
    labels: Dict[int, np.ndarray] = {}
    for sid, group in meta_df.groupby("surveyId"):
        vec = np.zeros(n_classes, dtype=np.float32)
        for sp in group["speciesId"].dropna():
            idx = sp2idx.get(int(sp))
            if idx is not None:
                vec[idx] = 1.0
        labels[int(sid)] = vec
    return labels


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class GLC25ImageDataset(Dataset):
    """
    Dataset image-only pour GeoLifeCLEF 2025.

    Paramètres
    ----------
    survey_ids  : liste des surveyId à inclure
    labels      : dict surveyId → vecteur multi-hot (None pour le jeu test)
    patches_dir : dossier racine des TIFF
    augment     : flip aléatoire H/V (activer pour train seulement)
    """

    def __init__(
        self,
        survey_ids: List[int],
        labels: Optional[Dict[int, np.ndarray]],
        patches_dir: str,
        augment: bool = False,
        norm: str = "percentile",   # "percentile" ou "zscore"
        image_size: int = 64,       # 64 (ResNet) ou 224 (Swin)
    ):
        self.survey_ids  = survey_ids
        self.labels      = labels
        self.patches_dir = patches_dir
        self.augment     = augment
        self.norm        = norm
        self.image_size  = image_size

    def __len__(self) -> int:
        return len(self.survey_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid  = self.survey_ids[idx]
        path = survey_to_tiff_path(sid, self.patches_dir)

        img = load_tiff(path)             # (4, 64, 64) float32
        img = self._normalize(img)

        if self.augment:
            img = self._augment(img)

        t = torch.from_numpy(img)         # (4, 64, 64)
        if self.image_size != 64:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        sample = {
            "image":     t,
            "survey_id": torch.tensor(sid, dtype=torch.long),
        }
        if self.labels is not None:
            sample["label"] = torch.from_numpy(self.labels[sid])
        return sample

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalisation par bande : percentile 2-98 → [0,1] ou Z-score → µ=0 σ=1."""
        out = np.empty_like(img)
        for b in range(img.shape[0]):
            if self.norm == "zscore":
                mu  = img[b].mean()
                std = img[b].std()
                out[b] = (img[b] - mu) / (std + 1e-6)
            else:  # percentile (défaut)
                lo = np.percentile(img[b], 2)
                hi = np.percentile(img[b], 98)
                out[b] = np.clip((img[b] - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(img[b])
        return out

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()   # flip horizontal
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()   # flip vertical
        return img


# ─────────────────────────────────────────────
# Factory train / val
# ─────────────────────────────────────────────

def spatial_block_split(
    meta_df: pd.DataFrame,
    val_fraction: float = 0.2,
    block_size_deg: float = 0.09,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Découpage spatial : assigne chaque survey à une cellule de grille ~10×10 km,
    puis réserve val_fraction des blocs pour la validation.

    Évite l'autocorrélation spatiale : deux surveys dans le même bloc
    vont tous les deux en train ou tous les deux en val.

    Paramètres
    ----------
    block_size_deg : taille d'un bloc en degrés (~0.09° ≈ 10 km en latitude)
    """
    surveys = meta_df.drop_duplicates("surveyId")[["surveyId", "lat", "lon"]].copy()
    surveys["block_lat"] = (surveys["lat"] / block_size_deg).apply(np.floor).astype(int)
    surveys["block_lon"] = (surveys["lon"] / block_size_deg).apply(np.floor).astype(int)
    surveys["block_id"]  = list(zip(surveys["block_lat"], surveys["block_lon"]))

    blocks = sorted(surveys["block_id"].unique().tolist())
    rng    = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n_val_blocks = max(1, int(len(blocks) * val_fraction))
    val_blocks   = set(map(tuple, blocks[:n_val_blocks]))

    val_ids   = surveys[surveys["block_id"].apply(lambda b: b in val_blocks)]["surveyId"].tolist()
    train_ids = surveys[surveys["block_id"].apply(lambda b: b not in val_blocks)]["surveyId"].tolist()

    print(f"  Blocs totaux: {len(blocks)} | Blocs val: {n_val_blocks} | Blocs train: {len(blocks) - n_val_blocks}")
    return train_ids, val_ids


def build_datasets(
    data_root: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_surveys: Optional[int] = None,
    norm: str = "percentile",
    image_size: int = 64,
) -> Tuple["GLC25ImageDataset", "GLC25ImageDataset", dict]:
    """
    Construit les datasets train et val à partir des données PA.

    Le découpage train/val est spatial (blocs ~10×10 km) pour éviter
    l'autocorrélation spatiale, conformément au protocole du challenge GLC25.

    Paramètres
    ----------
    max_surveys : si fourni, limite le nombre de surveys (utile pour test rapide sur CPU/MPS)

    Retourne
    --------
    train_ds, val_ds, meta
      meta : species_list, num_classes
    """
    # Cherche le dossier data/ : Mac local ou serveur (/data/challenge2026MIASHS)
    _candidates = [
        os.path.join(data_root, "data"),
        os.path.join(data_root, "..", "..", "data", "challenge2026MIASHS"),
    ]
    _data_dir = next((p for p in _candidates if os.path.isdir(p)), None)
    if _data_dir is None:
        raise FileNotFoundError(f"Dossier data/ introuvable. Chemins testés : {_candidates}")
    _data_dir = os.path.normpath(_data_dir)

    meta_path   = os.path.join(_data_dir, "GLC25_PA_metadata_train.csv")
    patches_dir = os.path.join(_data_dir, "SatelitePatches", "PA-train")

    print("Chargement des métadonnées PA...")
    meta_df = pd.read_csv(meta_path)
    meta_df["speciesId"] = meta_df["speciesId"].dropna().astype(int)

    species_list = sorted(meta_df["speciesId"].dropna().unique().tolist())
    num_classes  = len(species_list)
    print(f"  {num_classes} espèces | {meta_df['surveyId'].nunique()} surveys")

    print("Construction des labels multi-hot...")
    labels = build_labels(meta_df, species_list)

    print("Découpage spatial train/val (blocs ~10×10 km)...")
    train_ids, val_ids = spatial_block_split(meta_df, val_fraction=val_fraction, seed=seed)

    # Ne garder que les surveys qui ont un label (image disponible)
    train_ids = [sid for sid in train_ids if sid in labels]
    val_ids   = [sid for sid in val_ids   if sid in labels]

    if max_surveys is not None:
        rng       = np.random.default_rng(seed)
        all_ids   = train_ids + val_ids
        all_ids   = rng.choice(all_ids, size=min(max_surveys, len(all_ids)), replace=False).tolist()
        n_val     = int(len(all_ids) * val_fraction)
        val_ids   = all_ids[:n_val]
        train_ids = all_ids[n_val:]
        print(f"  [subset] Limité à {max_surveys} surveys")

    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_ds = GLC25ImageDataset(train_ids, labels, patches_dir, augment=True,  norm=norm, image_size=image_size)
    val_ds   = GLC25ImageDataset(val_ids,   labels, patches_dir, augment=False, norm=norm, image_size=image_size)

    meta = {"species_list": species_list, "num_classes": num_classes}
    return train_ds, val_ds, meta
