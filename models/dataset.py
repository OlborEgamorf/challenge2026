"""
GLC25MultiModalDataset
======================
Dataset combinant :
  - patches satellites TIFF 4 bandes (R, G, B, NIR)  →  shape (4, 64, 64)
  - variables environnementales tabulaires             →  shape (N_TAB_FEATURES,)
  - labels multi-hot (présence/absence d'espèces)     →  shape (NUM_CLASSES,)

Utilisation :
    from models.dataset import build_datasets
    train_ds, val_ds, meta = build_datasets(cfg)
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
    où AB = chiffres 3-4 depuis la droite, CD = 2 derniers chiffres.
    Ex : 3018575  →  ./75/85/3018575.tiff
    """
    s = str(survey_id)
    cd = s[-2:] if len(s) >= 2 else s
    ab = s[-4:-2] if len(s) >= 4 else (s[:-2] if len(s) > 2 else (s[0] if len(s) == 3 else ""))
    return os.path.join(base_dir, cd, ab, f"{survey_id}.tiff")


def load_tiff(path: str) -> np.ndarray:
    """Charge un TIFF 4 bandes, remplace les nodata (-1) par 0, retourne float32."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (4, 64, 64)
    arr[arr == -1.0] = 0.0
    return arr


# ─────────────────────────────────────────────
# Construction des features tabulaires
# ─────────────────────────────────────────────

TABULAR_SOURCES = {
    "bioclim":   "ClimateAverage_1981-2010/GLC25-PA-train-bioclimatic.csv",
    "elevation": "Elevation/GLC25-PA-train-elevation.csv",
    "footprint": "HumanFootprint/GLC25-PA-train-human_footprint.csv",
    "landcover": "LandCover/GLC25-PA-train-landcover.csv",
    "soilgrids": "SoilGrids/GLC25-PA-train-soilgrids.csv",
}

TABULAR_SOURCES_TEST = {
    "bioclim":   "ClimateAverage_1981-2010/GLC25-PA-test-bioclimatic.csv",
    "elevation": "Elevation/GLC25-PA-test-elevation.csv",
    "footprint": "HumanFootprint/GLC25-PA-test-human_footprint.csv",
    "landcover": "LandCover/GLC25-PA-test-landcover.csv",
    "soilgrids": "SoilGrids/GLC25-PA-test-soilgrids.csv",
}


def load_tabular(env_dir: str, split: str = "train") -> pd.DataFrame:
    """
    Fusionne tous les CSV environnementaux en un seul DataFrame indexé par surveyId.
    Les NaN (surtout soilgrids ~78%) sont imputés par la médiane de la colonne de train.
    """
    sources = TABULAR_SOURCES if split in ("train", "val") else TABULAR_SOURCES_TEST
    merged = None
    for name, rel_path in sources.items():
        path = os.path.join(env_dir, rel_path)
        df = pd.read_csv(path)
        # Nettoyer les colonnes parasites
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="surveyId", how="inner")
    return merged.set_index("surveyId")


def build_tabular_stats(tab_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (median, std) sur le jeu train pour normalisation z-score (après imputation médiane)."""
    median = tab_df.median().values.astype(np.float32)
    # Imputer avant de calculer std
    filled = tab_df.fillna(tab_df.median())
    std = filled.std().values.astype(np.float32)
    std[std == 0] = 1.0   # éviter division par zéro
    return median, std


# ─────────────────────────────────────────────
# Construction des labels multi-hot
# ─────────────────────────────────────────────

def build_labels(meta_df: pd.DataFrame, species_list: List[int]) -> Dict[int, np.ndarray]:
    """
    Retourne un dict  surveyId → vecteur multi-hot float32 (NUM_CLASSES,).
    species_list : liste ordonnée des speciesId (index 0..NUM_CLASSES-1).
    """
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
# Dataset principal
# ─────────────────────────────────────────────

class GLC25MultiModalDataset(Dataset):
    """
    Dataset multi-modal pour GeoLifeCLEF 2025 (données PA uniquement).

    Paramètres
    ----------
    survey_ids    : liste des surveyId à inclure dans ce split
    labels        : dict surveyId → vecteur multi-hot (None pour le test)
    tab_df        : DataFrame tabulaire indexé par surveyId
    tab_median    : médiane par colonne (pour imputation NaN)
    tab_mean      : moyenne par colonne (pour z-score)
    tab_std       : écart-type par colonne (pour z-score)
    patches_dir   : dossier racine des TIFF (ex. data/SatelitePatches/PA-train)
    augment       : si True, applique flip aléatoire (train seulement)
    """

    def __init__(
        self,
        survey_ids: List[int],
        labels: Optional[Dict[int, np.ndarray]],
        tab_df: pd.DataFrame,
        tab_median: np.ndarray,
        tab_mean: np.ndarray,
        tab_std: np.ndarray,
        patches_dir: str,
        augment: bool = False,
    ):
        self.survey_ids  = survey_ids
        self.labels      = labels
        self.tab_df      = tab_df
        self.tab_median  = tab_median
        self.tab_mean    = tab_mean
        self.tab_std     = tab_std
        self.patches_dir = patches_dir
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.survey_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.survey_ids[idx]

        # ── Image ────────────────────────────────────────────────
        path = survey_to_tiff_path(sid, self.patches_dir)
        img = load_tiff(path)          # (4, 64, 64) float32
        img = self._normalize_image(img)

        if self.augment:
            img = self._augment(img)

        # ── Tabulaire ────────────────────────────────────────────
        if sid in self.tab_df.index:
            tab_raw = self.tab_df.loc[sid].values.astype(np.float32)
        else:
            tab_raw = self.tab_median.copy()

        # Imputation NaN puis z-score
        nan_mask = np.isnan(tab_raw)
        tab_raw[nan_mask] = self.tab_median[nan_mask]
        tab = (tab_raw - self.tab_mean) / self.tab_std

        # ── Label ─────────────────────────────────────────────────
        sample = {
            "image": torch.from_numpy(img),
            "tabular": torch.from_numpy(tab),
            "survey_id": torch.tensor(sid, dtype=torch.long),
        }
        if self.labels is not None:
            sample["label"] = torch.from_numpy(self.labels[sid])
        return sample

    # ── helpers ──────────────────────────────────────────────────

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalisation par bande : percentile 2-98 → [0, 1].
        Plus robuste que min-max face aux valeurs aberrantes Sentinel-2.
        """
        out = np.empty_like(img)
        for b in range(img.shape[0]):
            lo = np.percentile(img[b], 2)
            hi = np.percentile(img[b], 98)
            if hi > lo:
                out[b] = np.clip((img[b] - lo) / (hi - lo), 0.0, 1.0)
            else:
                out[b] = 0.0
        return out

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Flip horizontal + vertical aléatoire (invariance spatiale)."""
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()   # flip horizontal
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()   # flip vertical
        return img


# ─────────────────────────────────────────────
# Fonction de construction train/val
# ─────────────────────────────────────────────

def build_datasets(
    data_root: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple["GLC25MultiModalDataset", "GLC25MultiModalDataset", dict]:
    """
    Construit les datasets train et val à partir des données PA.

    Retourne
    --------
    train_ds, val_ds, meta
      meta contient : species_list, num_classes, n_tab_features, tab_mean, tab_std
    """
    # Chemins
    meta_path   = "../data/GLC25_PA_metadata_train.csv"
    env_dir     = os.path.join(data_root, "data", "EnvironmentalValues")
    patches_dir = "../../../data/challenge2026MIASHS/PA/PA-train"

    # Métadonnées
    print("Chargement des métadonnées PA...")
    meta_df = pd.read_csv(meta_path)
    meta_df["speciesId"] = meta_df["speciesId"].dropna().astype(int)

    # Liste des espèces
    species_list = sorted(meta_df["speciesId"].dropna().unique().tolist())
    num_classes  = len(species_list)
    print(f"  {num_classes} espèces | {meta_df['surveyId'].nunique()} surveys")

    # Tabulaire
    print("Chargement des features tabulaires...")
    tab_df = load_tabular(env_dir, split="train")
    tab_median, tab_std = build_tabular_stats(tab_df)
    tab_mean  = tab_df.fillna(tab_df.median()).mean().values.astype(np.float32)
    n_tab     = tab_df.shape[1]
    print(f"  {n_tab} features tabulaires")

    # Labels multi-hot
    print("Construction des labels multi-hot...")
    labels = build_labels(meta_df, species_list)

    # Split train / val (par surveyId, pas par ligne)
    all_ids = sorted(labels.keys())
    rng     = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n_val   = int(len(all_ids) * val_fraction)
    val_ids   = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"  Train: {len(train_ids)} surveys | Val: {len(val_ids)} surveys")

    train_ds = GLC25MultiModalDataset(
        train_ids, labels, tab_df, tab_median, tab_mean, tab_std,
        patches_dir, augment=True,
    )
    val_ds = GLC25MultiModalDataset(
        val_ids, labels, tab_df, tab_median, tab_mean, tab_std,
        patches_dir, augment=False,
    )

    meta = {
        "species_list": species_list,
        "num_classes":  num_classes,
        "n_tab_features": n_tab,
        "tab_mean": tab_mean,
        "tab_std":  tab_std,
        "tab_median": tab_median,
    }
    return train_ds, val_ds, meta
