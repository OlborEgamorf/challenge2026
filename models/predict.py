"""
Inférence et génération du fichier de soumission GLC25
=======================================================

Usage
-----
python models/predict.py --checkpoint checkpoints/glc25-last.ckpt

Le script génère  submissions/submission_YYYYMMDD_HHMMSS.csv
au format attendu par le challenge.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.dataset import (
    GLC25MultiModalDataset,
    load_tabular,
    build_tabular_stats,
    survey_to_tiff_path,
    TABULAR_SOURCES_TEST,
)
from models.model import GLC25FusionSystem


# ─────────────────────────────────────────────────────────────────────────────
# Dataset test
# ─────────────────────────────────────────────────────────────────────────────

def build_test_dataset(
    data_root: str,
    tab_mean: np.ndarray,
    tab_std: np.ndarray,
    tab_median: np.ndarray,
    train_tab_df: pd.DataFrame,
) -> GLC25MultiModalDataset:
    """Construit le dataset de test (PA-test) sans labels."""
    meta_test = pd.read_csv(
        os.path.join(data_root, "data", "GLC25_PA_metadata_test.csv")
    )
    survey_ids = sorted(meta_test["surveyId"].unique().tolist())

    env_dir = os.path.join(data_root, "data", "EnvironmentalValues")
    tab_test_dfs = []
    for name, rel_path in TABULAR_SOURCES_TEST.items():
        path = os.path.join(env_dir, rel_path)
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        tab_test_dfs.append(df.set_index("surveyId"))

    # Merge
    tab_test = tab_test_dfs[0]
    for df in tab_test_dfs[1:]:
        tab_test = tab_test.join(df, how="inner")

    # Réaligner colonnes sur celles du train
    for col in train_tab_df.columns:
        if col not in tab_test.columns:
            tab_test[col] = np.nan
    tab_test = tab_test[train_tab_df.columns]

    patches_dir = os.path.join(data_root, "data", "SatelitePatches", "PA-test")

    return GLC25MultiModalDataset(
        survey_ids=survey_ids,
        labels=None,
        tab_df=tab_test,
        tab_median=tab_median,
        tab_mean=tab_mean,
        tab_std=tab_std,
        patches_dir=patches_dir,
        augment=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: GLC25FusionSystem,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Retourne (survey_ids, probs) — probs : (N_test, num_classes) float32."""
    model = model.to(device)
    model.eval()

    all_ids   = []
    all_probs = []

    for batch in dataloader:
        image   = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        sids    = batch["survey_id"].cpu().numpy()

        logits = model(image, tabular)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_ids.append(sids)
        all_probs.append(probs)

    all_ids   = np.concatenate(all_ids)
    all_probs = np.vstack(all_probs)   # (N_test, num_classes)

    return all_ids, all_probs


def build_submission(
    survey_ids: np.ndarray,
    probs: np.ndarray,
    species_list: list,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Construit le CSV de soumission.
    Chaque ligne : surveyId, predictions (speciesId dont prob > threshold, séparés par espaces).
    Un seuil de 0.3 est utilisé (plus permissif que 0.5, améliore le rappel sur le F1 sample).
    """
    species_arr = np.array(species_list)
    rows = []
    for sid, prob in zip(survey_ids, probs):
        mask    = prob >= threshold
        pred_sp = species_arr[mask].astype(int)
        # Fallback : si aucune espèce ne dépasse le seuil, prendre la plus probable
        if len(pred_sp) == 0:
            pred_sp = species_arr[[np.argmax(prob)]].astype(int)
        rows.append({
            "surveyId":    int(sid),
            "predictions": " ".join(map(str, pred_sp)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Chemin vers le .ckpt")
    p.add_argument("--threshold",  type=float, default=0.3,
                   help="Seuil sigmoid pour prédire une espèce (défaut 0.3)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers",    type=int, default=8)
    p.add_argument("--gpu",        type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Charger le modèle ────────────────────────────────────────────────
    print(f"Chargement du checkpoint : {args.checkpoint}")
    model = GLC25FusionSystem.load_from_checkpoint(args.checkpoint)

    # ── Reconstruire les stats tabulaires (nécessaires pour le dataset test)
    env_dir = os.path.join(ROOT, "data", "EnvironmentalValues")
    from models.dataset import load_tabular, build_tabular_stats
    train_tab = load_tabular(env_dir, split="train")
    tab_median, tab_std = build_tabular_stats(train_tab)
    tab_mean = train_tab.fillna(train_tab.median()).mean().values.astype("float32")

    # ── Récupérer la liste d'espèces depuis les métadonnées train ────────
    meta_train = pd.read_csv(os.path.join(ROOT, "data", "GLC25_PA_metadata_train.csv"))
    species_list = sorted(meta_train["speciesId"].dropna().unique().tolist())

    # ── Dataset et DataLoader test ───────────────────────────────────────
    test_ds = build_test_dataset(ROOT, tab_mean, tab_std, tab_median, train_tab)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"Surveys test : {len(test_ds)}")

    # ── Inférence ────────────────────────────────────────────────────────
    print("Inférence en cours...")
    survey_ids, probs = run_inference(model, test_dl, device)

    # ── Soumission ───────────────────────────────────────────────────────
    submission = build_submission(survey_ids, probs, species_list, threshold=args.threshold)

    out_dir = os.path.join(ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_{ts}.csv")
    submission.to_csv(out_path, index=False)

    print(f"\nSoumission sauvegardée : {out_path}")
    print(submission.head())
