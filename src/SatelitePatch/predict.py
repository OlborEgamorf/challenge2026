"""
Génération de la soumission GLC25
===================================

Usage (serveur)
---------------
python models/predict.py --checkpoint checkpoints/glc25-epoch=XX-val_F1=0.XXXX.ckpt

Usage (Mac)
-----------
python3.11 models/predict.py --checkpoint checkpoints/mac/glc25-mac-epoch=XX-val_F1=0.XXXX.ckpt

Format de sortie (identique à GLC25_SAMPLE_SUBMISSION.csv) :
    surveyId,predictions
    642,3301 7301 2436 4600 ...
    1792,2564 4888 6082 5015 ...
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.SatelitePatch.dataset import GLC25ImageDataset, survey_to_tiff_path
from src.SatelitePatch.model import GLC25ImageSystem
from src.SatelitePatch.model_swin import GLC25SwinSystem


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
# Dataset test
# ─────────────────────────────────────────────────────────────────────────────

def find_data_root(script_root: str) -> str:
    """Trouve le dossier data/ en testant plusieurs emplacements (Mac / serveur)."""
    candidates = [
        os.path.join(script_root, "data"),
        os.path.join(script_root, "..", "..", "data", "challenge2026MIASHS"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Dossier data/ introuvable. Chemins testés : {candidates}")


def build_test_dataset(data_root: str, image_size: int = 64) -> GLC25ImageDataset:
    """Construit le dataset PA-test (sans labels)."""
    data_dir    = find_data_root(data_root)
    meta        = pd.read_csv(os.path.join(data_dir, "GLC25_PA_metadata_test.csv"))
    survey_ids  = sorted(meta["surveyId"].unique().tolist())
    patches_dir = os.path.join(data_dir, "SatelitePatches", "PA-test")
    return GLC25ImageDataset(
        survey_ids=survey_ids,
        labels=None,
        patches_dir=patches_dir,
        augment=False,
        image_size=image_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataloader, device) -> tuple:
    """Retourne (survey_ids, probs) — probs : (N_test, num_classes) float32."""
    model = model.to(device).eval()
    all_ids, all_probs = [], []

    for batch in dataloader:
        image = batch["image"].to(device)
        probs = torch.sigmoid(model(image)).cpu().numpy()
        all_ids.append(batch["survey_id"].numpy())
        all_probs.append(probs)

    return np.concatenate(all_ids), np.vstack(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Construction de la soumission
# ─────────────────────────────────────────────────────────────────────────────

def build_submission(
    survey_ids: np.ndarray,
    probs: np.ndarray,
    species_list: list,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Format attendu :
        surveyId,predictions
        642,3301 7301 2436 ...

    Stratégie : toutes les espèces dont prob > threshold.
    Fallback : si aucune espèce ne dépasse le seuil → prendre la plus probable.
    """
    species_arr = np.array(species_list, dtype=int)
    rows = []
    for sid, prob in zip(survey_ids, probs):
        mask = prob >= threshold
        pred = species_arr[mask]
        if len(pred) == 0:
            pred = species_arr[[np.argmax(prob)]]
        rows.append({
            "surveyId":    int(sid),
            "predictions": " ".join(map(str, pred)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Génération soumission GLC25")
    p.add_argument("--checkpoint", required=True,
                   help="Chemin vers le .ckpt (ex: checkpoints/glc25-epoch=27-val_F1=0.0599.ckpt)")
    p.add_argument("--threshold",  type=float, default=0.3,
                   help="Seuil sigmoid pour prédire une espèce (défaut 0.3)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers",    type=int, default=8)
    p.add_argument("--gpu",        type=int, default=None,
                   help="Index GPU (défaut: auto-sélection du GPU le plus libre)")
    p.add_argument("--model",      type=str, default="resnet", choices=["resnet", "swin"],
                   help="Architecture du checkpoint : resnet (défaut) ou swin")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Appareil
    if torch.cuda.is_available():
        gpu_idx = args.gpu if args.gpu is not None else select_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # Modèle
    print(f"[Checkpoint] {args.checkpoint}")
    ModelClass = GLC25SwinSystem if args.model == "swin" else GLC25ImageSystem
    model = ModelClass.load_from_checkpoint(args.checkpoint, map_location=device)

    # Liste des espèces (depuis les métadonnées train)
    _data_dir  = find_data_root(ROOT)
    meta_train = pd.read_csv(os.path.join(_data_dir, "GLC25_PA_metadata_train.csv"))
    meta_train["speciesId"] = meta_train["speciesId"].dropna().astype(int)
    species_list = sorted(meta_train["speciesId"].dropna().unique().tolist())
    print(f"[Espèces] {len(species_list)}")

    # Dataset test
    image_size = 224 if args.model == "swin" else 64
    test_ds = build_test_dataset(ROOT, image_size=image_size)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Test] {len(test_ds)} surveys")

    # Inférence
    print("Inférence en cours...")
    survey_ids, probs = run_inference(model, test_dl, device)

    # Soumission
    submission = build_submission(survey_ids, probs, species_list, threshold=args.threshold)

    out_dir  = os.path.join(ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_{ts}.csv")
    submission.to_csv(out_path, index=False)

    print(f"\n[OK] Soumission sauvegardée : {out_path}")
    print(f"     {len(submission)} surveys | threshold={args.threshold}")
    print(submission.head())
