import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from Landsat_resnet6 import LandsatCubeDataset, SEResNet, load_labels


# =============================================================================
# EXPORT KAGGLE SUBMISSION
#
# Garanties :
#   - Exactement 14 784 lignes (une par surveyId du fichier de métadonnées test)
#   - Sites sans cube → predictions = ""   (ligne vide mais présente)
#   - Sites avec cube → species prédites space-separated
#   - Format : surveyId,predictions
# =============================================================================

def export_kaggle_submission(
    model,
    test_dataset        : LandsatCubeDataset,
    all_test_survey_ids : list,          # liste exhaustive des 14 784 surveyId attendus
    idx2species         : dict,
    device,
    threshold   : float = 0.5,
    out_path    : str   = "res_landsat/submission.csv",
    batch_size  : int   = 128,
    num_workers : int   = 4,
) -> pd.DataFrame:
    """
    Args:
        model                : SEResNet entraîné
        test_dataset         : LandsatCubeDataset sur PA-test (labels=None)
        all_test_survey_ids  : tous les surveyId du CSV de métadonnées test
                               (pour garantir 14 784 lignes même si un cube manque)
        idx2species          : {class_index -> speciesId original}
        threshold            : seuil sigmoid (0.5 par défaut ; à optimiser sur val)
    """
    loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    model.eval()

    # Prédictions pour les cubes disponibles
    predictions_by_sid = {}  # type: dict

    with torch.no_grad():
        offset = 0
        for batch in loader:
            # Dataset en mode test retourne un tuple à 1 élément : (cube,)
            cubes = batch[0] if isinstance(batch, (list, tuple)) else batch
            cubes = cubes.to(device)

            probs = torch.sigmoid(model(cubes))      # (B, num_classes)
            preds = probs >= threshold               # bool (B, num_classes)

            for i in range(len(cubes)):
                global_idx = offset + i
                survey_id  = test_dataset._sid(test_dataset.files[global_idx])
                species_ids = [
                    str(int(idx2species[k]))
                    for k in preds[i].nonzero(as_tuple=True)[0].tolist()
                ]
                predictions_by_sid[survey_id] = " ".join(species_ids)

            offset += len(cubes)

    # Construction du DataFrame final sur l'ensemble exhaustif des surveyId
    # Les sites sans cube reçoivent une prédiction vide (chaîne "").
    rows = [
        {
            "surveyId"   : sid,
            "predictions": predictions_by_sid.get(int(sid), ""),
        }
        for sid in all_test_survey_ids
    ]
    df = pd.DataFrame(rows)[["surveyId", "predictions"]]

    # Vérifications
    assert len(df) == len(all_test_survey_ids), (
        f"Nombre de lignes incorrect : {len(df)} ≠ {len(all_test_survey_ids)}"
    )
    assert df["surveyId"].nunique() == len(df), "surveyId dupliqués détectés."

    n_empty    = (df["predictions"] == "").sum()
    n_no_cube  = len(all_test_survey_ids) - len(predictions_by_sid)
    n_no_pred  = n_empty - n_no_cube      # cubes présents mais aucune espèce prédite

    df.to_csv(out_path, index=False)
    print(f"Soumission exportée → {out_path}")
    print(f"  Lignes totales    : {len(df):,}")
    print(f"  Sites sans cube   : {n_no_cube:,}")
    print(f"  Sites sans préd.  : {n_no_pred:,}  "
          f"({'⚠ seuil trop haut ?' if n_no_pred > 0.05 * len(df) else 'OK'})")
    print(f"\nAperçu :")
    print(df.head(8).to_string(index=False))

    return df


# =============================================================================
# OPTIMISATION DU SEUIL SUR VALIDATION (optionnel mais recommandé)
# Cherche le seuil qui maximise le micro F1 sur l'ensemble de validation.
# =============================================================================

@torch.no_grad()
def find_best_threshold(
    model,
    val_loader,
    device,
    thresholds=None,
):
    if thresholds is None:
        thresholds = [i / 100 for i in range(20, 80, 5)]   # 0.20 … 0.75

    model.eval()
    all_probs, all_labels = [], []

    for cubes, labels in val_loader:
        cubes = cubes.to(device)
        probs = torch.sigmoid(model(cubes)).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs  = torch.cat(all_probs,  dim=0)   # (N, C)
    all_labels = torch.cat(all_labels, dim=0)   # (N, C)

    best_t, best_f1 = 0.5, 0.0
    print("\nOptimisation du seuil :")
    print(f"{'Seuil':>8}  {'micro_F1':>10}")
    for t in thresholds:
        preds = all_probs >= t
        tp = (preds &  all_labels.bool()).float().sum(dim=0)
        fp = (preds & ~all_labels.bool()).float().sum(dim=0)
        fn = (~preds &  all_labels.bool()).float().sum(dim=0)
        sum_tp = tp.sum().item(); sum_fp = fp.sum().item(); sum_fn = fn.sum().item()
        prec = sum_tp / (sum_tp + sum_fp + 1e-8)
        rec  = sum_tp / (sum_tp + sum_fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        marker = "  ← best" if f1 > best_f1 else ""
        print(f"{t:8.2f}  {f1:10.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nMeilleur seuil : {best_t:.2f}  →  micro F1 = {best_f1:.4f}")
    return best_t


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Chemins ───────────────────────────────────────────────────────────────
    CSV_TRAIN    = "/data/challenge2026MIASHS/GLC25_PA_metadata_train.csv"
    CSV_TEST     = "/data/challenge2026MIASHS/GLC25_PA_metadata_test.csv"   # ← métadonnées test
    DATA_TEST    = "/data/challenge2026MIASHS/SateliteTimeSeries-Landsat/cubes/PA-test"
    DATA_TRAIN   = "/data/challenge2026MIASHS/SateliteTimeSeries-Landsat/cubes/PA-train"
    CKPT         = "res_landsat/best_seresnet.pt"
    OUT_DIR      = "res_landsat"
    os.makedirs(OUT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}\n")

    # ── 1. Récupération de idx2species depuis les labels d'entraînement ───────
    labels, species2idx, idx2species, num_classes = load_labels(CSV_TRAIN)

    # ── 2. Liste exhaustive des 14 784 surveyId de test ───────────────────────
    df_test = pd.read_csv(CSV_TEST)
    # Adapter le nom de colonne si nécessaire (surveyId ou survey_id)
    sid_col = "surveyId" if "surveyId" in df_test.columns else df_test.columns[0]
    all_test_survey_ids = df_test[sid_col].tolist()
    print(f"surveyId test attendus : {len(all_test_survey_ids):,}")
    assert len(all_test_survey_ids) == 14784, (
        f"Nombre de surveyId test inattendu : {len(all_test_survey_ids)}"
    )

    # ── 3. Dataset test (sans labels) ─────────────────────────────────────────
    test_dataset = LandsatCubeDataset(root_dir=DATA_TEST, labels=None)

    # ── 4. Modèle ─────────────────────────────────────────────────────────────
    model = SEResNet(num_classes=num_classes, base_ch=64, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f"Checkpoint chargé : {CKPT}\n")

    # ── 5. (Optionnel) Optimisation du seuil sur le jeu de validation ─────────
    # Décommenter si vous voulez chercher le seuil optimal.
    #
    # from Landsat_resnet6 import make_dataloaders
    # train_ds_full = LandsatCubeDataset(root_dir=DATA_TRAIN, labels=labels)
    # _, val_loader, _ = make_dataloaders(train_ds_full, val_ratio=0.15, batch_size=128)
    # THRESHOLD = find_best_threshold(model, val_loader, DEVICE)

    THRESHOLD = 0.5      # ← remplacer par la valeur trouvée ci-dessus si optimisé

    # ── 6. Export soumission ──────────────────────────────────────────────────
    export_kaggle_submission(
        model                = model,
        test_dataset         = test_dataset,
        all_test_survey_ids  = all_test_survey_ids,
        idx2species          = idx2species,
        device               = DEVICE,
        threshold            = THRESHOLD,
        out_path             = f"{OUT_DIR}/submission.csv",
    )