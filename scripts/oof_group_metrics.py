"""
Out-of-fold predictions from saved K-fold ABMIL checkpoints. No retraining:
each slide is scored by the fold model that held it out, on exactly the same tiles as in training.

    from oof_group_metrics import collect_oof_predictions, group_metrics

    oof = collect_oof_predictions(df, filename_col="filename", label_col="label", group_col="tissue",
                                  feature_key=["name_a", "name_b"], zarr_dir=ZARR_DIR, checkpoint_dir="checkpoints/",
                                  patient_col="rekvnr")
    group_metrics(oof, "tissue") 
    # oof.attrs["configs"], oof.attrs["skipped"]: training config per fold, unreadable slides
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from abmil import ZarrSlideDataset, load_checkpoint, _find_fold_checkpoint


@torch.no_grad()
def collect_oof_predictions(df, filename_col, label_col, group_col, feature_key, zarr_dir,
                            checkpoint_dir, n_splits=5, patient_col=None):
    """One row per held-out slide: slide, fold, group, (patient), label (0/1), prob_pos, pred, n_tiles, feature_key.

    feature_key: one name, or a list of names for the same features stored under different names
    (e.g. ["uni_features", "UNI"]); each slide uses the first name it has.
    """
    keys = [feature_key] if isinstance(feature_key, str) else list(feature_key)
    meta = df.set_index(filename_col)
    if not meta.index.is_unique:
        raise ValueError(f"Duplicate values in '{filename_col}'; each slide must appear once.")

    rows, skipped, configs = [], [], {}
    for fold in range(1, n_splits + 1):
        with open(os.path.join(checkpoint_dir, f"fold_{fold}_test_slides.json")) as f:
            test_slides = json.load(f)

        ckpt_path = _find_fold_checkpoint(checkpoint_dir, fold)
        model, config, label_mapping = load_checkpoint(ckpt_path)
        configs[fold] = dict(config)
        device = next(model.parameters()).device

        # Rebuild the fold's test set exactly as in training:
        # same slides, same order, same label mapping, same tile subsampling (max_tiles + fold seed).
        test_df = meta.loc[test_slides].reset_index()
        test_df[label_col] = test_df[label_col].map(label_mapping)
        if test_df[label_col].isna().any():
            raise ValueError(f"Fold {fold}: labels not in checkpoint mapping {label_mapping}")

        # One dataset per key name; same df, seed and index, so the tile subsample is identical
        datasets = {key: ZarrSlideDataset(
            df=test_df,
            filename_col=filename_col,
            label_col=label_col,
            feature_key=key,
            zarr_dir=zarr_dir,
            max_tiles=config.get("max_tiles"),
            seed=config.get("random_state"),
        ) for key in keys}

        fold_rows = []
        # Iterate over the full saved list and skip unreadable slides: indices are unchanged,
        # so the remaining slides get exactly the same tile subsample as in training.
        for i in range(len(test_df)):
            feats, used_key, error = None, None, None
            for key, dataset in datasets.items():
                try:
                    feats, _, label = dataset[i]
                    used_key = key
                    break
                except KeyError as e:  # this slide has no table under this key name: try the next one
                    error = f"no feature table {list(keys)}: {e}"
                except Exception as e:  # unreadable slide: no point trying another key
                    error = f"{type(e).__name__}: {e}"
                    break
            if feats is None:
                skipped.append({"slide": test_df[filename_col].iloc[i], "fold": fold,
                                group_col: test_df[group_col].iloc[i], "error": error})
                continue
            if feats.shape[0] == 0:
                continue
            probs = torch.softmax(model(feats.to(device))["logits"].float(), dim=0)  # fp32, as in validate_ABMIL
            fold_rows.append({
                "slide": test_df[filename_col].iloc[i],
                "fold": fold,
                group_col: test_df[group_col].iloc[i],
                **({patient_col: test_df[patient_col].iloc[i]} if patient_col else {}),
                "label": int(label.item()),
                "prob_pos": float(probs[1].item()),
                "pred": int(probs.argmax().item()),
                "n_tiles": int(feats.shape[0]),  # tiles the model saw (capped at max_tiles)
                "feature_key": used_key,
            })

        # Sanity check: equals the AUC in the checkpoint filename when no slides were skipped
        f_df = pd.DataFrame(fold_rows)
        n_skip = sum(s["fold"] == fold for s in skipped)
        auc = roc_auc_score(f_df["label"], f_df["prob_pos"]) if len(f_df) and f_df["label"].nunique() == 2 else np.nan
        print(f"Fold {fold}: AUC {auc:.4f} on {len(f_df)} slides, skipped {n_skip} "
              f"(checkpoint: {os.path.basename(ckpt_path)})")
        rows.extend(fold_rows)

    oof = pd.DataFrame(rows)
    if len(keys) > 1:
        print("\nSlides per feature key:", oof["feature_key"].value_counts().to_dict())
    skipped_df = pd.DataFrame(skipped, columns=["slide", "fold", group_col, "error"])
    oof.attrs["skipped"] = skipped_df
    oof.attrs["configs"] = configs
    oof.attrs["label_mapping"] = label_mapping
    if len(skipped_df):
        print(f"\nSkipped {len(skipped_df)} unreadable slides. Per {group_col}:")
        print(skipped_df[group_col].value_counts().to_string())
    return oof


def group_metrics(oof, group_col):
    """AUC and accuracy per group on pooled out-of-fold predictions, plus an overall row."""
    rows = []
    for group, d in list(oof.groupby(group_col)) + [("ALL", oof)]:
        y = d["label"]
        rows.append({
            group_col: group,
            "n_slides": len(d),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "auc": roc_auc_score(y, d["prob_pos"]) if y.nunique() == 2 else np.nan,  # undefined with one class
            "accuracy": float((y == d["pred"]).mean()),
        })
    return pd.DataFrame(rows)
