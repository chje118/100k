import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wsidata import open_wsi
import lazyslide as zs
from abmil import load_checkpoint


class TileAuditor:
    """
    Compute additive class-wise contributions for a binary ABMIL checkpoint
    and plot disease / neutral / low-attention tile grids for Phase 1 audit.
    """
    def __init__(
        self,
        checkpoint_path: str,
        zarr_dir: str,
        feature_key: str = "features_h-optimus-0",
        tile_key: str = "tiles224",
        top_k: int = 10,
    ):
        self.checkpoint_path = checkpoint_path
        self.zarr_dir = zarr_dir
        self.feature_key = feature_key
        self.tile_key = tile_key
        self.top_k = top_k

        # Load model once
        self.model, self.config, self.label_mapping = load_checkpoint(checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def compute_contributions_for_slide(self, slide_path: str):
        zarr_path = os.path.join(
            self.zarr_dir,
            os.path.basename(str(slide_path)).replace(".mrxs", ".zarr"),
        )

        wsi = open_wsi(slide_path, zarr_path)
        adata = wsi.tables[self.feature_key]

        feats = torch.from_numpy(adata.X).float()  # [N, 1536]
        tile_ids = np.asarray(adata.obs["tile_id"])

        with torch.inference_mode():
            logits, attention = self.model(feats.to(self.device))

        probs = torch.softmax(logits.float(), dim=0).cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # Contributions
        with torch.inference_mode():
            feats_f32 = feats.to(self.device).float()
            attn_f32 = attention.to(self.device).float()
            a = attn_f32.mean(dim=1)  # [N]
            W = self.model.classifier.weight.float()  # [C, D]
            b = self.model.classifier.bias.float()  # [C]
            contrib = a[:, None] * (feats_f32 @ W.T)  # [N, C]

        contrib_np = contrib.cpu().numpy()
        a_np = a.cpu().numpy()

        disease_score = contrib_np[:, pred_idx]
        neutrality_score = -np.abs(contrib_np).max(axis=1)

        tile_table = wsi.shapes[self.tile_key].copy()
        tile_table["tile_id"] = tile_table["tile_id"].astype(tile_ids.dtype)

        tile_df = pd.DataFrame(
            {
                "tile_id": tile_ids,
                "attention": a_np,
                "contrib_0": contrib_np[:, 0],
                "contrib_1": contrib_np[:, 1],
                "disease_score": disease_score,
                "neutrality_score": neutrality_score,
            }
        )

        tile_table = tile_table.merge(tile_df, on="tile_id", how="inner")
        return tile_table, pred_idx, float(probs[pred_idx])

    def plot_grid(self, tiles_gdf, slide_path, title="Tiles"):
        zarr_path = os.path.join(
            self.zarr_dir,
            os.path.basename(str(slide_path)).replace(".mrxs", ".zarr"),
        )
        wsi = open_wsi(slide_path, zarr_path)
        tk = self.tile_key

        n = len(tiles_gdf)
        cols = 2
        rows = int(np.ceil(n / cols)) if n > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.asarray(axes).flatten()

        for j, (_, tr) in enumerate(tiles_gdf.iterrows()):
            g = tr["geometry"]
            if not hasattr(g, "bounds"):
                axes[j].axis("off")
                continue
            minx, miny, maxx, maxy = g.bounds
            zs.pl.tiles(
                wsi, tile_key=tk, zoom=(minx, maxx, miny, maxy), ax=axes[j]
            )
            axes[j].set_title(f"tile {tr.get('tile_id', j)}")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def audit_slide(self, slide_path: str, tissue: str = None):
        tile_table, pred_idx, conf = self.compute_contributions_for_slide(slide_path)
        print("Predicted class:", pred_idx, "Confidence:", conf)

        disease_tiles = (
            tile_table.sort_values("disease_score", ascending=False)
            .head(self.top_k)
            .copy()
        )
        neutral_tiles = (
            tile_table.sort_values("neutrality_score", ascending=False)
            .head(self.top_k)
            .copy()
        )
        low_attn_tiles = (
            tile_table.sort_values("attention", ascending=True)
            .head(self.top_k)
            .copy()
        )

        tissue_str = f" ({tissue})" if tissue else ""
        self.plot_grid(disease_tiles, slide_path, f"Disease-evidence tiles{tissue_str}")
        self.plot_grid(neutral_tiles, slide_path, f"Neutral tiles{tissue_str}")
        self.plot_grid(low_attn_tiles, slide_path, f"Low-attention tiles{tissue_str}")

        impression = input("One-line impression for this slide: ")
        print("Recorded impression:", impression)
        return impression