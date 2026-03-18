import os
from wsidata import open_wsi
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from upsetplot import from_contents, UpSet
import seaborn as sns


class SpatialAgreement:
    def __init__(self, filenames, zarr_dir, models, tile_key = 'tiles_224'):
        self.filenames = filenames
        self.zarr_dir = zarr_dir
        self.models = models
        self.tile_key = tile_key
        self.dom_cols = [f"domain_{m}" for m in self.models]
        self.agreement_dict = self.get_agreement_dict()

    def load_slide(self, path):
        zarr_path = os.path.join(self.zarr_dir, os.path.basename(path).replace(".mrxs", ".zarr"))
        wsi = open_wsi(path, zarr_path)
        return wsi

    def get_shapes(self, wsi):
        gdf = wsi.shapes[self.tile_key]
        return gdf

    def contingency_tables(self, gdf):
        """
        Build contingency tables using the first model as reference.

        Expects `gdf` to contain one column per model named `domain_{model}`.
        Returns a dict mapping each non-reference model -> crosstab DataFrame.
        """
        missing = [c for c in self.dom_cols if c not in gdf.columns]
        if missing:
            raise KeyError(f"Missing expected domain columns in gdf: {missing}")

        ref_model = self.models[0]
        ref_col = f"domain_{ref_model}"

        cont_tables = {}
        for model in self.models[1:]:
            col = f"domain_{model}"
            tab = pd.crosstab(gdf[ref_col], gdf[col], dropna=False)
            cont_tables[model] = tab

        return cont_tables

    def domain_alignment(self, gdf, cont_tables):
        """
        Align domain labels across models by remapping each non-reference model's
        domain IDs onto the reference model's domain IDs.

        - Reference model is `self.models[0]`.
        - Uses a max-overlap one-to-one assignment (Hungarian algorithm) on the
          contingency table.
        - Any remaining (unassigned) target labels are mapped to the reference
          label with the largest overlap (many-to-one fallback).

        Returns a `gdf_aligned` where each `domain_{model}` column has been
        remapped to the reference domain ID space.
        """
        ref_model = self.models[0]
        ref_col = f"domain_{ref_model}"

        gdf_aligned = gdf.copy()
        for model in self.models[1:]:
            tab = cont_tables.get(model)
            if tab is None:
                raise KeyError(f"Missing contingency table for model '{model}'.")

            # tab rows are reference labels, columns are target labels
            cost = -tab.to_numpy(dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping = {tab.columns[c]: tab.index[r] for r, c in zip(row_ind, col_ind)}

            # fallback for any target labels not covered by the 1:1 assignment
            for target_label in tab.columns:
                if target_label in mapping:
                    continue
                col_counts = tab[target_label]
                if col_counts.sum() == 0: # no overlap -> NaN
                    mapping[target_label] = np.nan
                else: # assign to best match
                    mapping[target_label] = col_counts.idxmax()

            target_col = f"domain_{model}"
            gdf_aligned[target_col] = gdf_aligned[target_col].map(mapping)

        gdf_aligned[ref_col] = gdf_aligned[ref_col]
        return gdf_aligned

    def agreement_level(self, gdf_aligned):
        """
        Compute per-tile agreement across aligned domain columns.

        Expects `gdf_aligned` to contain `domain_{model}` for each model in
        `self.models`, where all non-reference columns have already been remapped
        into the reference model's label space.

        Produces `agreement_level` = number of agreeing pairs among models.
        """
        if len(self.dom_cols) < 2:
            raise ValueError("Need at least 2 models to compute agreement.")

        agree = np.zeros(len(gdf_aligned), dtype=int)
        for i in range(len(self.dom_cols)):
            a = gdf_aligned[self.dom_cols[i]]
            for j in range(i + 1, len(self.dom_cols)):
                b = gdf_aligned[self.dom_cols[j]]
                agree += (a == b).astype(int).to_numpy()

        gdf = gdf_aligned.copy()
        gdf["agreement_level"] = agree
        return gdf

    def get_agreement_dict(self):
        agreement_dict = {}

        for i, path in enumerate(self.filenames):
            wsi = self.load_slide(path)
            gdf = self.get_shapes(wsi)
            cont_tables = self.contingency_tabels(gdf)
            gdf_aligned = self.domain_alignment(gdf, cont_tables)
            gdf_aligned = self.agreement_level(gdf_aligned)
            agreement_dict[i] = gdf_aligned

        self.agreement_dict = agreement_dict
        return self.agreement_dict

    def slide_level_agreement(self, slide_idx):
        gdf_aligned = self.agreement_dict[slide_idx]

        n_tiles = len(gdf_aligned)
        n_models = len(self.models)
        if n_models < 2:
            raise ValueError("Need at least 2 models to compute agreement.")

        out = {}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                m1, m2 = self.models[i], self.models[j]
                c1, c2 = f"domain_{m1}", f"domain_{m2}"
                rate = float((gdf_aligned[c1] == gdf_aligned[c2]).mean())
                out[f"{m1}_vs_{m2}"] = rate

        # fraction of tiles where *all* models agree
        if n_models == 2:
            full_rate = float((gdf_aligned[self.dom_cols[0]] == gdf_aligned[self.dom_cols[1]]).mean())
        else:
            base = gdf_aligned[self.dom_cols[0]]
            all_equal = np.ones(n_tiles, dtype=bool)
            for col in self.dom_cols[1:]:
                all_equal &= (gdf_aligned[col] == base).to_numpy()
            full_rate = float(all_equal.mean())

        out["full_agreement"] = full_rate
        return out

    def plot_agreement_map(self, slide_idx, show=True):
        """
        Plot spatial agreement, with generalized coloring for any number of models:
        - 0 agreeing pairs -> "Disagreement"
        - max agreeing pairs -> "Strong"
        - otherwise -> "Moderate"
        """
        gdf_aligned = self.agreement_dict[slide_idx]
        
        n_models = len(self.models)
        if n_models < 2:
            raise ValueError("Need at least 2 models to compute agreement.")
        max_pairs = n_models * (n_models - 1) // 2

        colors = {
            "strong": "#1a9850",
            "moderate": "#fee08b",
            "disagreement": "#d73027",
        }

        level = gdf_aligned["agreement_level"]
        agreement_class = np.where(level == max_pairs, "strong", 
            np.where(level == 0, "disagreement", "moderate"),
        )
        plot_df = gdf_aligned.copy()
        plot_df["agreement_class"] = agreement_class
        plot_df["agreement_color"] = plot_df["agreement_class"].map(colors)

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_df.plot(color=plot_df["agreement_color"], linewidth=0, ax=ax)

        legend_patches = [
            mpatches.Patch(color=colors["strong"], label=f"Strong ({max_pairs}/{max_pairs})"),
            mpatches.Patch(color=colors["disagreement"], label=f"Disagreement (0/{max_pairs})"),
        ]
        if max_pairs > 1:
            legend_patches.insert(
                1, mpatches.Patch(color=colors["moderate"], label=f"Moderate (1..{max_pairs-1}/{max_pairs})")
            )
        ax.legend(handles=legend_patches, title="Agreement (pairwise)", loc="center left")
        ax.set_title("Spatial Agreement Across Models")
        ax.set_axis_off()
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax, plot_df

    def overall_slide_agreement(self):
        """
        Slide-level agreement for all slides, to use for boxplots and stripplots:
        - x: comparison (pair name or "all_agree")
        - y: agreement_rate (0..1)
        """
        rows = []
        # change d, k and v to understable variable names
        for slide_idx in range(len(self.filenames)):
            d = self.slide_level_agreement(slide_idx)
            for k, v in d.items():
                rows.append(
                    {
                        "slide_idx": slide_idx,
                        "comparison": "all_agree" if k == "full_agreement" else k,
                        "agreement_rate": float(v),
                    }
                )
        return pd.DataFrame(rows)

    def boxplot_slide_agreement(self, show=True, percent=True):
        """
        Boxplot of slide-level agreement rates across slides.
        """
        df = self.overall_slide_agreement()
        
        if percent:
            df = df.copy()
            df["agreement_rate"] = 100.0 * df["agreement_rate"]
            ylab = "Agreement (%)"
        else:
            ylab = "Agreement"

        fig, ax = plt.subplots(figsize=(10, 4))
        
        sns.boxplot(data=df, x="comparison", y="agreement_rate")
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax, df

    def stripplot_slide_agreement(self, show=True, percent=True):
        """
        Stripplot of slide-level agreement rates (one point per slide per comparison).
        """
        df = self.overall_slide_agreement()
        if percent:
            df = df.copy()
            df["agreement_rate"] = 100.0 * df["agreement_rate"]
            ylab = "Agreement (%)"
        else:
            ylab = "Agreement"

        fig, ax = plt.subplots(figsize=(10, 4))
        
        sns.stripplot(
            data=df,
            x="comparison",
            y="agreement_rate",
            jitter=True,
            size=3,
            alpha=0.7,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax, df
