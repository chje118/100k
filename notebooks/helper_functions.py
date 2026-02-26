""" Helper Functions for Data Manipulation of 100K dataset. """

import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, Dict
import math
import re
from tissue_artifact_segmentation import load_big_cache, already_processed
import os

# ---------- General Helper Function ----------

def lists2tuples(df):
    """Convert lists in a DataFrame to tuples for hashability."""
    df = df.map(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def tuples2lists(df):
    """Convert tuples in a DataFrame to lists."""
    df = df.map(lambda x: list(x) if isinstance(x, tuple) else x)
    return df

def flatten(v):
    if isinstance(v, (list, tuple)):
        out = []
        for item in v:
            out.extend(flatten(item))
        return out
    else:
        return [v]

def strings2lists(x):
    if pd.isnull(x):
        return []
    try:
        val = ast.literal_eval(x)
    except Exception:
        return [str(x)]
        
    flat = flatten(val)
    return list({str(item) for item in flat})

def is_missing(x):
    """ Check for missing values or empty lists. """
    try:
        if isinstance(x, list):
            return len(x) == 0
        elif isinstance (x, tuple):
            return len(x) == 0
        return pd.isnull(x)
    except Exception:
        return False

def combine_values(series):
    uniques = series.dropna().unique()
    return ", ".join(map(str, uniques))


# --------- Subset Dataset ----------
def subset_df(df, col, val):
    df_subset = df[df[col]==val].copy()
    return df_subset

def subset_df_list(df, col, val):
    df_subset = df[df[col].apply(lambda x: val in x)].copy()
    return df_subset

# ---------- SNOMED Helper Functions ----------

def extract_snomed_codes(df, valid_codes, main_letters=("T", "M")):
    """
    Method to extract SNOMED codes and add SNOMED columns (T, M, Other) to the dataframe.
        df: dataframe containing "snomed kode"
        valid_codes: set of all valid SNOMED codes
        main_letters = ("T", "M"): letters to create separate columns for
    """
    for letter in main_letters:
        df[letter] = df["snomed kode"].apply(lambda x: extract_by_letter(x, letter, valid_codes))    
    df["Other"] = df["snomed kode"].apply(lambda x: extract_other(x, valid_codes, main_letters))
    return df

def extract_by_letter(cell, letter, valid_codes):
    if pd.isna(cell):
        return []
    parts = re.split(r"[ ,;]+", str(cell))
    codes = {part for part in parts if part in valid_codes and part.startswith(letter)}
    return list(codes)

def extract_other(cell, valid_codes, main_letters):
    if pd.isna(cell):
        return []
    parts = re.split(r"[ ,;]+", str(cell))
    codes = {part for part in parts if part in valid_codes and part[0] not in main_letters}
    return list(codes)

def find_missing_codes(row, letter, text_columns, valid_codes):
    if is_missing(row[letter]):
        found = []
        for c in text_columns: 
            codes = extract_by_letter(row[c], letter, valid_codes)
            if codes: 
                found.extend(codes)
        return list(set(found)) if found else None
    return None

def fill_from_candidates(row, col, valid_codes):
    current = row[col]
    candidates = row[f"{col} candidates"]
    if is_missing(current):
        if candidates is not None:
            return list({c for c in candidates if c in valid_codes})
        return []
    return current

def extract_snomed_codes_from_cell(cell, valid_codes):
    if pd.isna(cell):
        return set()

    parts = re.split(r"[ ,;]+", str(cell))
    return {p for p in parts if p in valid_codes}

def extract_snomed_from_all_columns(row, columns, valid_codes):
    found = set()
    for col in columns:
        found |= extract_snomed_codes_from_cell(row[col], valid_codes)
    return list(found)

    

# ---------- Other Helper Functions ----------

def find_undersoeger(row, columns):
    pattern = r'(?i)undersøger:\s*([^;.]+?)(?=[;.]\s*|$)'
    names = []
    for col in columns:
        text = row[col]
        if isinstance(text, str):
            matches = re.findall(pattern, text)
            for m in matches:
                parts = re.split(r'\s*,\s*', m)
                for p in parts:
                    clean = " ".join(p.split())
                    clean = re.sub(r'[\s"\-]+$', '', clean)
                    if clean:
                        names.append(clean)
    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)
    return unique

def find_team(row, team_column, text_columns, valid_teams):
    if row[team_column] in valid_teams:
        return None
        
    found = []

    for col in text_columns:
        text = str(row[col]).lower()
        for team in valid_teams:
            if team.lower() in text:
                found.append(team)

    return list(set(found)) if found else None


# ---------- Generate Subset ----------

def with_tissue_artifact(df, cache_file, segmentation_type, status="complete", version="default"):
    cache = load_big_cache(cache_file)
    processed = already_processed(cache)

    def lookup_status(path):
        slide_name = os.path.basename(path)
        key = (slide_name, segmentation_type, version)
        return processed.get(key)
    
    df = df.copy()
    df["status"] = df["filename"].apply(lookup_status)

    df_subset = df[df["status"].str.startswith(status, na=False)]
    print("Dataframe subset length: ", len(df_subset), f"({segmentation_type}, {version}, {status})")
    
    return df_subset




def with_features(df, feature_file):
    df_feature = pd.read_csv(feature_file)
    df_feature = df_feature_result[df_feature_result['status'] == "feature extraction complete"]["wsi_path"].astype(str)
    with_features = set(df_feature)
    print("WSIs with features detected: ", len(with_features))
    files = list(with_features)
    df_subset = df[df["filename"].isin(files)].copy()
    print("Dataframe subset length: ", len(df_sub))
    return df_subset

def top_tissues(df, n_tissues):
    all_tissues = [cat for sublist in df['T category'] for cat in sublist]
    top_counts = pd.Series(all_tissues).value_counts().head(n_tissues)
    top_tissues = top_counts.index.tolist()
    return top_tissues

def tissue_subset(df, tissue):
    subset = df[df["T category"].apply(lambda x: len(x) == 1 and tissue in x)]
    return subset

def sample_by_patient(df, n_patients, n_samples):
    """ Pick n_samples from n_patients. """
    patients = (
        df.groupby("rekvnr")
          .size()
          .sort_values(ascending=False)
          .index.tolist()
    )
    selected = []
    for p in patients:
        if len(selected) >= n_patients:
            break
        df_patient = df[df["rekvnr"] == p].head(n_samples)
        selected.append(df_patient)
    if selected:
        return pd.concat(selected, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns)

def generate_subset(df, n_patients, n_samples, n_tissues):
    tissues = top_tissues(df, n_tissues)
    subsets = []

    for tissue in tissues:
        # 1. Subset by tissue
        subset = tissue_subset(df, tissue)

        # 2. Filter WSIs with tissue segmentation
        subset_tissue = subset[subset["filename"].isin(with_tissue)]
        
        # 3. Among these, prioritize WSIs with feature extraction
        subset_features = subset_tissue[subset_tissue["filename"].isin(with_features)]
        
        # 4. Sample WSIs with features first
        sample_feature = sample_by_patient(subset_features, n_patients, n_samples)

        # 5. If not enough, add WSIs with tissue-only
        pts_needed = n_patients - sample_feature["rekvnr"].nunique()
        remaining_tissue = subset_tissue[~subset_tissue["filename"].isin(sample_feature["filename"])]
        sample_tissue = sample_by_patient(remaining_tissue, pts_needed, n_samples)

        # 6. If still not enough (no features & no tissue), take last samples from the original subset
        pts_needed -= sample_tissue["rekvnr"].nunique()
        remaining_fallback = subset[
            ~subset["filename"].isin(sample_feature["filename"])
            & ~subset["filename"].isin(sample_tissue["filename"])
        ]
        sample_fallback = sample_by_patient(remaining_fallback, pts_needed, n_samples)
      
        # 7. Combine chosen WSIs for this tissue
        combined = pd.concat([sample_feature, sample_tissue, sample_fallback], ignore_index=True)

        print(f"{tissue} ({len(subset)} total):")
        print(f"  {len(sample_feature)} with features")
        print(f"  {len(sample_tissue)} with tissue only")
        print(f"  {len(sample_fallback)} fallback")
        print(f"  → Combined: {len(combined)}\n")
        
        subsets.append(combined)

    # 8. Combine all tissues
    df_subset = pd.concat(subsets, ignore_index=True)
    print(f"\nFinal combined subset size: {len(df_subset)}")
    return df_subset

# Example usage:
# subset_f = generate_subset(df_HE_f, n_patients = 3, n_samples = 5, n_tissues = 5)

# ----------

class MetadataExplorer:
    def __init__(self, df: pd.DataFrame, code_to_text: Optional[Dict] = None):
        self.df = df
        if code_to_text:
            self.code_to_text = code_to_text

    def get_subset_dict(self, subset_col: str) -> dict:
        """Get a dictionary of DataFrames, each corresponding to a unique value in the specified column."""
        subset_dict = {group: data.copy() for group, data in self.df.groupby(subset_col)}
        return subset_dict

    def available_keys(self, subset_col: str) -> None:
        subset_dict = self.get_subset_dict(subset_col)
        print(list(subset_dict.keys()))
        return None

    def get_subset_df(self, subset_col: str, subset_name: str) -> pd.DataFrame:
        subset_dict = self.get_subset_dict(subset_col)
        subset_df = subset_dict.get(subset_name)
        if subset_df is None:
            subset_df = self.df
            print(f"Error occurred, while subsetting {subset_col} ({subset_name}). Returning main DataFrame.")
        return subset_df

    def _count_codes(self, df: pd.DataFrame, letter: str) -> pd.DataFrame:
        all_codes = [c for code in df[letter] for c in code]
        counts = Counter(all_codes)
        counts_df = pd.DataFrame(counts.items(), columns=[letter, 'count']).sort_values('count', ascending=False)
        if self.code_to_text:
            counts_df['description'] = counts_df[letter].map(self.code_to_text)
        else: 
            counts_df['description'] = counts_df[letter]
        return counts_df

    def _count_categories(self, df: pd.DataFrame, letter: str) -> pd.DataFrame:
        category_col = f"{letter} category"
        all_categories = [c for category in df[category_col] for c in category]
        counts = Counter(all_categories)
        counts_df = pd.DataFrame(counts.items(), columns=['description', 'count']).sort_values('count', ascending=False)
        return counts_df

    def plot_top_counts(self, letter: str, n: int = 10, text: str = '', by_category: bool = False, subset_col: Optional[str] = None, subset_name: Optional[str] = None, exclude_text: Optional[list[str]] = None):

        # Select subset if specified
        if subset_col and subset_name:
            df = self.get_subset_df(subset_col, subset_name)
        else: 
            df = self.df

        # Count either categories or codes
        if by_category: 
            counts_df = self._count_categories(df, letter) 
        else: 
            counts_df = self._count_codes(df, letter)

        # Exclude texts if provided
        if exclude_text:
            exclude_lower = [t.lower() for t in exclude_text]
            counts_df = counts_df[~counts_df['description'].str.lower().isin(exclude_lower)]

        # Plot
        top_n = counts_df.head(n)
        plt.figure(figsize=(10,6))
        plt.barh(top_n['description'], top_n['count'], color='skyblue')
        plt.title(f"Top {n} {letter} codes {text}")
        plt.xlabel("Frequency")
        plt.ylabel(f"{letter} {'categories' if by_category else 'codes'}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()    

    def plot_many(self, subset_col: str, letter: str, n: int = 10, by_category: bool = False, exclude_text: list[str] = None):
        subset_dict = self.get_subset_dict(subset_col)
        n_groups = len(subset_dict)

        # --- Determine grid size ---
        n_cols = 2
        n_rows = math.ceil(n_groups / n_cols)

        # Create one figure with one subplot per group
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharex=False)

        # Flatten axes array for easy iteration
        axes = axes.flatten() if n_groups > 1 else [axes]
        
        for ax, (group, df) in zip(axes, subset_dict.items()):
            df = df.copy()

            # --- Count codes or categories ---
            if by_category:
                counts_df = self._count_categories(df, letter)
            else:
                counts_df = self._count_codes(df, letter)

            # --- Exclude unwanted texts ---
            if exclude_text:
                exclude_lower = [t.lower() for t in exclude_text]
                counts_df = counts_df[~counts_df['description'].str.lower().isin(exclude_lower)]

            # --- Take top n and plot ---
            top_n = counts_df.head(n).sort_values('count', ascending=True) 
            ax.barh(top_n['description'], top_n['count'], color='steelblue')
            ax.set_title(f"{subset_col}: {group}", fontsize=12)
            ax.set_xlabel('Number of cases')
            ax.set_ylabel('')

        # Turn off any unused subplots
        for ax in axes[len(subset_dict):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    data = "path to data"
    df_all = pd.read_csv(data)

    # Initialize MetadataExplorer
    explorer = MetadataExplorer(df_all)

    # Example: Plot top T codes overall
    explorer.plot_top_counts(letter = 'T', n=5)

    # Example: Plot top M codes for each sex
    explorer.plot_many(subset_col = 'sex', letter = 'M', n=5)