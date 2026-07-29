""" Helper Functions for Data Manipulation of 100K dataset. """

import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, Dict
import math
import re
from tissue_artifact_segmentation import _load_cache, _get_processed_entries
import os

# ---------- General helper functions ----------

def lists2tuples(df):
    """Convert lists in a DataFrame to tuples for hashability."""
    df = df.map(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def tuples2lists(df):
    """Convert tuples in a DataFrame to lists."""
    df = df.map(lambda x: list(x) if isinstance(x, tuple) else x)
    return df

def _flatten(v):
    if isinstance(v, (list, tuple)):
        out = []
        for item in v:
            out.extend(_flatten(item))
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
        
    flat = _flatten(val)
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


# --------- Methods to subset a dataframe ----------

def subset_df(df, col, val):
    """ Return a subset of the dataframe where `col` equals `val`. """
    df_subset = df[df[col]==val].copy()
    return df_subset

def subset_df_list(df, col, val):
    """ Return a subset of the dataframe where `col` contains `val` in its list. """
    df_subset = df[df[col].apply(lambda x: val in x)].copy()
    return df_subset

def contains_word(cell, word):
    """ Check if `word` (case-insensitive) is present in `cell`. """
    # Normalize search term
    w = str(word).lower()

    # Missing values
    if pd.isnull(cell):
        return False

    # Iterable containers (list/tuple/set)
    if isinstance(cell, (list, tuple, set)):
        for item in cell:
            if item is None:
                continue
            if w in str(item).lower():
                return True
        return False

    # Strings or other scalar types (use str fallback)
    return w in str(cell).lower()

def subset_df_word(df, col, word):
    """ Return rows where `word` (case-insensitive) appears in `col`. """
    mask = df[col].apply(contains_word, args=(word,))
    return df[mask].copy()


def subset_df_processed(df, cache_file, category=None, status=None, model=None, filename_col="filename"):
    """Return dataframe rows that match processed cache entries.

    Filters are optional and can be combined:
    - category: e.g. "features", "tissue", "artifact"
    - status: prefix match, e.g. "complete" or "error:"
    - model: case-insensitive substring match, e.g. "h-optimus-0"
    """
    cache = _load_cache(cache_file)
    entries = _get_processed_entries(cache)

    model_lower = str(model).lower() if model is not None else None
    selected_paths = set()

    for (_, entry_category, entry_model), data in entries.items():
        if category is not None and entry_category != category:
            continue

        entry_status = str(data.get("status", ""))
        if status is not None and not entry_status.startswith(str(status)):
            continue

        if model_lower is not None:
            model_text = str(data.get("model", entry_model)).lower()
            if model_lower not in model_text:
                continue

        wsi_path = data.get("wsi_path")
        if wsi_path:
            selected_paths.add(str(wsi_path))

    df_subset = df[df[filename_col].isin(selected_paths)].copy()
    print(
        "Dataframe subset length:", len(df_subset), 
        f"(category={category}, status={status}, model={model})",
    )
    
    return df_subset


# ---------- Extract metadata from text columns ----------

def extract_snomed_from_cell(cell, valid_codes):
    if pd.isna(cell):
        return set()

    parts = re.split(r"[ ,;]+", str(cell))
    return {p for p in parts if p in valid_codes}

def extract_snomed_from_all_columns(row, columns, valid_codes):
    """ Extract valid SNOMED codes from specified columns in a row, returning a list of unique found codes. """
    found = set()
    for col in columns:
        found |= extract_snomed_from_cell(row[col], valid_codes)
    return list(found)

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

