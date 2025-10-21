# Helper functions for data manipulation
import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, Dict
import math


def to_tuples(df):
    """Convert lists in a DataFrame to tuples for hashability."""
    df = df.map(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def to_lists(df):
    """Convert tuples in a DataFrame to lists."""
    df = df.map(lambda x: list(x) if isinstance(x, tuple) else x)
    return df

def convert_and_flatten(df, columns):
    """Convert string representations of lists/tuples in specified columns to actual lists and flatten them."""
    for col in columns:
        df[col] = df[col].apply(lambda x: [] if pd.isnull(x) 
                                else [item for sub in ast.literal_eval(x) 
                                      for item in (sub if isinstance(sub, (list, tuple)) else [sub])])
    return df

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

    def plot_many(self, subset_col: str, letter: str, n: int = 10, by_category: bool = False, exclude_text: Optional[list[str]] = None):
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

    # Convert and flatten specified columns
    df_all = convert_and_flatten(df_all, ["wsi filenames", "T", "M", "Other", "T category", "M category"])
    
    # Initialize MetadataExplorer
    explorer = MetadataExplorer(df_all)

    # Example: Plot top T codes overall
    explorer.plot_top_counts(letter = 'T', n=5)

    # Example: Plot top M codes for each sex
    explorer.plot_many(subset_col = 'sex', letter = 'M', n=5)