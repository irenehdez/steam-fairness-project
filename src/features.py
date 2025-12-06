import ast
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MultiLabelBinarizer

def _parse_genre_cell(cell):
    """
    Parse a single cell from the 'genres' column.

    The dataset typically stores genres as a string representation
    of a Python list, e.g. "['Action', 'Adventure']".
    This helper tries ast.literal_eval; if it fails, it will try
    to split on common separators, otherwise returns an empty list.
    """
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    s = str(cell)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: try splitting on common separators
    if ";" in s:
        return [g.strip() for g in s.split(";") if g.strip()]
    if "," in s:
        return [g.strip() for g in s.split(",") if g.strip()]
    return [s.strip()] if s.strip() else []

def build_item_feature_matrix(
        games_df: pd.DataFrame,
        new_to_old_item: Dict[int, int],
) -> Tuple[csr_matrix, pd.DataFrame, List[str]]:
    """
    Build a sparse feature matrix for items, aligned with reindexed item_ids.

    Parameters
    ----------
    games_df : pd.DataFrame
        Raw games dataframe from games.csv.
        Must contain at least: ['item_id', 'publisher', 'genres',
                                'price', 'metascore', 'release_date'].
        Here 'item_id' is the ORIGINAL id (before reindexing).
    new_to_old_item : dict
        Mapping: new_item_id -> old_item_id, as returned by reindex_interactions.

    Returns
    -------
    X_items : csr_matrix, shape (n_items, n_features)
        Sparse feature matrix where row i corresponds to item with
        reindexed id i (0..n_items-1).
    items_df : pd.DataFrame
        DataFrame with one row per item (index=new_item_id) containing
        the intermediate feature columns (useful for debugging / analysis).
    feature_names : list of str
        Names of the columns in X_items, in order.
    """
    old_to_new_item = {old_id: new_id for new_id, old_id in new_to_old_item.items()}

    games_sub = games_df.copy()
    games_sub = games_sub[games_sub['item_id'].isin(old_to_new_item.keys())].copy()

    games_sub['new_item_id'] = games_sub['item_id'].map(old_to_new_item)
    games_sub = games_sub.dropna(subset=['new_item_id']).copy()
    games_sub['new_item_id'] = games_sub['new_item_id'].astype(int)

    games_sub = games_sub.sort_values('new_item_id').reset_index(drop=True)

    games_sub["genres_list"] = games_sub["genres"].apply(_parse_genre_cell)
    mlb = MultiLabelBinarizer(sparse_output=True)
    genres_sparse = mlb.fit_transform(games_sub["genres_list"])
    genre_feature_names = [f"genre_{g}" for g in mlb.classes_]

    if "price" in games_sub.columns:
        price_num = pd.to_numeric(games_sub["price"], errors='coerce')
        median_price = price_num.median()
        games_sub["price_clean"] = price_num.fillna(median_price)
    else:
        games_sub["price_clean"] = 0.0

    if "metascore" in games_sub.columns:
        metascore_num = pd.to_numeric(games_sub["metascore"], errors='coerce')
        median_metascore = metascore_num.median()
        games_sub["metascore_clean"] = metascore_num.fillna(median_metascore)
    else:
        games_sub["metascore_clean"] = 0.0

    if "release_date" in games_sub.columns:
        games_sub["release_year"] = (
            pd.to_datetime(games_sub["release_date"], errors='coerce')
            .dt.year
            .fillna(0)
            .astype(int)
        )
    else:
        games_sub["release_year"] = 0

    numeric_cols = ["price_clean", "metascore_clean", "release_year"]
    numeric_matrix = csr_matrix(games_sub[numeric_cols].to_numpy(dtype=np.float32))


    games_sub["publisher_clean"] = games_sub["publisher"].fillna("unknown")
    publisher_dummies = pd.get_dummies(games_sub["publisher_clean"], prefix="pub", dtype=np.uint8)
    publisher_sparse = csr_matrix(publisher_dummies.to_numpy(dtype=np.uint8))

    X_items = hstack([publisher_sparse, genres_sparse, numeric_matrix], format='csr')

    feature_names = (list(publisher_dummies.columns) + genre_feature_names + numeric_cols)

    items_df = games_sub.set_index('new_item_id')

    return X_items, items_df, feature_names