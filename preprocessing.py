import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def reindex_interactions(df_interactions: pd.DataFrame):
    """
    Takes a dataframe with columns ['user_id', 'item_id', ...]
    and returns:
      - df with new 'user_id' and 'item_id' (0..n_users-1, 0..n_items-1)
      - mappings new->old for users and items
    """
    df = df_interactions.copy()

    df.rename(columns={"user_id": "old_user_id", "item_id": "old_item_id"}, inplace=True)

    user_id_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(df["old_user_id"].unique())
    }
    df["user_id"] = df["old_user_id"].map(user_id_mapping)

    item_id_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(df["old_item_id"].unique())
    }
    df["item_id"] = df["old_item_id"].map(item_id_mapping)

    new_to_old_user = {v: k for k, v in user_id_mapping.items()}
    new_to_old_item = {v: k for k, v in item_id_mapping.items()}

    return df, new_to_old_user, new_to_old_item

def create_interaction_matrix(df_interactions: pd.DataFrame) -> csr_matrix:
    """
    Build a binary user-item interaction matrix in CSR format.
    Assumes df has columns ['user_id', 'item_id'] already reindexed.
    """
    n_users = int(df_interactions["user_id"].max()) + 1
    n_items = int(df_interactions["item_id"].max()) + 1

    rows = df_interactions["user_id"].values
    cols = df_interactions["item_id"].values
    data = np.ones(len(df_interactions), dtype=np.float32)

    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    return interaction_matrix

def min_users_per_item(X: csr_matrix, new_to_old_item: dict, min_users: int):
    """
    Remove items (columns) with < min_users interactions.
    Returns:
      - filtered interaction matrix
      - updated new_to_old_item mapping
    """
    item_mask = np.asarray(X.getnnz(axis=0)).ravel() >= min_users
    X_filtered = X[:, item_mask]

    kept_old_cols = np.where(item_mask)[0]
    updated_mapping = {
        new_idx: new_to_old_item[old_idx]
        for new_idx, old_idx in enumerate(kept_old_cols)
    }

    return X_filtered, updated_mapping

def min_items_per_user(X: csr_matrix, new_to_old_user: dict, min_items: int):
    """
    Remove users (rows) with < min_items interactions.
    Returns:
      - filtered interaction matrix
      - updated new_to_old_user mapping
    """
    user_mask = np.asarray(X.getnnz(axis=1)).ravel() >= min_items
    X_filtered = X[user_mask]

    kept_old_rows = np.where(user_mask)[0]
    updated_mapping = {
        new_idx: new_to_old_user[old_idx]
        for new_idx, old_idx in enumerate(kept_old_rows)
    }

    return X_filtered, updated_mapping