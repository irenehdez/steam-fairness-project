from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

def build_lgbm_dataset(
        interactions_df: pd.DataFrame,
        n_items: int,
        X_items: csr_matrix,
        n_neg_per_pos: int = 4,
        random_state: int = 42,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a LightGBM ranking dataset from user–item interactions.

    For each user u in `interactions_df`:
      - take all positively interacted items (from this split)
      - sample `n_neg_per_pos` * (#positives) items that u has
        NOT interacted with (negative sampling)
      - concatenate positives + negatives into a candidate set
      - assign labels 1 to positives, 0 to negatives
      - construct feature rows for all (u, i) pairs:
           [ user_feature , item_features[i, :] ]
      - record group size (number of candidates for this user)

    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame with at least ['user_id', 'item_id'] (REINDEXED ids),
        containing interactions for a given split (e.g., train or val).

    n_items : int
        Total number of items (max_item_id + 1). Typically
        `X_items.shape[0]`.

    X_items : csr_matrix, shape (n_items, n_item_features)
        Item feature matrix produced by `build_item_feature_matrix`.
        Row j corresponds to item with reindexed id j.

    n_neg_per_pos++++++ : int, default=4
        Ratio of negatives per positive interaction when sampling.

    random_state : int, default=42
        Seed for the RNG used in negative sampling.

    Returns
    -------
    X_all : csr_matrix, shape (n_samples, 1 + n_item_features)
        Sparse feature matrix with one row per (user, item) pair.
        The first column is the (numeric) user_id; the remaining
        columns correspond to `X_items` features.

    y_all : np.ndarray, shape (n_samples,)
        Labels: 1 for positive interactions, 0 for negatives.

    group : np.ndarray, shape (n_users_in_split,)
        Number of samples (candidates) per user, in the same order
        in which users were processed. This is the `group` argument
        required by LightGBM's ranking objectives.

    user_ids_all : np.ndarray, shape (n_samples,)
        For each row in X_all / y_all, the corresponding user_id.
        Useful later to map predictions back to users.

    item_ids_all : np.ndarray, shape (n_samples,)
        For each row in X_all / y_all, the corresponding item_id.
        Useful later to reconstruct recommendation rankings.
    """
    rng = np.random.default_rng(random_state)

    all_items = np.arange(n_items, dtype=np.int32)

    X_blocks = []
    y_blocks = []
    group_sizes = []
    user_ids_blocks = []
    item_ids_blocks = []

    users_in_split = interactions_df["user_id"].unique()

    for user_id in users_in_split:
        # Positive items for this user in the split
        user_pos_items = interactions_df.loc[
            interactions_df["user_id"] == user_id, "item_id"
        ].unique()

        if len(user_pos_items) == 0:
            continue

        user_pos_items_sorted = np.sort(user_pos_items)
        candidate_neg_pool = np.setdiff1d(all_items, user_pos_items_sorted, assume_unique=True)

        if len(candidate_neg_pool) == 0:
            continue

        n_pos = len(user_pos_items)
        n_neg = n_neg_per_pos * n_pos
        n_neg = min(n_neg, len(candidate_neg_pool))

        neg_items = rng.choice(candidate_neg_pool, size=n_neg, replace=False)

        item_ids_u = np.concatenate([user_pos_items, neg_items])
        labels_u = np.concatenate(
            [np.ones_like(user_pos_items, dtype=np.float32),
             np.zeros_like(neg_items, dtype=np.float32)]
        )

        X_items_u = X_items[item_ids_u]

        user_col = np.full((len(item_ids_u), 1), user_id, dtype=np.int32)
        X_user = csr_matrix(user_col)

        X_u = hstack([X_user, X_items_u], format='csr')

        X_blocks.append(X_u)
        y_blocks.append(labels_u)
        group_sizes.append(len(item_ids_u))
        user_ids_blocks.append(np.full(len(item_ids_u), user_id, dtype=np.int32))
        item_ids_blocks.append(item_ids_u.astype(np.int32))

    if not X_blocks:
        raise ValueError("No users with interactions found in the provided interactions_df.")
    
    X_all = vstack(X_blocks, format='csr')
    y_all = np.concatenate(y_blocks)
    group = np.array(group_sizes, dtype=np.int32)
    user_ids_all = np.concatenate(user_ids_blocks)
    item_ids_all = np.concatenate(item_ids_blocks)

    return X_all, y_all, group, user_ids_all, item_ids_all