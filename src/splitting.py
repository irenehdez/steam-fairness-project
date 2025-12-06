import numpy as np
import pandas as pd

def weak_generalization_split(
    df_interactions: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
):
    """
    Weak-generalization split per user SIN timestamp:
      - For each user, shuffle interactions
      - First train_frac -> train
      - Next val_frac   -> val
      - Rest            -> test

    Assumes df_interactions has at least columns: ['user_id', 'item_id'].
    Returns: train_df, val_df, test_df
    """
    assert np.isclose(train_frac + val_frac, 0.9), "train + val must be 0.9"
    rng = np.random.RandomState(seed)

    train_list = []
    val_list = []
    test_list = []

    for user_id, user_df in df_interactions.groupby("user_id"):
        # Shuffle rows for this user
        idx = np.arange(len(user_df))
        rng.shuffle(idx)
        user_df = user_df.iloc[idx]

        n_total = len(user_df)
        if n_total == 0:
            continue

        n_train = int(n_total * train_frac)
        n_val = int((train_frac + val_frac) * n_total)

        train_list.append(user_df.iloc[:n_train])
        val_list.append(user_df.iloc[n_train:n_val])
        test_list.append(user_df.iloc[n_val:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, val_df, test_df