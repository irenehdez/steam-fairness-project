from typing import Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
import pandas as pd

def train_lgbm_ranker(
        X_train: csr_matrix,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: csr_matrix = None,
        y_val: np.ndarray = None,
        group_val: np.ndarray = None,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_data_in_leaf: int = 50,
        feature_fraction: float = 0.8,
        lambda_l2: float = 0.0,
        random_state: int = 42,
        sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    """
    Train a LightGBM ranking model (LambdaRank).

    Parameters
    ----------
    X_train, y_train, group_train
        Training data, labels and group sizes per user.
        X_train is a CSR matrix with first column = user_id,
        remaining columns = item features.

    X_val, y_val, group_val
        Optional validation data. If provided, early stopping
        is used based on validation NDCG.

    Hyperparameters
    ---------------
    num_leaves, learning_rate, n_estimators, min_data_in_leaf,
    feature_fraction, lambda_l2, random_state

    Returns
    -------
    model : lightgbm.Booster
        Trained LightGBM ranking model.
    """
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "lambda_l2": lambda_l2,
        "verbose": -1,
        "seed": random_state,
        "deterministic": True,
        "force_row_wise": True,
    }

    categorical_features = [0]  # user_id is categorical

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=group_train,
        free_raw_data=False,
        categorical_feature=categorical_features,
        weight=sample_weight, 
    )

    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None and group_val is not None:
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            group=group_val,
            free_raw_data=False,
            categorical_feature=categorical_features,
        )
        valid_sets.append(val_data)
        valid_names.append("valid")

    if len(valid_sets) > 1:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
        )

    return model


def recommend_lgbm_for_user(
        model: lgb.Booster,
        user_id: int,
        X_items: csr_matrix,
        known_items: np.ndarray = None,
        top_k: int = 10,
) -> np.ndarray:
    """
    Compute top-K item recommendations for a single user with LightGBM.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM ranking model.
    user_id : int
        Reindexed user_id (0..n_users-1).
    X_items : csr_matrix, shape (n_items, n_item_features)
        Item feature matrix, as returned by build_item_feature_matrix.
    known_items : np.ndarray or list, optional
        Items to exclude from recommendation (e.g., items already
        seen by the user in train). If None, no filtering.
    top_k : int
        Number of items to recommend.

    Returns
    -------
    rec_items : np.ndarray, shape (top_k,)
        Indices of recommended items (reindexed item_ids).
    """
    n_items = X_items.shape[0]

    # Build feature matrix for all items for this user
    user_col = np.full((n_items, 1), user_id, dtype=np.int32)
    X_user = csr_matrix(user_col)
    
    X_candidates = hstack([X_user, X_items], format="csr")
    # Predict scores
    scores = model.predict(X_candidates)
    scores = np.asarray(scores).ravel()

    # Exclude known items
    if known_items is not None and len(known_items) > 0:
        scores[np.asarray(known_items, dtype=np.int64)] = -np.inf  # Set scores of known items to -inf

    if top_k >= len(scores):
        topk_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k)[:top_k]
        topk_idx = part[np.argsort(-scores[part])]

    return topk_idx[:top_k]


def evaluate_lgbm_on_split(
        model: lgb.Booster,
        X_items: csr_matrix,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        k: int = 10,
) -> Tuple[float, float]:
    """
    Evaluate a trained LightGBM model on a test split using Recall@K and NDCG@K.

    Parameters
    ----------
    model : lgb.Booster
        Trained model.
    X_items : csr_matrix
        Item feature matrix.
    train_df : pd.DataFrame
        Interactions used for training. Used to know which items each user
        has already seen, so we can filter them at recommendation time.
    test_df : pd.DataFrame
        Interactions used as ground truth for evaluation.
    k : int
        Cutoff for Recall@K and NDCG@K.

    Returns
    -------
    avg_recall, avg_ndcg : floats
        Average Recall@K and NDCG@K over test users.
    """

    from .metrics import evaluate_topk

    user_relevance: Dict[int, set] = (
        test_df.groupby("user_id")["item_id"]
        .apply(set).to_dict()
    )

    user_seen: Dict[int, np.ndarray] = (
        train_df.groupby("user_id")["item_id"]
        .apply(lambda s: s.values).to_dict()
    )

    user_recs: Dict[int, np.ndarray] = {}

    for user_id in user_relevance.keys():
        known_items = user_seen.get(user_id, np.array([], dtype=np.int64))
        rec_items = recommend_lgbm_for_user(
            model,
            user_id = user_id,
            X_items = X_items,
            known_items=known_items,
            top_k=k,
        )
        user_recs[user_id] = rec_items

    avg_recall, avg_ndcg = evaluate_topk(user_recs, user_relevance, k=k)
    return avg_recall, avg_ndcg, user_recs