import numpy as np
from scipy.sparse import csr_matrix

def train_ease(X: csr_matrix, l2_reg: float = 1e3) -> np.ndarray:
    """
    Train EASE model.

    Parameters
    X: csr_matrix, shape (n_users, n_items)
       Binary user-item interaction matrix for TRAIN set.
    l2_reg: float
         L2 regularization strength (lambda).

    Returns
    B: np.ndarray, shape (n_items, n_items)
       Item-item weight matrix for EASE model.
    """
    G = X.T @ X  # Gram matrix (n_items x n_items)
    G = G.toarray()  # Convert to dense

    n_items = G.shape[0]

    G[np.diag_indices(n_items)] += l2_reg

    P = np.linalg.inv(G)  # Inverse of regularized Gram matrix

    B = -P / np.diag(P)  # Compute weights
    np.fill_diagonal(B, 0.0)  # Zero out diagonal

    return B

def recommend_ease_for_user(
        X: csr_matrix,
        B: np.ndarray,
        user_id: int,
        top_k: int = 10
) -> np.ndarray:
    """
    Compute top-k recommendations for a single user with EASE.

    Parameters
    ----------
    X : csr_matrix, shape (n_users, n_items)
        TRAIN interaction matrix.
    B : np.ndarray, shape (n_items, n_items)
        Weight matrix returned by train_ease.
    user_id : int
        Row index of the user in X (reindexed user_id).
    top_k : int
        Number of items to recommend.

    Returns
    -------
    rec_items : np.ndarray, shape (top_k,)
        Indices of recommended items (item_ids reindexed).
    """
    x_u = X[user_id]
    scores = x_u @ B  # Compute scores
    scores = np.asarray(scores).ravel()

    # Filter out items already interacted with
    interacted_items = x_u.indices
    scores[interacted_items] = -np.inf

    if top_k >= len(scores):
        topk_idx = np.argsort(-scores)
    else:
        topk_part = np.argpartition(-scores, top_k)[:top_k]
        topk_idx = topk_part[np.argsort(-scores[topk_part])]

    return topk_idx[:top_k]