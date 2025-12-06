import numpy as np
from typing import Iterable, Set, Dict

def recall_at_k(recommended: Iterable[int], relevant: Set[int], k: int) -> float:
    """
    Recall@k for a single user.
    """
    rec_k = list(recommended)[:k]
    if len(relevant) == 0:
        return 0.0
    
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)

def ndcg_at_k(recommended: Iterable[int], relevant: Set[int], k: int) -> float:
    """
    NDCG@k for a single user.
    """
    rec_k = list(recommended)[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # Ideal DCG
    n_relevant = min(len(relevant), k)
    if n_relevant == 0:
        return 0.0
    
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_topk(
        user_recs: Dict[int, np.ndarray],
        user_relevants: Dict[int, Set[int]],
        k: int = 10
):
    """
    Compute average Recall@k and NDCG@k over users.

    Parameters
    ----------
    user_recs : dict user_id -> np.ndarray of recommended item_ids
    user_relevant : dict user_id -> set of relevant item_ids (from TEST)
    """
    recalls = []
    ndcgs = []

    for user_id, recs in user_recs.items():
        relevant = user_relevants.get(user_id, set())
        if not relevant:
            continue
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    
    if not recalls:
        return 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(ndcgs))