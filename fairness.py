from typing import Dict
import numpy as np
import pandas as pd

def gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1D array of non-negative values.
    Gini = 0 -> perfectly equal, Gini = 1 -> maximal inequality.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return 0.0
    
    values = np.clip(values, 0, None)
    if np.all(values == 0):
        return 0.0
    
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumvals = np.cumsum(sorted_vals)

    index = np.arange(1, n + 1)
    g = (2.0 * np.sum(index * sorted_vals) / (n * cumvals[-1])) - (n + 1.0) / n
    return float(g)


def compute_publisher_exposure(
        user_recs: Dict[int, np.ndarray],
        item_to_publisher: Dict[int, str],
        k: int = 10,
) -> pd.Series:
    """
    Count how many times each publisher appears in the top-k recommendations.

    Parameters
    ----------
    user_recs : dict
        Mapping user_id -> array of recommended item_ids (reindexed).
    item_to_publisher : dict
        Mapping new_item_id -> publisher name (string).
    k : int
        Cutoff for top-k exposure.
    
    Returns
    -------
    exposure_per_publisher : pd.Series
        Index = publisher, value = exposure count.
    """
    counts = {}
    for _, rec_items in user_recs.items():
        topk = rec_items[:k]
        for item_id in topk:
            pub = item_to_publisher.get(int(item_id), None)
            if pub is None:
                continue
            counts[pub] = counts.get(pub, 0) + 1
            
    exposure_series = pd.Series(counts).sort_values(ascending=False)
    return exposure_series 


def evaluate_provider_fairness(
        user_recs: Dict[int, np.ndarray],
        item_to_publisher: Dict[int, str],
        k: int = 10,
):
    """
    Compute provider-side fairness metrics from recommendation lists.

    Returns
    -------
    gini_exposure : float
        Gini coefficient of exposure across publishers.
    exposure_series : pd.Series
        Exposure count per publisher (sorted descending).
    """
    exposure = compute_publisher_exposure(user_recs, item_to_publisher, k=k)
    gini_exposure = gini(exposure.values)

    return gini_exposure, exposure


def compute_publisher_weights(
        items_df: pd.DataFrame,
        power: float = 1.0,
) -> Dict[str, float]:
    """
    Compute base weights per publisher, inversely proportional to their
    frequency in the *item catalog*.

    Parameters
    ----------
    items_df : pd.DataFrame
        Must have index = new_item_id and column 'publisher_clean'.
    power : float
        Exponent to control how aggressively we upweight rare publishers.
        power = 1.0 -> 1 / freq, power = 0.5 -> 1 / sqrt(freq), ...

    Returns
    -------
    pub_weights : dict
        Mapping publisher_name -> base weight (mean normalized to 1.0).
    """
    counts = items_df["publisher_clean"].value_counts()
    freq = counts / counts.sum()

    # Inverse frequency: rarer publishers -> larger weight
    inv = 1.0 / np.power(freq.values, power)

    #Normalize so that mean weight = 1.0 (keeps scale of the loss stable)
    inv = inv / inv.mean()

    pub_weights = dict(zip(counts.index, inv))
    return pub_weights


def make_lgbm_sample_weights(
        item_ids: np.ndarray,
        items_df: pd.DataFrame,
        pub_weights: Dict[str, float],
        lambda_fair: float = 0.5,
) -> np.ndarray:
    """
    Build per-example weights for LightGBM, interpolating between
    uniform weights (1.0) and publisher-based fairness weights.

    Parameters
    ----------
    item_ids : np.ndarray
        Reindexed item_ids for each training example (same order as X_train).
    items_df : pd.DataFrame
        Indexed by new_item_id, with column 'publisher_clean'.
    pub_weights : dict
        publisher_name -> base weight (from compute_publisher_weights).
    lambda_fair : float in [0, 1]
        0.0 -> no fairness (all weights = 1)
        1.0 -> full fairness weights

    Returns
    -------
    sample_weights : np.ndarray
        One weight per training example.
    """
    publishers = items_df.loc[item_ids, "publisher_clean"].values

    w_pub = np.array(
        [pub_weights.get(p, 1.0) for p in publishers],
        dtype=np.float64,
    )

    # Interpolate between uniform (1.0) and fairness weights
    sample_weights = (1.0 - lambda_fair) * 1.0 + lambda_fair * w_pub
    return sample_weights