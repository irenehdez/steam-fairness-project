from .data_loading import load_interactions, load_games
from .preprocessing import (
    reindex_interactions,
    create_interaction_matrix,
)
from .eda import (
    check_missing_values,
    check_duplicates_and_keys,
    check_numeric_anomalies,
    check_distributions,
)
from .splitting import weak_generalization_split
from .models_ease import train_ease, recommend_ease_for_user
from .metrics import evaluate_topk
from .features import build_item_feature_matrix
from .lgbm_dataset import build_lgbm_dataset
from .models_lgbm import train_lgbm_ranker, evaluate_lgbm_on_split
from .fairness import evaluate_provider_fairness, compute_publisher_weights, make_lgbm_sample_weights

def run_lgbm_fairness_experiments(
        train_df,
        test_df,
        X_items,
        items_df,
        lambda_values,
        n_neg_per_pos: int = 4,
        random_state: int = 42,
        n_estimators: int = 200,
):
    """
    Run the LightGBM + fairness experiments for different lambda_fair values.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions (user_id, item_id, ...).
    test_df : pd.DataFrame
        Test interactions used for evaluation.
    X_items : scipy sparse matrix
        Item feature matrix (n_items x n_features).
    items_df : pd.DataFrame
        DataFrame with item-level metadata aligned with X_items.
    lambda_values : list of float
        List of lambda_fair values to evaluate.
    n_neg_per_pos : int
        Number of negative samples per positive interaction.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        One dict per lambda_fair with metrics:
        { "lambda_fair", "recall@10", "ndcg@10", "gini@10" }.
    """
    print("\nBuilding LightGBM training dataset...")
    n_items_total = X_items.shape[0]

    X_train_lgbm, y_train_lgbm, group_train, user_ids_train, item_ids_train = build_lgbm_dataset(
        train_df,
        n_items=n_items_total,
        X_items=X_items,
        n_neg_per_pos=n_neg_per_pos,
        random_state=random_state,
    )

    print("LightGBM train dataset built.")
    print("X_train_lgbm shape:", X_train_lgbm.shape)
    print("y_train_lgbm shape:", y_train_lgbm.shape)
    print("Number of groups (users) in train:", len(group_train))

    # Base publisher weights
    pub_weights = compute_publisher_weights(items_df, power=1.0)

    # Mapping from item_id to publisher (used for fairness evaluation)
    item_to_publisher = items_df["publisher_clean"].to_dict()

    lgbm_results = []
    for lambda_fair in lambda_values:
        print("\n" + "=" * 60)
        print(f"Training LightGBM with lambda_fair = {lambda_fair}")
        print("=" * 60)

        # Interpolated per-example sample weights
        sample_weights = make_lgbm_sample_weights(
            item_ids=item_ids_train,
            items_df=items_df,
            pub_weights=pub_weights,
            lambda_fair=lambda_fair,
        )

        # Train LightGBM ranker
        print("\nTraining LightGBM ranker...")
        lgbm_model = train_lgbm_ranker(
            X_train=X_train_lgbm,
            y_train=y_train_lgbm,
            group_train=group_train,
            X_val=None,
            y_val=None,
            group_val=None,
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=200,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            lambda_l2=0.0,
            random_state=random_state,
            sample_weight=sample_weights,
            n_estimators=n_estimators,
        )
        print("LightGBM model trained.")

        # Evaluate LightGBM model on test split
        print("\nEvaluating LightBGM model on test split...")
        lgbm_recall, lgbm_ndcg, lgbm_user_recs = evaluate_lgbm_on_split(
            lgbm_model,
            X_items=X_items,
            train_df=train_df,
            test_df=test_df,
            k=10,
        )
        print(f"LightGBM Recall@10: {lgbm_recall:.4f}")
        print(f"LightGBM NDCG@10: {lgbm_ndcg:.4f}")

        #Provider fairness for LightGBM
        lgbm_gini, lgbm_exposure = evaluate_provider_fairness(
            lgbm_user_recs,
            item_to_publisher=item_to_publisher,
            k=10,
        )
        print(f"LightGBM provider Gini@10: {lgbm_gini:.4f}")
        print("Top 5 publishers by exposure (LightGBM):")
        print(lgbm_exposure.head(5))

        #Save scalar metrics
        lgbm_results.append(
            {
                "lambda_fair": lambda_fair,
                "recall@10": lgbm_recall,
                "ndcg@10": lgbm_ndcg,
                "gini@10": lgbm_gini,
            }
        )

    print("\n\n-- Summary: LightGBM fairness-accuracy trade-off --")
    print("lambda_fair | Recall@10 | NDCG@10 | Gini@10")
    for res in lgbm_results:
        print(
            f"{res['lambda_fair']:10.2f} | "
            f"{res['recall@10']:.4f} | "
            f"{res['ndcg@10']:.4f} | "
            f"{res['gini@10']:.4f}"
        )

    return lgbm_results

def run_ease_baseline(train_df, test_df, item_to_publisher, lambda_reg: float = 1e3, K: int = 10):
    """
    Train and evaluate the EASE baseline model, including provider fairness.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions.
    test_df : pd.DataFrame
        Test interactions (ground truth).
    item_to_publisher : dict
        Mapping from item_id to publisher.
    lambda_reg : float
        L2 regularization parameter for EASE.
    K : int
        Cutoff for top-K recommendations.

    Returns
    -------
    dict
        Metrics and exposure information for EASE:
        { "recall@K", "ndcg@K", "gini@K", "exposure_df" }.
    """
    print("\nCreating interaction matrix for EASE...")
    X = create_interaction_matrix(train_df)
    print("Done. Shape:", X.shape)
    print("Density:", X.nnz / (X.shape[0] * X.shape[1]))

    print("\nTraining EASE model...")
    B = train_ease(X, l2_reg=lambda_reg)
    print("Done.")

    print("\nPreparing ground truth from test set...")
    user_relevance = (
        test_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    print("\nGenerating EASE recommendations...")
    user_recs = {}
    n_users = X.shape[0]

    for user_id in range(n_users):
        if user_id not in user_relevance:
            continue
        rec_items = recommend_ease_for_user(X, B, user_id, top_k=K)
        user_recs[user_id] = rec_items

    print("\nEvaluating EASE recommendations...")
    avg_recall, avg_ndcg = evaluate_topk(user_recs, user_relevance, k=K)
    print(f"EASE Average Recall@{K}: {avg_recall:.4f}")
    print(f"EASE Average NDCG@{K}: {avg_ndcg:.4f}")

    print("\nEvaluating provider fairness for EASE...")
    ease_gini, ease_exposure = evaluate_provider_fairness(
        user_recs,
        item_to_publisher=item_to_publisher,
        k=K,
    )
    print(f"EASE provider Gini@{K}: {ease_gini:.4f}")
    print("Top 5 publishers by exposure (EASE):")
    print(ease_exposure.head(5))

    return {
        "recall@K": avg_recall,
        "ndcg@K": avg_ndcg,
        "gini@K": ease_gini,
        "exposure_df": ease_exposure,
    }


def main():
    DEBUG = True # False experimento grande

    # Load data
    interactions_df = load_interactions()
    games_df = load_games()
    
    print("Columns in train_interactions:")
    print(interactions_df.columns)

    print("\nFirst 5 interactions:")
    print(interactions_df.head())

    print("\nFirst 5 games:")
    print(games_df.head())

    # EDA checks
    check_missing_values(interactions_df, "train_interactions")
    check_missing_values(games_df, "games")
    check_duplicates_and_keys(interactions_df, games_df)
    check_numeric_anomalies(games_df, interactions_df)
    check_distributions(games_df, interactions_df)

    # Preprocessing
    print("\nReindexing user_id and item_id...")
    interactions_reindexed, new_to_old_user, new_to_old_item = reindex_interactions(interactions_df)
    print("Done. Users:", interactions_reindexed['user_id'].nunique(),
          "Items:", interactions_reindexed['item_id'].nunique())
    
    # Subconjunto para acelerar
    if DEBUG:
        N_USERS_DEBUG = 5000
        unique_users = interactions_reindexed["user_id"].drop_duplicates()
        print("Total users:", len(unique_users))

        sample_users = unique_users.sample(N_USERS_DEBUG, random_state=42)
        interactions_reindexed = interactions_reindexed[interactions_reindexed["user_id"].isin(sample_users)]
        print(f"[DEBUG] Using only {N_USERS_DEBUG} users."
              f"New interactions size: {len(interactions_reindexed)}")
    
    # Weak generalization split
    print("\nPerforming weak generalization split...")
    train_df, val_df, test_df = weak_generalization_split(interactions_reindexed)
    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    # Build item feature matrix
    print("\nBuilding item feature matrix...")
    X_items, items_df, feature_names = build_item_feature_matrix(
        games_df,
        new_to_old_item = new_to_old_item,
    )
    print("Done. Item feature matrix shape:", X_items.shape)
    print("Number of feature names:", len(feature_names))
    
    # Maping from reindexed item_id -> publisher name
    item_to_publisher = items_df["publisher_clean"].to_dict()

    # LightGBM + fairness experiments
    if DEBUG:
        lambda_values = [0.0, 1.0]
        n_estimators = 50
    else:
        lambda_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        n_estimators = 200
    
    lgbm_results = run_lgbm_fairness_experiments(
        train_df=train_df,
        test_df=test_df,
        X_items=X_items,
        items_df=items_df,
        lambda_values=lambda_values,
        n_neg_per_pos=4,
        random_state=42,
        n_estimators=n_estimators,
    )

    # EASE baseline
    ease_results = run_ease_baseline(
        train_df=train_df,
        test_df=test_df,
        item_to_publisher=item_to_publisher,
        lambda_reg=1e3,
        K=10,
    )

    print("\n\n=== Final summary ===")
    print("LightGBM fairness trade-off (per lambda_fair):")
    print(lgbm_results)
    print("\nEASE baseline results:")
    print(ease_results)



if __name__ == "__main__":
    main()