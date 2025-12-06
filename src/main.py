from .data_loading import load_interactions, load_games
from .preprocessing import (
    reindex_interactions,
    create_interaction_matrix,
    min_users_per_item,
    min_items_per_user,
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

def main():
    interactions_df = load_interactions()
    games_df = load_games()
    
    print("Columns in train_interactions:")
    print(interactions_df.columns)

    print("\nFirst 5 interactions:")
    print(interactions_df.head())

    print("\nFirst 5 games:")
    print(games_df.head())

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

    # Build LightGBM datasets
    print("\nBuilding LightGBM datasets...")
    n_items_total = X_items.shape[0]

    X_train_lgbm, y_train_lgbm, group_train, user_ids_train, item_ids_train = build_lgbm_dataset(
        train_df,
        n_items=n_items_total,
        X_items=X_items,
        n_neg_per_pos=4,
        random_state=42,
    )

    print("LightGBM train dataset built.")
    print("X_train_lgbm shape:", X_train_lgbm.shape)
    print("y_train_lgbm shape:", y_train_lgbm.shape)
    print("Number of groups (users) in train:", len(group_train))

    # Fairness-aware sample weights for LightGBM
    # 1) base weights per publisher (inverse frequency)
    pub_weights = compute_publisher_weights(items_df, power=1.0)

    #2) per-example weights interpolated with lambda_fair
    lambda_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    lgbm_results = []

    for lambda_fair in lambda_values:
        print("\n" + "=" * 60)
        print(f"Training LightGBM with lambda_fair = {lambda_fair}")
        print("=" * 60)


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
            random_state=42,
            sample_weight=sample_weights,
        )
        print("LightGBM model trained.")

        # Evaluate LightGBM model on test split
        print("\nEvaluating LightGBM model on test split...")
        lgbm_recall, lgbm_ndcg, lgbm_user_recs = evaluate_lgbm_on_split(
            lgbm_model,
            X_items=X_items,
            train_df=train_df,
            test_df=test_df,
            k=10,
        )
        print(f"LightGBM Recall@10: {lgbm_recall:.4f}")
        print(f"LightGBM NDCG@10: {lgbm_ndcg:.4f}")

        # Provider fairness for LightGBM
        lgbm_gini, lgbm_exposure = evaluate_provider_fairness(
            lgbm_user_recs,
            item_to_publisher=item_to_publisher,
            k=10,
        )
        print(f"LightGBM provider Gini@10: {lgbm_gini:.4f}")
        print("Top 5 publishers by exposure (LightGBM):")
        print(lgbm_exposure.head(5))

        # Save results
        lgbm_results.append(
            {
                "lambda_fair": lambda_fair,
                "recall@10": lgbm_recall,
                "ndcg@10": lgbm_ndcg,
                "gini@10": lgbm_gini,
            }
        )

    print("\n\n--Summary: LightGBM fairness-accuracy trade-off--")
    print("lambda_fair | Recall@10 | NDCG@10 | Gini@10")
    for res in lgbm_results:
        print(
            f"{res['lambda_fair']:10.2f} | "
            f"{res['recall@10']:.4f} | "
            f"{res['ndcg@10']:.4f} | "
            f"{res['gini@10']:.4f}"
        )

    # Create interaction matrix (only if needed)
    print("\nCreating interaction matrix...")
    X = create_interaction_matrix(train_df)
    print("Done. Shape:", X.shape)
    print("Density:", X.nnz / (X.shape[0] * X.shape[1]))

    # MIN_USERS_PER_ITEM = 20
    # MIN_ITEMS_PER_USER = 5

    # print(f"\nFiltering items with < {MIN_USERS_PER_ITEM} users...")
    # X, new_to_old_item = min_users_per_item(X, new_to_old_item, MIN_USERS_PER_ITEM)
    # print("Done. Shape:", X.shape)

    # print(f"\nFiltering users with < {MIN_ITEMS_PER_USER} items...")
    # X, new_to_old_user = min_items_per_user(X, new_to_old_user, MIN_ITEMS_PER_USER)
    # print("Done. Shape:", X.shape)

    # print("\nFinal interaction matrix shape:", X.shape)
    # print(" #users:", X.shape[0])
    # print(" #items:", X.shape[1])
    # print(" density:", X.nnz / (X.shape[0] * X.shape[1]))

    # Train EASE model
    print("\nTraining EASE model...")
    lambda_reg = 1e3
    B = train_ease(X, l2_reg=lambda_reg)
    print("Done.")

    # Prepare ground truth from test set
    print("\nPreparing ground truth from test set...")
    user_relevance = (
        test_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # Generate recommendations for each user
    print("\nGenerating recommendations...")
    user_recs = {}
    n_users = X.shape[0]
    K = 10

    for user_id in range(n_users):
        if user_id not in user_relevance:
            continue  # Only recommend for users in test set
        rec_items = recommend_ease_for_user(X, B, user_id, top_k=K)
        user_recs[user_id] = rec_items

    # Evaluate recommendations
    print("\nEvaluating recommendations...")
    avg_recall, avg_ndcg = evaluate_topk(user_recs, user_relevance, k=K)
    print(f"Average Recall@{K}: {avg_recall:.4f}")
    print(f"Average NDCG@{K}: {avg_ndcg:.4f}")

    # Provider fairness for EASE
    ease_gini, ease_exposure = evaluate_provider_fairness(
        user_recs,
        item_to_publisher=item_to_publisher,
        k=K,
    )
    print(f"EASE provider Gini@{K}: {ease_gini:.4f}")
    print("Top 5 publishers by exposure (EASE):")
    print(ease_exposure.head(5))

if __name__ == "__main__":
    main()