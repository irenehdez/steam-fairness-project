import pandas as pd

def check_missing_values(df: pd.DataFrame, name: str):
    print(f"\n=== Missing values in {name} ===")
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing_pct[missing_pct > 0].head(20))  # mostramos solo las que tienen algo de missing
    num_complete = (df.isna().sum() == 0).sum()
    print(f"{num_complete} columns with no missing data.")

def check_duplicates_and_keys(interactions: pd.DataFrame, games: pd.DataFrame):
    print("\n=== Duplicates & key consistency ===")

    # Duplicated rows in interactions
    dup_inter = interactions.duplicated(subset=["user_id", "item_id"]).sum()
    print(f"Duplicated user_id-item_id pairs in train_interactions: {dup_inter}")

    # Duplicated item_id in games
    dup_games = games.duplicated(subset=["item_id"]).sum()
    print(f"Duplicated item_id in games: {dup_games}")

    # Foreign key check: all items in interactions exist in games
    inter_item_ids = set(interactions["item_id"].unique())
    game_item_ids = set(games["item_id"].unique())
    missing_in_games = inter_item_ids - game_item_ids
    print(f"Items in interactions not found in games: {len(missing_in_games)}")

def check_numeric_anomalies(games: pd.DataFrame, interactions: pd.DataFrame):
    print("\n=== Numeric anomalies ===")

    # Price in games
    if "price" in games.columns:
        # intentar convertir a numérico por si viene como string
        prices = pd.to_numeric(games["price"], errors="coerce")
        num_neg = (prices < 0).sum()
        print(f"games.price: {num_neg} negative values")
        print(f"games.price: min={prices.min()}, max={prices.max()}, 99th percentile={prices.quantile(0.99)}")

    # Playtime in interactions (si existe)
    if "playtime" in interactions.columns:
        playtime = interactions["playtime"]
        num_neg_pt = (playtime < 0).sum()
        print(f"train_interactions.playtime: {num_neg_pt} negative values")
        print(f"train_interactions.playtime: min={playtime.min()}, max={playtime.max()}, 99th percentile={playtime.quantile(0.99)}")

def check_distributions(games: pd.DataFrame, interactions: pd.DataFrame):
    print("\n=== Distributions & summary stats ===")

    # Basic stats for numeric columns in games
    num_cols_games = games.select_dtypes(include=["number"]).columns
    if len(num_cols_games) > 0:
        print("\nNumeric summary for games:")
        print(games[num_cols_games].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T)

    # Interactions per user and per item
    print("\nInteractions per user/item:")
    user_counts = interactions["user_id"].value_counts()
    item_counts = interactions["item_id"].value_counts()
    print(f"Users: mean={user_counts.mean():.2f}, median={user_counts.median()}, max={user_counts.max()}")
    print(f"Items: mean={item_counts.mean():.2f}, median={item_counts.median()}, max={item_counts.max()}")

    # Top publishers
    if "publisher" in games.columns:
        print("\nTop 10 publishers by number of games:")
        print(games["publisher"].value_counts().head(10))
