import pandas as pd
from .config import INTERACTIONS_PATH, GAMES_PATH

def load_interactions():
    """Load the interactions data from the CSV file."""
    print(f"Loading interactions data from {INTERACTIONS_PATH}")
    df = pd.read_csv(INTERACTIONS_PATH)
    print(f"Loaded {len(df)} interactions.")
    return df

def load_games():
    """Load the games data from the CSV file."""
    print(f"Loading games data from {GAMES_PATH}")
    df = pd.read_csv(GAMES_PATH)
    print(f"Loaded {len(df)} games.")
    return df