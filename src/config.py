import os
from pathlib import Path

if "STEAM_DATA_DIR" in os.environ:
    RAW_DATA_DIR = Path(os.environ["STEAM_DATA_DIR"])
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"

INTERACTIONS_PATH = RAW_DATA_DIR / "train_interactions.csv"
GAMES_PATH = RAW_DATA_DIR / "games.csv"