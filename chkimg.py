import os
from config import DATA_DIR
path = f"{DATA_DIR}/pl_market_values.csv"
print("Resolved path:", os.path.abspath(path))
print("Exists:", os.path.exists(path))