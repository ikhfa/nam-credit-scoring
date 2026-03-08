"""Download the Taiwan Credit Card Default dataset from UCI ML Repository."""

import os
import pandas as pd


def download_dataset(save_path: str = "data/raw/credit_default.csv") -> pd.DataFrame:
    """Download UCI dataset ID 350 and save as CSV.

    Args:
        save_path: Path to save the raw CSV file.

    Returns:
        DataFrame with features and target combined.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}, loading from disk.")
        return pd.read_csv(save_path)

    print("Downloading Taiwan Credit Card Default dataset from UCI...")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    # Map generic names (X1..X23, Y) to descriptive names using metadata
    var_info = dataset.variables
    rename_map = dict(zip(var_info["name"], var_info["description"]))
    rename_map["Y"] = "default"

    df = pd.concat([X, y], axis=1)
    df.rename(columns=rename_map, inplace=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path} ({len(df)} rows, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    download_dataset()
