"""Standalone NAM training script."""

import os
import sys
import json
import random
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.split import load_splits
from src.models.nam_trainer import NAMTrainer


def main(config_path: str = "configs/default.yaml", skip_search: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits(config["data"]["processed_dir"])

    X_train = splits["X_train"].values
    X_cal = splits["X_cal"].values
    y_train = splits["y_train"]
    y_cal = splits["y_cal"]

    trainer = NAMTrainer(device=device)

    if skip_search:
        best_config = config["nam"]
    else:
        print("Running hyperparameter search...")
        best_config, results = trainer.hyperparameter_search(
            X_train, y_train, n_trials=50, n_folds=5, seed=seed
        )
        os.makedirs("results/logs", exist_ok=True)
        with open("results/logs/nam_search_results.json", "w") as f:
            json.dump(
                [{"config": r["config"], "mean_auc": r["mean_auc"]} for r in results],
                f, indent=2, default=str,
            )

    print(f"\nTraining final model with: {best_config}")
    model, history = trainer.train_model(
        X_train, y_train, X_cal, y_cal, best_config, verbose=True
    )

    # Save model
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/nam_best.pt")
    with open("results/models/nam_config.json", "w") as f:
        json.dump(best_config, f, indent=2, default=str)

    print(f"\nBest validation AUC: {history['best_val_auc']:.4f}")
    print("Model saved to results/models/nam_best.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()
    main(args.config, args.skip_search)
