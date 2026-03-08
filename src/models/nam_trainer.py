"""Training loop for NAM with early stopping, logging, and hyperparameter search."""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models.nam import NAM


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        improved = (
            score > self.best_score if self.mode == "max"
            else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class NAMTrainer:
    """Handles NAM training, validation, and hyperparameter search."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: dict,
        verbose: bool = True,
    ) -> tuple[NAM, dict]:
        """Train a single NAM model.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            config: Dict with keys: hidden_sizes, dropout, learning_rate,
                weight_decay, batch_size, max_epochs, early_stop_patience,
                output_penalty, feature_dropout.
            verbose: Whether to print progress.

        Returns:
            Trained NAM model (best checkpoint) and training history dict.
        """
        num_features = X_train.shape[1]

        model = NAM(
            num_features=num_features,
            hidden_sizes=config.get("hidden_sizes", [64, 64, 64]),
            dropout=config.get("dropout", 0.3),
            feature_dropout=config.get("feature_dropout", 0.0),
        ).to(self.device)

        # Class weight for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5
        )

        # Data loaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 256),
            shuffle=True,
        )

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        early_stopping = EarlyStopping(
            patience=config.get("early_stop_patience", 20), mode="max"
        )
        output_penalty = config.get("output_penalty", 0.0)

        history = {"train_loss": [], "val_auc": []}

        for epoch in range(config.get("max_epochs", 500)):
            # Training
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits, contributions = model(X_batch)
                loss = criterion(logits, y_batch)

                # Output penalty for smoother shape functions
                if output_penalty > 0:
                    penalty = sum(c.pow(2).mean() for c in contributions)
                    loss = loss + output_penalty * penalty

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            val_auc = roc_auc_score(y_val, val_probs)
            history["val_auc"].append(val_auc)

            scheduler.step(val_auc)

            if verbose and (epoch + 1) % 25 == 0:
                print(
                    f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}"
                )

            if early_stopping.step(val_auc, model):
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        model.load_state_dict(early_stopping.best_state)
        history["best_val_auc"] = early_stopping.best_score

        return model, history

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
        n_folds: int = 5,
        seed: int = 42,
        verbose: bool = True,
    ) -> tuple[dict, list[dict]]:
        """Random search over NAM hyperparameters with cross-validation.

        Args:
            X_train, y_train: Full training data (will be split into CV folds).
            n_trials: Number of random configurations to try.
            n_folds: Number of CV folds.
            seed: Random seed.
            verbose: Whether to print progress.

        Returns:
            Best config dict and list of all trial results.
        """
        rng = np.random.RandomState(seed)

        search_space = {
            "hidden_sizes": [[32, 32, 32], [64, 64, 64], [128, 128, 128]],
            "dropout": [0.1, 0.2, 0.3, 0.5],
            "learning_rate": [5e-4, 1e-3, 3e-3],
            "weight_decay": [0, 1e-5, 1e-4],
            "batch_size": [128, 256, 512],
            "output_penalty": [0.0, 1e-4, 1e-3],
            "feature_dropout": [0.0, 0.05, 0.1],
        }

        # Fixed params
        fixed = {"max_epochs": 500, "early_stop_patience": 20}

        all_results = []
        best_auc = -1
        best_config = None

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for trial in range(n_trials):
            config = {
                key: values[rng.randint(len(values))]
                for key, values in search_space.items()
            }
            config.update(fixed)

            if verbose:
                print(f"\nTrial {trial+1}/{n_trials}: {config}")

            fold_aucs = []
            t0 = time.time()

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]

                model, hist = self.train_model(
                    X_tr, y_tr, X_vl, y_vl, config, verbose=False
                )
                fold_aucs.append(hist["best_val_auc"])

            mean_auc = np.mean(fold_aucs)
            elapsed = time.time() - t0

            result = {
                "config": config,
                "mean_auc": mean_auc,
                "fold_aucs": fold_aucs,
                "time_s": elapsed,
            }
            all_results.append(result)

            if verbose:
                print(f"  Mean AUC: {mean_auc:.4f} ({elapsed:.1f}s)")

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_config = config

        if verbose:
            print(f"\nBest config (AUC={best_auc:.4f}): {best_config}")

        return best_config, all_results
