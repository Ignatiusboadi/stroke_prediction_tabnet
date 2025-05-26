import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from itertools import product
import logging
from datetime import datetime
import os


class TabNetCVLogger:
    """
    A class to perform nested cross-validation for TabNetClassifier
    with logging and tracking per fold.
    """

    def __init__(self, X, y, cat_cols, param_grid, n_splits=3, log_dir="logs"):
        """
        Initializes the cross-validation wrapper with logging support.

        Parameters:
        - X: Feature DataFrame
        - y: Target labels
        - cat_cols: List of categorical column names
        - param_grid: Dictionary of hyperparameters to search
        - n_splits: Number of cross-validation folds
        - log_dir: Directory to store log files
        """
        self.X = X
        self.y = y
        self.cat_cols = cat_cols
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.cat_idxs, self.cat_dims = self._compute_cat_info()
        self.all_params = self._expand_grid()
        self._setup_logging(log_dir)

    def _compute_cat_info(self):
        cat_idxs = [i for i, col in enumerate(self.X.columns) if col in self.cat_cols]
        cat_dims = [self.X[col].nunique() for col in self.cat_cols]
        return cat_idxs, cat_dims

    def _expand_grid(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _setup_logging(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"tabnet_cv_{timestamp}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.info("Logging started")

    def evaluate(self):
        best_auc = 0
        best_model = None
        best_params = None
        X_array = self.X.values
        y_array = self.y.values
        outer_cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for params in self.all_params:
            aucs = []
            logging.info(f"Evaluating parameters: {params}")
            for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_array, y_array)):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]

                model = TabNetClassifier(
                    cat_idxs=self.cat_idxs,
                    cat_dims=self.cat_dims,
                    cat_emb_dim=params['cat_emb_dim'],
                    n_d=params['n_d'],
                    n_a=params['n_d'],
                    n_steps=params['n_steps'],
                    gamma=params['gamma'],
                    lambda_sparse=params['lambda_sparse'],
                    optimizer_params=params['optimizer_params'],
                    verbose=0
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_name=['valid'],
                    max_epochs=100,
                    patience=10,
                    batch_size=256,
                    virtual_batch_size=64,
                    num_workers=0,
                    drop_last=False,
                    eval_metric=['auc']
                )

                preds = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, preds)
                aucs.append(auc)
                logging.info(f"Fold {fold + 1}: AUC = {auc:.4f}")

            mean_auc = np.mean(aucs)
            logging.info(f"Mean AUC for params {params}: {mean_auc:.4f}")

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_model = model
                best_params = params

        logging.info(f"Best AUC: {best_auc:.4f} with params: {best_params}")
        return best_auc, best_model, best_params

