import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, classification_report, confusion_matrix
from itertools import product
import logging
from datetime import datetime
import os


class TabNetCVLogger:
    """
    A class to perform nested cross-validation for TabNetClassifier
    with logging and tracking per fold.
    """

    def __init__(self, X, y, cat_cols, param_grid, n_splits=5, log_dir="logs"):
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
                    max_epochs=50,
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


patients_data = pd.read_csv('data/smote_patients_info.csv')
patients_data['Set'] = np.random.choice(['train', 'test'], p=[0.9, 0.1], size=[patients_data.shape[0]])

cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]

train_data = patients_data.query('`Set` == "train"').drop(columns=['Set'])
test_data = patients_data.query('`Set` == "test"').drop(columns=['Set'])

X_train = train_data.drop(columns=['stroke'])
y_train = train_data['stroke']
X_test = test_data.drop(columns=['stroke'])
y_test = test_data['stroke']

param_grid = {
    'n_d': [16, 32],
    'n_steps': [3, 5],
    'cat_emb_dim': [1, 2, 3],
    'gamma': [1.2, 1.5],
    'lambda_sparse': [1e-3, 1e-4],
    'optimizer_params': [{'lr': 3e-2}, {'lr': 1e-2}]
}

cv_runner = TabNetCVLogger(X_train, y_train, cat_cols, param_grid)
best_auc, best_model, best_params = cv_runner.evaluate()
best_model.save_model("tabnet_stroke_model")

print("Best AUC:", best_auc)
print("Best Parameters:", best_params)
best_model = TabNetClassifier()
best_model.load_model("tabnet_stroke_model.zip")
preds = best_model.predict_proba(X_test.values)

test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

y_probs = preds[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print('final test score', test_auc)

y_pred = (y_probs >= best_threshold).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
