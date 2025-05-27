import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from itertools import product
import logging
from datetime import datetime
import os


class XGBoostCVLogger:
    def __init__(self, X, y, param_grid, n_splits=5, log_dir="logs"):
        self.X = X
        self.y = y
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.all_params = self._expand_grid()
        self._setup_logging(log_dir)

    def _expand_grid(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _setup_logging(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"xgboost_cv_{timestamp}.log")
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

                model = XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    **params
                )

                model.fit(X_train, y_train)

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


# Load your data
data = pd.read_csv('data/smote_patients_info.csv')
data['Set'] = np.random.choice(['train', 'test'], p=[0.9, 0.1], size=[data.shape[0]])

train_data = data.query('Set == "train"').drop(columns=['Set'])
test_data = data.query('Set == "test"').drop(columns=['Set'])

X_train = train_data.drop(columns=['stroke'])
y_train = train_data['stroke']
X_test = test_data.drop(columns=['stroke'])
y_test = test_data['stroke']

param_grid = {
    'max_depth': [3, 4],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [50, 100],
    'alpha': [0, 10]
}

cv_runner = XGBoostCVLogger(X_train, y_train, param_grid)
best_auc, best_model, best_params = cv_runner.evaluate()

# Evaluate on test set
preds = best_model.predict_proba(X_test.values)
test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

# Optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, preds[:, 1])
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best AUC (CV):", best_auc)
print("Best Parameters:", best_params)
print("Final Test AUC:", test_auc)

y_pred = (preds[:, 1] >= best_threshold).astype(int)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
