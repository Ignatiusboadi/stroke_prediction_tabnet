from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

patients_data = pd.read_csv('data/smote_patients_info.csv')
patients_data['Set'] = np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1], size=[patients_data.shape[0]])

cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status",
            "stroke"]
cat_ind = [i for i, f in enumerate(patients_data.columns) if f in cat_cols]
cat_dim = [patients_data[f].nunique() for f in patients_data.columns if f in cat_cols]
