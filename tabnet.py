from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

patients_data = pd.read_csv('data/smote_patients_info.csv')
patients_data['Set'] = np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1], size=[patients_data.shape[0]])

cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
cat_ind = [i for i, f in enumerate(patients_data.columns) if f in cat_cols]
cat_dim = [patients_data[f].nunique() for f in patients_data.columns if f in cat_cols]

train = patients_data.query('`Set` == "train"').drop(columns=['Set'])
valid = patients_data.query('`Set` == "valid"').drop(columns=['Set'])
test = patients_data.query('`Set` == "test"').drop(columns=['Set'])

y_train = train['stroke'].values
X_train = train.drop(columns=['stroke']).values
y_valid = valid['stroke'].values
X_valid = valid.drop(columns=['stroke']).values
y_test = test['stroke'].values
X_test = test.drop(columns=['stroke']).values

model = TabNetClassifier(cat_idxs=cat_ind, cat_dims=cat_dim, cat_emb_dim=3)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_name=['train', 'valid'],
          max_epochs=30, patience=20, batch_size=256, virtual_batch_size=64, num_workers=0, drop_last=False,
          eval_metric=['auc']
          )
