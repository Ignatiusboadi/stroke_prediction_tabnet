from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

patients_data = pd.read_csv('data/smote_patients_info.csv')
patients_data['Set'] = np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1], size=[patients_data.shape[0]])


print(patients_data.columns)