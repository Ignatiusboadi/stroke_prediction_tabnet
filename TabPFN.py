from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

import numpy as np
import pandas as pd


patients_data = pd.read_csv('data/smote_patients_info.csv')
print(patients_data['stroke'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(patients_data.drop(columns='stroke'),
                                                    patients_data['stroke'], test_size=0.5)

tab_model = TabPFNClassifier()
tab_model.fit(X_train, y_train)

preds = tab_model.predict_proba(X_test)

test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

y_probs = preds[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(best_threshold)
y_pred = (y_probs >= best_threshold).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=list(patients_data.columns),
    class_names=[1, 0],
    mode='classification'
)

exp_0 = lime_explainer.explain_instance(data_row=X_test.iloc[2].values, predict_fn=tab_model.predict_proba)

print(exp_0.as_list())
