from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('data/patients_info.csv')

print('data info after reading data')
data.info()
data = data.drop(columns=['id'])
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

print('data info after filling in missing values')
data.info()

cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status",
            "stroke"]
cont_cols = ["age", "avg_glucose_level", "bmi"]

for col in cat_cols:
    print(data[col].unique())

print(data['gender'].value_counts())
data = data.drop(index=data.query('`gender` == "Other"').index)
print('unique values of gender', data.gender.unique())

print(data[cont_cols].describe())

data = data.drop(index=data.query('`bmi` > 47').index)

label_encoder = LabelEncoder()
for col in cat_cols:
    data[col] = label_encoder.fit_transform(data[col])

print(data['stroke'].value_counts())

sampler = SMOTE(random_state=1)
X = data.drop(columns=['stroke'])
y = data['stroke']

X, y = sampler.fit_resample(X, y.values)
y = pd.DataFrame({'stroke': y})

data = pd.concat([X, y], axis=1).sample(frac=1)
print(data['stroke'].value_counts())
print(data.shape)

data.to_csv('data/smote_patients_info.csv', index=False)
