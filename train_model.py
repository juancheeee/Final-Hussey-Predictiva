

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



df = pd.read_csv("dataset_modeling.csv")



features = [
    'budget',
    'runtime',
    'release_year',
    'release_month',
    'num_companies',
    'main_genre',
    'vote_average',
    'vote_count',
    'popularity'
]

X = df[features]
y = df['rentable']




X = pd.get_dummies(X, drop_first=True)


X.columns = (
    X.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.replace('>', '', regex=False)
    .str.replace("'", '', regex=False)
)

X = X.astype(float)
X = X.loc[:, ~X.columns.duplicated()]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


joblib.dump(model, "modelo_final.pkl")
joblib.dump(X.columns.tolist(), "columnas_modelo.pkl")

print("Modelo y columnas guardados correctamente.")
