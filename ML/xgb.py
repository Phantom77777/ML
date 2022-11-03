import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import xgboost as xgb


data = pd.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
data

# THIS IS ONLY REQUIRED FOR XGBOOST IS NOT REQUIRED FOR ANY OTHER MODEL.
le = LabelEncoder()
data['variety'] = le.fit_transform(data['variety'])
data

X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

svc = SVC(random_state=3)
svc.fit(X_train, y_train)

xgBOOOOOOOOSt = xgb.XGBClassifier()
xgBOOOOOOOOSt.fit(X_train, y_train)
y_pred_Train = svc.predict(X_train)
y_pred_Test = svc.predict(X_test)
print(classification_report(y_train, y_pred_Train))
print(classification_report(y_test, y_pred_Test))
x_pred = {
    'sepal.length': 5.1,
    'sepal.width': 3.5,
    'petal.length': 1.4,
    'petal.width': 0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {xgBOOOOOOOOSt.predict(df)[0]}")

