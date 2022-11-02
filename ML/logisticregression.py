
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

from ML import X_train, y_train, X_test, y_test

lr = LogisticRegression(random_state=3)
lr.fit(X_train, y_train)
y_pred_Train = lr.predict(X_train)
y_pred_Test = lr.predict(X_test)
print(classification_report(y_train, y_pred_Train))
print(classification_report(y_test, y_pred_Test))
x_pred = {
    'sepal.length': 5.1,
    'sepal.width': 3.5,
    'petal.length': 1.4,
    'petal.width': 0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {lr.predict(df)[0]}")

