import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

df = pd.read_csv("insurance.csv")
df.head()

plt.scatter(df.age, df.bought_insurance, marker='+', color='red')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], df.bought_insurance, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)
# y_predicted = model.predict(X_test)

# probability for a x to be in either of class
model.predict_proba(X_test)

model.score(X_test, y_test)

model.predict([[56]])
