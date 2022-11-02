import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

df = pd.read_csv('cars.csv')
df.head()

# convert transmission and owner fetures into number by using labelEncoder

print(df['transmission'].unique())
print(df['owner'].unique())

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

le = LabelEncoder()
oe = OrdinalEncoder()

df['transmission'] = le.fit_transform(df['transmission'])

# oe = OrdinalEncoder(categories=[['First Owner','Second Owner', 'Fourth & Above Owner','Third Owner','Test Drive Car']])

# notice double brackets
# df[['owner']] = oe.fit_transform(df[['owner']])
df['owner'] = le.fit_transform(df['owner'])


df['owner'].unique()

X = df[['year_bought', 'km_driven', 'transmission', 'owner']]
y = df['selling_price']
X.insert(0, 'x0', 1)
X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

import numpy as np

def calculate_theta(X, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),y)

theta = calculate_theta(X_train, y_train)

def predict(theta, X):
    return np.matmul(X, theta)

y_pred = predict(theta, X_test.values)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
results

# taking ip and predicting
# X_ip = [1] + list(map(int,input("Enter year_bought, km_driven, transmission, owner : ").split()))[:4]
X_ip = [1] + [23, 24, 25, 3]
# X_ip = np.array(X_ip).reshape(1, len(X_ip))
print('Predicted Selling Price of car is : ', predict(theta, X_ip))


# Multiple Linear Regression - Actual vs Predicted
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4))
ax = plt.axes()
ax.scatter(range(len(y_test)), y_test)
ax.scatter(range(len(y_test)), y_pred)
ax.ticklabel_format(style='plain')
plt.legend(['Actual', 'Predicted'])
plt.show()
