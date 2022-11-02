import pandas as pd

dataset = pd.read_csv('churn.csv')[0:201]
dataset.head()

dataset.describe()

# Geography and gender are not in numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))
correl_matrix = dataset.corr().round(2)
sns.heatmap(data=correl_matrix, annot=True)
plt.show()

# Hence we can see that most important features for predicting class label 'Exited' are : Age, isActiveMember, HasCrCard, Gender, Geography_France

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

X = dataset[['Age', 'Geography_France',
             'IsActiveMember', 'HasCrCard', 'Gender']]
y = dataset['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=13)

dtree = DecisionTreeClassifier(random_state=13)
dtree.fit(X_train, y_train)
dtree.score(X_test, y_test)
y_pred_train = dtree.predict(X_train)
# y_pred_test = dtree.predict(X_test)


print(classification_report(y_train, y_pred_train))

# input and predicting
X_ip = list(map(int,input("Enter Age, isActiveMember, HasCrCard, Gender, Geography_France : ").split()))[:5]

print("predicted class : ")
dtree.predict([X_ip])[0]

# Visualise the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(40, 40))
plot_tree(dtree, feature_names=X_train.columns, filled=True, fontsize=20,
          rounded=True)
plt.show()
