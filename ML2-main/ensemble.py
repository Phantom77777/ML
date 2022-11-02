import pandas as pd 
df = pd.read_csv("pima-indians-diabetes.csv")
df.head()

# data exploration
# finding if anycol has null values
df.isnull().sum()

df.describe()

# we can see min/max
# and accordingly decide for outlier detection

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
X = df.drop('Outcome', axis='columns')
y = df.Outcome

# standard scaling of data 
# because Glucose min-max is from 0-200
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# X_scaled will be a numpy array
# printing first two rows 
# X_scaled[:2]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=10)

# n-estimator: no of subsets
# oob score: out of bag, becayse of our random sampling strategy, we might miss some samples 
bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

bag_model.fit(X_train, y_train)
bag_model.score(X_test, y_test)

# only using standalone model of decision tree i,e without bagging
d_model = DecisionTreeClassifier()
d_model.fit(X_train, y_train)
d_model.score(X_test, y_test)

