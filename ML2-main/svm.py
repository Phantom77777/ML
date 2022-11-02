# Actual SVM training 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# getting only feature columns in X
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creating SVM classifier
model = SVC()

# training model
model.fit(X_train, y_train) 

# finding accuracy
model.score(X_test, y_test)

# X_ip = list(map(float, input("sepal length(cm), sepal width (cm), petal length (cm), petal width (cm)").split()))[:4]

X_ip = [4.4, 2.1, 1.2, 0.5]
# model.predict([X_ip])[0]
model.predict([X_ip])[0]
# 4.4 2.1 1.2 0.5

