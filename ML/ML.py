# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

# from sklearn.datasets import load_boston
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]
# # features = boston.data[:,0:2]
# # target = boston.target
# regression = LinearRegression()
# model = regression.fit(data,target)

# print(model.intercept_)
# model.coef_

plt.scatter(data.rm, data.medv)
# lstat, dis and rm are the most correlated variables with medv

from turtle import color


def simple_linear_regression(x, y):
    sample_size = np.size(x)
    # Calculating means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    SS_xy = np.sum(x*y) - sample_size*x_mean*y_mean
    SS_xx = np.sum(x*x) - sample_size*x_mean*x_mean

    # Calculating regression coefficients 
    b_1 = SS_xy/SS_xx
    b_0 = y_mean - b_1*x_mean

    plt.scatter(x,y)

    y_pred = b_0 + b_1*x
    plt.plot(x,y_pred,color="r")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

simple_linear_regression(data.rm, data.medv)

def multiple_linear_regression(x1,x2,y):
    sample_size = np.size(x1)

    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    y_mean = np.mean(y)

    SS_x1x1 = np.sum(x1*x1)**2 - (np.sum(x1)*np.sum(x1))/sample_size
    SS_x1x2 = np.sum(x1*x2)**2 - (np.sum(x1)*np.sum(x2))/sample_size
    SS_x1y = np.sum(x1*y)**2 - (np.sum(x1)*np.sum(y))/sample_size
    SS_x2x2 = np.sum(x2*x2)**2 - (np.sum(x2)*np.sum(x2))/sample_size
    SS_x2y = np,sum(x2*y)**2 - (np.sum(x2)*np.sum(y))/sample_size
    SS_yy = np.sum(y*y)**2 - (np.sum(y)*np.sum(y))/sample_size

    b1 = (np.sum(x1*y)*np.sum(x2*x2) - np.sum(x1*x2)*np.sum(x2*y))/(np.sum(x1*x1)*np.sum(x2*x2) - np.sum(x1*x2)*np.sum(x1*x2))
    b2 = (np.sum(x1*x1)*np.sum(x2*y) - np.sum(x1*x2)*np.sum(x1*y))/(np.sum(x1*x1)*np.sum(x2*x2) - np.sum(x1*x2)*np.sum(x1*x2))
    b0 = y_mean - b1*x1_mean - b2*x2_mean

    print(b0,b1,b2)
    plt.scatter(x1,y,color ="g")
    plt.scatter(x2,y,color="b")
    y_pred_x1 = b0 + b1*x1 + b2*x2
    # y_pred_x2 = b0 + b2*x2_mean + b1*x1_mean
    plt.plot(x1,y_pred_x1,color="r")
    plt.plot(x2,y_pred_x1,color="y")
    plt.Line2D(x1,y_pred_x1,color="r")
    plt.show()
multiple_linear_regression(data.rm,data.dis,data.medv)    
    


import numpy as np
import matplotlib.pyplot as plt
  
def estimate_coef(points, size):
    
    x1_mean = np.mean(points[0])
    x2_mean = np.mean(points[1])
    y_mean = np.mean(points[2])
  
    # calculating cross-deviation and deviation about x
    x1_square = np.sum(np.square(points[0]))
    x2_square = np.sum(np.square(points[1]))
    
    sum = 0 
    for i in range(size):
      sum = sum + (points[0][i]*points[2][i])
    x1_y = np.array(sum)

    sum = 0 
    for i in range(size):
      sum = sum + (points[1][i]*points[2][i])
    x2_y = np.array(sum)

    sum = 0 
    for i in range(size):
      sum = sum + (points[0][i]*points[1][i])
    x1_x2 = np.array(sum)

    b1 = (x2_square*x1_y - x1_x2*x2_y)/(x1_square*x2_square - x1_x2**2) 
    b2 = (x1_square*x2_y-x1_x2*x1_y)/(x1_square*x2_square-x1_x2**2)
    b0 = y_mean - b1*x1_mean - b2*x2_mean
    return (b0, b1, b2)


def plot_regression_line(points, b):
    y_pred = b[0] + b[1]*points[0] + b[2]*points[1]
  
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(points[0], points[1], points[2], color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(points[0], points[1], y_pred, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_zlabel('Y', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)


    fig.tight_layout()
      

def main():
    size = int(input("Size of array "))
    points = np.array([[int(x) for x in map(int, input(f"Enter the values for variable {i+1}:").split())] for i in range(3)])
    
    b = estimate_coef(points, size)

    print("Estimated coefficients:\nb0 = {}  \
          \nb1 = {} \nb2 = {}".format(b[0], b[1], b[2]))
  
    plot_regression_line(points, b)
    points = main()
 
##!pip install xgboost

 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.decomposition import PCA

 
data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
data

 
# THIS IS ONLY REQUIRED FOR XGBOOST IS NOT REQUIRED FOR ANY OTHER MODEL.
le = LabelEncoder()
data['variety'] = le.fit_transform(data['variety'])
data

 
X = data[['sepal.length','sepal.width','petal.length','petal.width']]
y = data['variety']

 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state= 3)

 
# # Logistic Regression

 
lr = LogisticRegression(random_state=3)
lr.fit(X_train,y_train)
y_pred_Train = lr.predict(X_train)
y_pred_Test = lr.predict(X_test)
print(classification_report(y_train,y_pred_Train))
print(classification_report(y_test,y_pred_Test))
x_pred = {
    'sepal.length':5.1,
    'sepal.width':3.5,
    'petal.length':1.4,
    'petal.width':0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {lr.predict(df)[0]}")

 
# # Decision Tree Classifier

 

dtree = DecisionTreeClassifier(random_state=3)
dtree.fit(X_train,y_train)
y_pred_Train = dtree.predict(X_train)
y_pred_Test = dtree.predict(X_test)
print(classification_report(y_train,y_pred_Train))
print(classification_report(y_test,y_pred_Test))
x_pred = {
    'sepal.length':5.1,
    'sepal.width':3.5,
    'petal.length':1.4,
    'petal.width':0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {dtree.predict(df)[0]}")


 
# # SVC

 
svc = SVC(random_state=3)
svc.fit(X_train,y_train)
y_pred_Train = svc.predict(X_train)
y_pred_Test = svc.predict(X_test)
print(classification_report(y_train,y_pred_Train))
print(classification_report(y_test,y_pred_Test))
x_pred = {
    'sepal.length':5.1,
    'sepal.width':3.5,
    'petal.length':1.4,
    'petal.width':0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {svc.predict(df)[0]}")

 
# # XGBoost

 
xgBOOOOOOOOSt = xgb.XGBClassifier()
xgBOOOOOOOOSt.fit(X_train,y_train)
y_pred_Train = svc.predict(X_train)
y_pred_Test = svc.predict(X_test)
print(classification_report(y_train,y_pred_Train))
print(classification_report(y_test,y_pred_Test))
x_pred = {
    'sepal.length':5.1,
    'sepal.width':3.5,
    'petal.length':1.4,
    'petal.width':0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {xgBOOOOOOOOSt.predict(df)[0]}")

 
# # Random Forest

 
rf = RandomForestClassifier(random_state =3)
rf.fit(X_train,y_train)
y_pred_Train = svc.predict(X_train)
y_pred_Test = svc.predict(X_test)
print(classification_report(y_train,y_pred_Train))
print(classification_report(y_test,y_pred_Test))
x_pred = {
    'sepal.length':5.1,
    'sepal.width':3.5,
    'petal.length':1.4,
    'petal.width':0.2
}
df = pd.DataFrame.from_dict([x_pred])
print(f"Prediction is: {rf.predict(df)[0]}")

 
# # PCA

 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
# principalDf
finalDf = pd.concat([principalDf,y],axis = 1)
finalDf
