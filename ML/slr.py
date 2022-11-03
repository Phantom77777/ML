import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

plt.scatter(data.rm, data.medv)

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