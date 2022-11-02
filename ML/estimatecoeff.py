import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

plt.scatter(data.rm, data.medv)
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
        sum = sum + (points[0][i] * points[2][i])
    x1_y = np.array(sum)

    sum = 0
    for i in range(size):
        sum = sum + (points[1][i] * points[2][i])
    x2_y = np.array(sum)

    sum = 0
    for i in range(size):
        sum = sum + (points[0][i] * points[1][i])
    x1_x2 = np.array(sum)

    b1 = (x2_square * x1_y - x1_x2 * x2_y) / (x1_square * x2_square - x1_x2 ** 2)
    b2 = (x1_square * x2_y - x1_x2 * x1_y) / (x1_square * x2_square - x1_x2 ** 2)
    b0 = y_mean - b1 * x1_mean - b2 * x2_mean
    return (b0, b1, b2)


