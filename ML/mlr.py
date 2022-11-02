import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

plt.scatter(data.rm, data.medv)


def multiple_linear_regression(x1, x2, y):
    sample_size = np.size(x1)

    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    y_mean = np.mean(y)

    SS_x1x1 = np.sum(x1 * x1) ** 2 - (np.sum(x1) * np.sum(x1)) / sample_size
    SS_x1x2 = np.sum(x1 * x2) ** 2 - (np.sum(x1) * np.sum(x2)) / sample_size
    SS_x1y = np.sum(x1 * y) ** 2 - (np.sum(x1) * np.sum(y)) / sample_size
    SS_x2x2 = np.sum(x2 * x2) ** 2 - (np.sum(x2) * np.sum(x2)) / sample_size
    SS_x2y = np, sum(x2 * y) ** 2 - (np.sum(x2) * np.sum(y)) / sample_size
    SS_yy = np.sum(y * y) ** 2 - (np.sum(y) * np.sum(y)) / sample_size

    b1 = (np.sum(x1 * y) * np.sum(x2 * x2) - np.sum(x1 * x2) * np.sum(x2 * y)) / (
                np.sum(x1 * x1) * np.sum(x2 * x2) - np.sum(x1 * x2) * np.sum(x1 * x2))
    b2 = (np.sum(x1 * x1) * np.sum(x2 * y) - np.sum(x1 * x2) * np.sum(x1 * y)) / (
                np.sum(x1 * x1) * np.sum(x2 * x2) - np.sum(x1 * x2) * np.sum(x1 * x2))
    b0 = y_mean - b1 * x1_mean - b2 * x2_mean

    print(b0, b1, b2)
    plt.scatter(x1, y, color="g")
    plt.scatter(x2, y, color="b")
    y_pred_x1 = b0 + b1 * x1 + b2 * x2
    # y_pred_x2 = b0 + b2*x2_mean + b1*x1_mean
    plt.plot(x1, y_pred_x1, color="r")
    plt.plot(x2, y_pred_x1, color="y")
    plt.Line2D(x1, y_pred_x1, color="r")
    plt.show()


multiple_linear_regression(data.rm, data.dis, data.medv)

