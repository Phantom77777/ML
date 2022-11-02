import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

plt.scatter(data.rm, data.medv)

def plot_regression_line(points, b):
    y_pred = b[0] + b[1] * points[0] + b[2] * points[1]

    plt.style.use('default')
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(points[0], points[1], points[2], color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(points[0], points[1], y_pred, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
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
    points = np.array(
        [[int(x) for x in map(int, input(f"Enter the values for variable {i + 1}:").split())] for i in range(3)])

    ##b = estimate_coef(points, size)

    print("Estimated coefficients:\nb0 = {}  \ \nb1 = {} \nb2 = {}".format(b[0], b[1], b[2]))

   ## plot_regression_line(points, b)


points = main()