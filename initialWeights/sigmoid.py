
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1/(1 + np.exp(-x))
# dy=y*(1-y)
    return y

def sigmoid_differential_coefficient(x):
    y = 1 / (1 + np.exp(-x))
    dy = y * (1 - y)
    return y, dy


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-5, 5, 0.2)
    y, dy = sigmoid_differential_coefficient(x)
    plt.plot(x, y, label='$\\frac{1}{1+e^{-x}} }$')
    # plt.plot(x, dy, label='differential coefficient of sigmoid')
    plt.plot(x, dy, label='$\sigma(x)(1-\sigma(x))$')
    plt.xlabel('x Axis')
    plt.ylabel('y Axis')
    plt.title('Sigmoid Function')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()


