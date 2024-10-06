import numpy as np
import matplotlib.pyplot as plt

def BLUE(A, y, Sigma=None):
    if Sigma is None:
        Sigma = np.eye(len(y))
    Sigma_inv = np.inv(Sigma)
    x_hat = np.inv(A.T @ Sigma_inv @ A) @ A.T @ Sigma_inv @ y
    return x_hat

def fit_line(A, x, y, Sigma=None):
    x_hat = BLUE(A, y, Sigma)
    y_hat = A @ x
    plt.plot(x, y, 'o', label='Data')
    plt.plot(x, y_hat, label='Fit')
    plt.title('Data and Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()
    return x_hat, y_hat

def create_A(x, n):
    A = np.array([x**i for i in range(n)])
    return A


# 4 different ways to calculate the forward difference
def FD_1(x, y):
    result = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    return result

def FD_2(x, y):
    result = [(y[i+1] - y[i]) / (x[i+1] - x[1]) for i in range(len(x) - 1)]
    return result

def FD_3(x, y):
    result = []
    for i in range(len(x) - 1):
        result = result.append((y[i+1] - y[i]) / (x[i+1] - x[i]))
    return result

def FD_4(x, y):
    a = -1*np.eye(len(x) - 1, len(x))
    a[range(len(x) - 1), range(1, len(x))] = 1
    dx = x[1:] - x[:-1]
    result = 1/dx * a @ y
    return result
