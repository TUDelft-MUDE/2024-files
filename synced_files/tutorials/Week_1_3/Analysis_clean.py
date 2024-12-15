
def add(a, b):
    result = a+b
    return result

def gen_xhat(A, y):
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ y
    return x_hat

a = 1
b = 2
result = add(a, b)

result

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)  # 'skip_header=1' skips the first row (header)

t = data[:,0]
y = data[:,1]
n_rows = data.shape[0]
n_cols = data.shape[1]

plt.plot(t, y,'o')
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

one_vector = np.ones(n_rows)

print(one_vector+ t)

A = np.column_stack((one_vector, t))

x_hat = gen_xhat(A, y)
y_hat = A @ x_hat

plt.plot(t, y,'o')
plt.plot(t, y_hat)
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

A_new = np.column_stack((one_vector, t, t**2))
x_hat_new = gen_xhat(A_new, y)
y_hat_new = A_new @ x_hat_new
plt.plot(t, y,'o')
plt.plot(t, y_hat_new)
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

e_hat = y-y_hat_new

