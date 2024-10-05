## Some small bugs for you to find!
import numpy as np
import matplotlib.pylab as plt

# Part 1
# ======
# Create a matrix with 5's on the diagonal and 1's on the diagonal
# below the main diagonal
A = np.zeros((5, 5))
np.fill_diagonal(A, 5)
A[range(3), range(1, 5)] = 1

# Part 2
# ======
# We want to compute the exponent of x = 0, 4, 8, 12, 16, 20 
x = range(0, 21, 4)
y = np.exp(x)

# then we want to change the first value of x to 1 instead of 0
x[0] = 1
y = np.exp(x)

assert x[0]==1, "The first value of x should be 1"
assert y[0]==np.exp(1), "The first value of y should be exp(1)"

# Part 3
# ======
# Some arbitrary script to prepare arrays for plotting
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = np.linspace(0, 10, 11)

print(a.size())

for i in range(0, a.size):
    a[i] = a[i] +a[i-1]

c = a+b