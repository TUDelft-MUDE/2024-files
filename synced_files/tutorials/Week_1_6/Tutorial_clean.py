# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from func import *
import timeit

# ----------------------------------------
# import data 
data = np.loadtxt('data.txt')
x = data[:,0]
y = data[:,1]


# ----------------------------------------
A = #TODO: create the matrix A
x_hat, y_hat = #TODO: solve the system of equations
print(f"x_hat = {x_hat}")

# ----------------------------------------
# runtimes

funcs = [FD_1, FD_2, FD_3, FD_4]

assert np.allclose(funcs[0](x, y), funcs[1](x,y)), "FD_1 and FD_2 are not equal"
assert np.allclose(funcs[0](x, y), funcs[2](x,y)), "FD_1 and FD_3 are not equal"
assert np.allclose(funcs[0](x, y), funcs[3](x,y)), "FD_1 and FD_4 are not equal"

runtime = np.zeros(4)
for i in range(4):
    runtime[i] = timeit.timeit(lambda: funcs[i](x, y), number=1000)

print(f"runtimes: {runtime}")

