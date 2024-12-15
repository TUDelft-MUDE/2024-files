# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
from func import *
import timeit

# %%

data = np.loadtxt('data.txt')
x = data[:,0]
y = data[:,1]

# %%
A = 
x_hat, y_hat = 
print(f"x_hat = {x_hat}")

# %%

funcs = [FD_1, FD_2, FD_3, FD_4]

assert np.allclose(funcs[0](x, y), funcs[1](x,y)), "FD_1 and FD_2 are not equal"
assert np.allclose(funcs[0](x, y), funcs[2](x,y)), "FD_1 and FD_3 are not equal"
assert np.allclose(funcs[0](x, y), funcs[3](x,y)), "FD_1 and FD_4 are not equal"

runtime = np.zeros(4)
for i in range(4):
    runtime[i] = timeit.timeit(lambda: funcs[i](x, y), number=1000)

print(f"runtimes: {runtime}")

# %% [markdown]

