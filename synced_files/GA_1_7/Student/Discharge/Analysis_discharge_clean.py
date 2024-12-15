# ---

# ---

# %% [markdown]

# %% [markdown] id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945"

# %% [markdown] id="d33f1148-c72b-4c7e-bca7-45973b2570c5"

# %% id="4fc6e87d-c66e-43df-a937-e969acc409f8"
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

# %% [markdown]

# %% [markdown]

# %%

h, u = np.genfromtxt('dataset_hu.csv', delimiter=",", unpack=True, skip_header=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(h,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Water depth, h (m)')
ax[0].grid()

ax[1].plot(u,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water velocity, u (m/s)')
ax[1].grid()

# %%

print(stats.describe(h))

# %%

print(stats.describe(u))

# %% [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def ecdf(YOUR_INPUT:
    
    return YOUR_OUTPUT

# %%

# %% [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"

# %% [markdown]

# %% [markdown]

# %%

# %% [markdown]

# %% [markdown]

# %%

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

rs_h = 
rs_u = 

rs_q = 

q = 

# %% [markdown]

# %% [markdown]

# %%
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_h, rs_u, 40, 'k', label = 'Simulations')
axes.scatter(h, u, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

# %%

# %% [markdown]

