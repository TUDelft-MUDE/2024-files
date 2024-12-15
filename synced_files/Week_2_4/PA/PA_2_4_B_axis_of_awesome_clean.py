# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
A = np.array([1, 20, 300, 1, 2, 3])
print(A)

# %% [markdown]

# %%
B = np.reshape(A, (2, 3))
print(B)

# %% [markdown]

# %%
B.mean(axis=0)

# %% [markdown]

# %%
B.mean(axis=1)

# %% [markdown]

# %%
B.std(axis=1)

# %% [markdown]

# %% [markdown]

# %%
coins = [46, 28, 16, 27,
         22, 24, 31, 12,
         32, 36, 12, 0,
         41, 27, 21, 26,
         21, 19, 18, 35,
         14, 34, 8, 0,
         53, 34, 23, 35,
         28, 26, 18, 13,
         12, 14, 34, 0]

# %%
np.set_printoptions(precision=1)
print(f'The average number of coins spent per month is:')
print('The average number of coins spent per month for each year is:')
print(f'The average number of coins spent each september:')
print(f'The average number of coins spent each january:')
print(f'Max coins spent in any month:')
print(f'Max coins spent in any year:')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(coins.reshape(-1));

# %% [markdown]

# %%
increasing_series = np.arange(1, 50)
plot_acf(increasing_series);

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
strong_autocorr_positive = np.array([YOUR_CODE_HERE])
plot_acf(strong_autocorr_positive);

# %% [markdown]

