# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]

# %%
my_dict = {}
type(my_dict)

# %% [markdown]

# %%
my_dict = {'key': 5}

# %% [markdown]

# %%
my_dict['key']

# %% [markdown]

# %%
my_dict['array'] = [34, 634, 74, 7345]
my_dict['array'][3]

# %% [markdown]

# %%
shell = ['chick']
shell = {'shell': shell}
shell = {'shell': shell}
shell = {'shell': shell}
nest = {'egg': shell}
nest['egg']['shell']['shell']['shell'][0]

# %% [markdown]

# %% [markdown]

# %%
new_dict = {'names': ['Gauss', 'Newton', 'Lagrange', 'Euler'],
            'birth year': [1777, 1643, 1736, 1707]}

YOUR_CODE_HERE

# %% [markdown]

# %%
df = pd.DataFrame(new_dict)

YOUR_CODE_HERE

# %% [markdown]

# %%
guess = df.loc[df['birth year'] <= 1700, 'names']
print(guess)

# %%
YOUR_CODE_HERE

# %% [markdown]

# %%
print(type(df.loc[df['birth year'] <= 1700, 'names']))
print(type(df.loc[df['birth year'] <= 1700, 'names'].values))
print('The value in the series is an ndarray with first item:',
      df.loc[df['birth year'] <= 1700, 'names'].values[0])

# %% [markdown]

# %%
df.head()

# %% [markdown]

# %%
df.describe()

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
df = YOUR_CODE_HERE
YOUR_CODE_HERE.head()

# %% [markdown]

# %% [markdown]

# %%
names_of_earth_dams = YOUR_CODE_HERE
print('The earth fill dams are:', names_of_earth_dams)

# %% [markdown]

# %% [markdown]

# %%
df_earth = YOUR_CODE_HERE
df_earth.YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

