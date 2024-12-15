# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]

# %% [markdown]

# %%
from city import *

# %% [markdown]

# %% [markdown]

# %%
coordinates = np.array(
              [[7.000, 1.000],
               [4.000, 2.732],
               [3.000, 1.000],
               [5.000, 1.000],
               [3.000, 4.464],
               [1.000, 4.464],
               [6.000, 2.732],
               [1.000, 1.000],
               [7.000, 4.464],
               [2.000, 2.732],
               [5.000, 4.464]])

my_plan = Plan(YOUR_CODE_HERE, YOUR_CODE_HERE)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %%
my_plan.try_triangles([[0, 1, 2]])

# %% [markdown]

# %%
all_triangles = [YOUR_CODE_HERE]
my_plan.try_triangles(all_triangles)

# %% [markdown]

# %%
YOUR_CODE_HERE
my_plan.plot_triangles();

# %% [markdown]

# %% [markdown]

# %%
my_plan.plot_shared_sides([[[9, 2], [3, 8]]]);

# %% [markdown]

# %%
sides = [YOUR_CODE_HERE] 
my_plan.plot_shared_sides(sides);

# %% [markdown]

# %%
YOUR_CODE_HERE
my_plan.plot_shared_sides();

# %% [markdown]

# %% [markdown]

# %%
my_plan.get_bar_coordinates()
my_plan.plot_everything();

# %% [markdown]

# %%
my_plan.get_kapsalon_coordinates()
my_plan.plot_everything();

# %% [markdown]

