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
from city_solution import *

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

my_plan = Plan(coordinates, 2)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

my_plan.plot_coordinates();

# %% [markdown]

# %% [markdown]

# %%

my_plan.try_triangles([[1, 2, 3]])

# %% [markdown]

# %%

all_triangles = [[0, 3, 6],
                 [1, 2, 3],
                 [1, 2, 9],
                 [1, 3, 6],
                 [1, 4, 9],
                 [1, 4, 10],
                 [1, 6, 10],
                 [2, 7, 9],
                 [4, 5, 9],
                 [6, 8, 10]]
my_plan.try_triangles(all_triangles)

# %% [markdown]

# %%

my_plan.triangles = all_triangles
my_plan.plot_triangles();

# %% [markdown]

# %% [markdown]

# %%

my_plan.plot_shared_sides([[[1, 3], [3, 1]]]);

# %% [markdown]

# %%

sides = [[[3, 6], [0, 3]],
         [[1, 2], [1, 2]],
         [[1, 3], [1, 3]],
         [[1, 9], [2, 4]],
         [[9, 2], [2, 7]],
         [[1, 6], [3, 6]],
         [[1, 4], [4, 5]],
         [[9, 4], [4, 8]],
         [[1, 10], [5, 6]],
         [[10, 6], [6, 9]]]
my_plan.plot_shared_sides(sides);

# %% [markdown]

# %%

my_plan.shared_sides = sides
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

