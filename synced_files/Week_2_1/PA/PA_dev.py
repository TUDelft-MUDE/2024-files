# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: mude-base
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from city import *

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

# %%
my_plan = Plan(coordinates)
my_plan.define_triangles()
my_plan.get_all_sides();

# %%
my_plan.try_triangles()
# my_plan.plot_triangles(triangle_id=-10);

# %%
my_plan.refine_mesh()
my_plan.plot_coordinates();

# %%
my_plan = Plan(coordinates)
my_plan.plot_coordinates();

# %%
print(my_plan.side_length)

# %%
triangles = [[7, 9, 8]]
my_plan = Plan(coordinates)
my_plan.try_triangles([[7, 9, 2],
                       [9, 1, 2]], triangle_id=range(2))



# %%
my_plan.define_triangles()

# %%
my_plan.plot_triangles()
print(len(my_plan.triangles))
my_plan.triangles

# %%
my_plan.define_shared_sides()
print(my_plan.shared_sides)
my_plan.plot_shared_sides([my_plan.shared_sides[2]])

# %%
my_plan.get_all_sides()

# %%

# %%
len([[2, 3, 5]])

# %%
triangles = [[7, 9, 2],
             [9, 1, 2],
             [1, 3, 6],
             [3, 6, 0],
             [5, 9, 4],
             [9, 4, 1],
             [4, 1, 10],
             [1, 10, 6],
             [10, 6, 8],
             [1, 2, 3]]
my_plan = Plan(coordinates, triangles)
# my_plan.plot_triangle(9)
my_plan.plot_triangle(range(10))
# my_plan.plot_triangle([2, 7])
my_plan.check_triangles()

# %%
# my_plan.define_shared_sides([[[9, 2], [0, 1]]])
sides = [[[9, 2], [0, 8]]]
my_plan.define_shared_sides(sides)
my_plan.plot_shared_sides(range(len(sides)));

# %%
my_plan.get_kapsalon_coordinates()
my_plan.get_bar_coordinates()
my_plan.plot_everything();


# %%

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3:</b>   
#
# Run the cell below to check your work, but don't change anything. If the cell runs without error, you will pass the assignment once you commit it and push it to GitHub.
#
# </p>
# </div>

# %%
x=1
assert (
    x==1), (
        'cool')

# %%
import numpy as np
import sys
# np.set_printoptions()
np.set_printoptions(precision=3,
                    threshold=sys.maxsize,
                    floatmode='fixed')
x = np.array([[1.312323,2.,3.],[1.,2.,3.]])
print(x)

# %% [markdown]
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
